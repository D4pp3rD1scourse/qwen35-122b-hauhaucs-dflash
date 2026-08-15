#!/usr/bin/env python3
"""Streaming OpenAI-compatible proxy with conservative DFlash routing."""

import argparse
import http.client
import json
import re
import threading
import time
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


HOP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade",
}
CODE_RE = re.compile(r"```|\b(def|class|function|refactor|python|javascript|typescript|sql|json schema|code)\b", re.I)
REASONING_RE = re.compile(r"\b(calculate|solve|step by step|how many|find the|derive|prove|equation|percent|percentage)\b", re.I)
PROSE_RE = re.compile(r"\b(write|draft|compose|scene|story|poem|description|describe|rewrite|tone|paragraph|essay|prose)\b", re.I)
STRUCTURED_RE = re.compile(
    r"\b(?:return|output|respond|provide|emit|give)\b[^\n]{0,100}\bjson\b|"
    r"\bjson\s+(?:object|schema|with\s+keys?|containing|only|format)\b|"
    r"\b(?:valid|compact|strict)\s+json\b",
    re.I,
)
POLICIES = {
    "acceptance": {"reasoning": 1, "structured": 1, "code": 3, "prose": 0},
    "hybrid": {"reasoning": 1, "structured": 3, "code": 4, "prose": 0},
}


def prompt_text(body):
    parts = []
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        parts.append(prompt)
    elif isinstance(prompt, list):
        parts.extend(item for item in prompt if isinstance(item, str))
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
    return "\n".join(parts)


def classify(body, header_mode="auto", policy="acceptance"):
    """Return `(n_max, reason)` for an OpenAI-compatible request body."""
    mode = str(body.pop("dflash_mode", header_mode) or "auto").lower()
    if "speculative.n_max" in body:
        value = int(body["speculative.n_max"])
        return max(0, min(4, value)), "explicit-body"
    if mode in {"off", "0", "false", "baseline"}:
        return 0, "explicit-off"
    if mode in {"on", "4", "true", "dflash"}:
        return 4, "explicit-on"
    if mode != "auto":
        raise ValueError("dflash_mode must be auto, on, or off")

    if body.get("tools") or body.get("tool_choice"):
        return 0, "tools"
    response_format = body.get("response_format") or {}
    if isinstance(response_format, dict) and response_format.get("type") in {"json_object", "json_schema"}:
        return 1, "structured-output"

    text = prompt_text(body)
    family = ""
    if STRUCTURED_RE.search(text):
        family = "structured"
    elif CODE_RE.search(text):
        family = "code"
    elif REASONING_RE.search(text):
        family = "reasoning"
    elif PROSE_RE.search(text):
        family = "prose"
    if family:
        try:
            return POLICIES[policy][family], family
        except KeyError:
            raise ValueError("unknown routing policy")
    return 0, "conservative-default"


class RouterState:
    def __init__(self):
        self.lock = threading.Lock()
        self.started = time.time()
        self.counts = {}
        self.in_flight = 0

    def enter(self, nmax, reason, adaptive=True):
        with self.lock:
            self.in_flight += 1
            # Explicit caller modes are authoritative. Automatic drafting is
            # capped under actual simultaneous load to protect tail latency.
            effective = min(nmax, 2) if adaptive and self.in_flight > 1 else nmax
            self._record_locked(effective, reason if effective == nmax else f"{reason}:contention-cap2")
            return effective

    def _record_locked(self, nmax, reason):
        key = f"nmax_{nmax}:{reason}"
        self.counts[key] = self.counts.get(key, 0) + 1

    def exit(self):
        with self.lock:
            self.in_flight = max(0, self.in_flight - 1)

    def snapshot(self):
        with self.lock:
            return {"status": "ok", "uptime_seconds": time.time() - self.started,
                    "in_flight": self.in_flight, "routes": dict(self.counts)}


class RouterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qdl-dflash-router/1"

    def log_message(self, fmt, *args):
        print(f"{self.log_date_time_string()} {self.client_address[0]} {fmt % args}", flush=True)

    def _json(self, status, body):
        encoded = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _proxy(self):
        if self.path == "/router/health" or self.path == "/router/stats":
            self._json(200, self.server.state.snapshot())
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        route_reason = None
        entered = False
        if raw and self.path.split("?", 1)[0] in {"/v1/chat/completions", "/v1/completions", "/completion"}:
            try:
                body = json.loads(raw)
                header_mode = self.headers.get("X-DFlash-Mode", "auto")
                nmax, route_reason = classify(body, header_mode, self.server.policy)
                explicit = route_reason.startswith("explicit-")
                nmax = self.server.state.enter(nmax, route_reason, adaptive=not explicit)
                entered = True
                body["speculative.n_max"] = nmax
                raw = json.dumps(body, separators=(",", ":")).encode()
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                if entered:
                    self.server.state.exit()
                self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                return

        headers = {}
        for key, value in self.headers.items():
            if key.lower() not in HOP_HEADERS | {"host", "content-length", "x-dflash-mode"}:
                headers[key] = value
        if raw:
            headers["Content-Length"] = str(len(raw))
        headers["Host"] = f"{self.server.backend_host}:{self.server.backend_port}"

        conn = http.client.HTTPConnection(self.server.backend_host, self.server.backend_port, timeout=600)
        try:
            conn.request(self.command, self.path, body=raw or None, headers=headers)
            response = conn.getresponse()
            self.send_response(response.status, response.reason)
            content_length = response.getheader("Content-Length")
            for key, value in response.getheaders():
                if key.lower() not in HOP_HEADERS | {"content-length"}:
                    self.send_header(key, value)
            if content_length is not None:
                self.send_header("Content-Length", content_length)
                self.end_headers()
                while chunk := response.read(65536):
                    self.wfile.write(chunk)
            else:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while chunk := response.read(4096):
                    self.wfile.write(f"{len(chunk):X}\r\n".encode())
                    self.wfile.write(chunk)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        except (OSError, http.client.HTTPException) as exc:
            if not self.wfile.closed:
                try:
                    self._json(502, {"error": {"message": f"DFlash backend unavailable: {exc}", "type": "upstream_error"}})
                except OSError:
                    pass
        finally:
            if entered:
                self.server.state.exit()
            conn.close()

    do_GET = _proxy
    do_POST = _proxy
    do_PUT = _proxy
    do_DELETE = _proxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument("--backend-host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=8090)
    parser.add_argument("--policy", choices=sorted(POLICIES), default=os.environ.get("DFLASH_ROUTING_POLICY", "acceptance"))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), RouterHandler)
    server.backend_host = args.backend_host
    server.backend_port = args.backend_port
    server.policy = args.policy
    server.state = RouterState()
    print(f"routing {args.host}:{args.port} to {args.backend_host}:{args.backend_port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
