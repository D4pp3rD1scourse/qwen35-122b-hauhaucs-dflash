#!/usr/bin/env python3
"""Run a complete DFlash experiment against one already-loaded backend.

The controller never restarts or mutates the backend. Baseline requests use the
literal flat JSON key ``speculative.n_max`` with value zero, so target, drafter,
KV allocation, and server process remain identical across the comparison.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import platform
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_PROMPTS = [
    {"family": "reasoning", "route_key": "reasoning", "prompt": "Calculate 73 * 48 - 219 step by step."},
    {"family": "reasoning", "route_key": "reasoning", "prompt": "A tank is 3/5 full. Adding 84 liters makes it 4/5 full. Find its capacity."},
    {"family": "structured", "route_key": "structured_status", "prompt": "Return only compact JSON with keys host, port, healthy for spark-3737 port 8089."},
    {"family": "structured", "route_key": "structured_summary", "prompt": "Return only JSON containing the first five primes and their sum."},
    {"family": "code", "prompt": "Return only Python implementing stable_unique(items) while preserving order."},
    {"family": "code", "prompt": "Return only Python for binary_search(sorted_items, target)."},
    {"family": "prose", "prompt": "Write a concise scene about a lighthouse whose beam reveals yesterday."},
    {"family": "prose", "prompt": "Describe rain beginning over a silent desert in four sentences."},
]


def load_prompts(path, split=None, limit_per_family=None):
    if path is None:
        return DEFAULT_PROMPTS
    selected, counts = [], {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if split is not None and row.get("split") != split:
            continue
        family = row.get("family") or row.get("category")
        prompt = row.get("prompt")
        if prompt is None and row.get("messages"):
            prompt = row["messages"][-1]["content"]
        if not family or not isinstance(prompt, str):
            raise ValueError("prompt rows require family/category and prompt/messages")
        if limit_per_family is not None and counts.get(family, 0) >= limit_per_family:
            continue
        counts[family] = counts.get(family, 0) + 1
        selected.append({"family": family, "route_key": row.get("route_key") or family, "prompt": prompt})
    if not selected:
        raise ValueError("no prompts selected")
    return selected


def percentile(values, q):
    values = sorted(values)
    if not values:
        return None
    point = (len(values) - 1) * q
    low = int(point)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (point - low)


def summarize(rows):
    good = [row for row in rows if row["ok"]]
    drafted = sum(row.get("draft_n", 0) for row in good)
    accepted = sum(row.get("accepted", 0) for row in good)
    latencies = [row["elapsed_ms"] for row in good]
    rates = [row["tps"] for row in good if row.get("tps")]
    return {
        "requests": len(rows), "successes": len(good), "errors": len(rows) - len(good),
        "median_tps": statistics.median(rates) if rates else None,
        "p50_latency_ms": percentile(latencies, .50), "p95_latency_ms": percentile(latencies, .95),
        "draft_tokens": drafted, "accepted_tokens": accepted,
        "acceptance_rate": accepted / drafted if drafted else None,
    }


def request(url, model, item, depth, max_tokens, timeout, slot=None):
    body = {"model": model, "messages": [{"role": "user", "content": item["prompt"]}],
            "temperature": 0, "seed": 20260712, "max_tokens": max_tokens,
            "stream": False, "cache_prompt": False, "speculative.n_max": depth}
    if slot is not None:
        body["id_slot"] = slot
    started = time.perf_counter()
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = json.load(response)
        timings = result.get("timings", {})
        return {"ok": True, "family": item["family"], "route_key": item.get("route_key", item["family"]), "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                "depth": depth, "elapsed_ms": (time.perf_counter() - started) * 1000,
                "tps": timings.get("predicted_per_second"), "draft_n": timings.get("draft_n", 0),
                "accepted": timings.get("draft_n_accepted", 0), "content": result["choices"][0]["message"]["content"]}
    except Exception as exc:
        return {"ok": False, "family": item["family"], "route_key": item.get("route_key", item["family"]), "depth": depth, "error": repr(exc)}


def route_depth(policy, item):
    key = item.get("route_key", item["family"])
    if key in policy:
        return key, policy[key]
    return item["family"], policy[item["family"]]


def parse_route_policy(value):
    """Parse family=depth pairs, including explicit depth-zero bypass."""
    if not value:
        return None
    policy = {}
    for item in value.split(","):
        family, separator, raw_depth = item.partition("=")
        if not separator or not family.strip():
            raise argparse.ArgumentTypeError("route policy must use family=depth pairs")
        try:
            depth = int(raw_depth)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid route depth: {raw_depth}") from exc
        if not 0 <= depth <= 8:
            raise argparse.ArgumentTypeError("route depths must be 0..8")
        policy[family.strip()] = depth
    return policy


def run(args):
    started_wall = datetime.now(timezone.utc)
    started = time.perf_counter()
    prompts = load_prompts(args.prompts, args.prompt_split, args.limit_per_family)
    phases = {}
    baseline = [request(args.url, args.model, item, 0, args.max_tokens, args.timeout, 0) for item in prompts]
    phases["baseline_nmax_0"] = {"summary": summarize(baseline), "rows": baseline}
    for depth in range(1, args.max_depth + 1):
        rows = [request(args.url, args.model, item, depth, args.max_tokens, args.timeout, 0) for item in prompts]
        phases[f"depth_{depth}"] = {"summary": summarize(rows), "rows": rows}
    correctness = []
    for index, item in enumerate(prompts):
        base = baseline[index]
        for depth in range(1, args.max_depth + 1):
            row = phases[f"depth_{depth}"]["rows"][index]
            correctness.append({"family": item["family"], "depth": depth, "both_ok": base["ok"] and row["ok"],
                                "exact": base.get("content") == row.get("content")})
    phases["correctness"] = {"comparisons": len(correctness), "all_exact": all(x["exact"] for x in correctness),
                              "exact": sum(x["exact"] for x in correctness), "rows": correctness}
    route_policy = parse_route_policy(args.route_policy)
    if route_policy is not None:
        missing = sorted({item.get("route_key", item["family"]) for item in prompts
                          if item.get("route_key", item["family"]) not in route_policy
                          and item["family"] not in route_policy})
        if missing:
            raise ValueError(f"route policy missing families: {missing}")
        routed = [request(args.url, args.model, item, route_depth(route_policy, item)[1],
                          args.max_tokens, args.timeout, 0) for item in prompts]
        phases["routed"] = {"summary": summarize(routed), "rows": routed,
                            "policy": route_policy}
        routed_correctness = [{"family": item["family"], "route_key": route_depth(route_policy, item)[0], "depth": route_depth(route_policy, item)[1],
                               "both_ok": baseline[index]["ok"] and routed[index]["ok"],
                               "exact": baseline[index].get("content") == routed[index].get("content")}
                              for index, item in enumerate(prompts)]
        phases["routed_correctness"] = {
            "comparisons": len(routed_correctness),
            "all_exact": all(row["exact"] for row in routed_correctness),
            "exact": sum(row["exact"] for row in routed_correctness),
            "rows": routed_correctness,
        }
    concurrent_rows = []
    for index in range(0, len(prompts), 2):
        pair = prompts[index:index + 2]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            concurrent_rows.extend(f.result() for f in [pool.submit(request, args.url, args.model, item, args.concurrency_depth,
                                                                     args.max_tokens, args.timeout, slot)
                                                        for slot, item in enumerate(pair)])
    phases["concurrency_two"] = {"summary": summarize(concurrent_rows), "rows": concurrent_rows}
    if route_policy is not None:
        routed_concurrent = []
        for index in range(0, len(prompts), 2):
            pair = prompts[index:index + 2]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                routed_concurrent.extend(f.result() for f in [
                    pool.submit(request, args.url, args.model, item, route_depth(route_policy, item)[1],
                                args.max_tokens, args.timeout, slot)
                    for slot, item in enumerate(pair)])
        phases["routed_concurrency_two"] = {"summary": summarize(routed_concurrent),
                                             "rows": routed_concurrent, "policy": route_policy}
    soak = []
    for index in range(0, args.soak_requests, 2):
        pair = [prompts[index % len(prompts)], prompts[(index + 1) % len(prompts)]]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            soak.extend(f.result() for f in [pool.submit(request, args.url, args.model, item, args.concurrency_depth,
                                                         args.max_tokens, args.timeout, slot) for slot, item in enumerate(pair)])
    phases["soak"] = {"summary": summarize(soak), "rows": soak}
    if route_policy is not None:
        routed_soak = []
        for index in range(0, args.soak_requests, 2):
            pair = [prompts[index % len(prompts)], prompts[(index + 1) % len(prompts)]]
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                routed_soak.extend(f.result() for f in [
                    pool.submit(request, args.url, args.model, item, route_depth(route_policy, item)[1],
                                args.max_tokens, args.timeout, slot)
                    for slot, item in enumerate(pair)])
        phases["routed_soak"] = {"summary": summarize(routed_soak), "rows": routed_soak,
                                  "policy": route_policy}
    elapsed = time.perf_counter() - started
    return {"schema_version": 1, "experiment_id": started_wall.strftime("%Y%m%dT%H%M%SZ") + "-" + args.checkpoint,
            "checkpoint": args.checkpoint, "backend": {"url": args.url, "model": args.model, "single_load": True,
            "baseline_contract": {"field": "speculative.n_max", "value": 0}},
            "host": platform.node(), "started_at": started_wall.isoformat(), "wall_time_seconds": elapsed,
            "under_45_minutes": elapsed < 2700, "config": vars(args) | {
                "prompts": str(args.prompts) if args.prompts else None,
                "output_root": str(args.output_root),
            },
            "phases": phases}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8090/v1/chat/completions")
    parser.add_argument("--model", default="qwen35-122b-uncensored")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompts", type=Path)
    parser.add_argument("--prompt-split", choices=("train", "validation", "held_out"))
    parser.add_argument("--limit-per-family", type=int)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--timeout", type=int, default=240)
    parser.add_argument("--soak-requests", type=int, default=40)
    parser.add_argument("--max-depth", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--concurrency-depth", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--route-policy",
                        help="Comma-separated family=depth pairs; depth 0 is an explicit bypass")
    args = parser.parse_args()
    if args.concurrency_depth > args.max_depth:
        parser.error("--concurrency-depth cannot exceed --max-depth")
    result = run(args)
    bundle = args.output_root / result["experiment_id"]
    bundle.mkdir(parents=True, exist_ok=False)
    (bundle / "experiment.json").write_text(json.dumps(result, indent=2) + "\n")
    (bundle / "SUMMARY.json").write_text(json.dumps({"experiment_id": result["experiment_id"],
        "checkpoint": result["checkpoint"], "wall_time_seconds": result["wall_time_seconds"],
        "under_45_minutes": result["under_45_minutes"],
        "phases": {key: value.get("summary", {k: v for k, v in value.items() if k != "rows"}) for key, value in result["phases"].items()}}, indent=2) + "\n")
    print(bundle)
    if not result["under_45_minutes"] or result["phases"]["soak"]["summary"]["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
