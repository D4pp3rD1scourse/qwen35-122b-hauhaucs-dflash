#!/usr/bin/env python3
"""Discard rejected speculative rows by matching exact committed request tokens."""
import argparse
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path

import torch
from safetensors.torch import save_file

from inspect_capture import iter_full_records

LAYERS = 8
WIDTH = 3072


def load_expected(results):
    queues = defaultdict(deque)
    for line in results.read_text().splitlines():
        item = json.loads(line)
        item["expected_tokens"] = item["prompt_tokens"] + item["completion_tokens"]
        queues[item["slot"]].append(item)
    return queues


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reconcile(shards, results, output, raw_manifest=None):
    queues = load_expected(results)
    current = {}
    written = []
    discarded = 0

    def start(sequence):
        if not queues[sequence]:
            raise ValueError(f"capture has more requests than results for slot {sequence}")
        return {"request": queues[sequence].popleft(), "selected": {}, "raw_rows": 0}

    def finish(sequence):
        nonlocal discarded
        state = current.get(sequence)
        if not state:
            return
        request = state["request"]
        expected = request["expected_tokens"]
        required = len(expected) - 1  # final sampled token is not necessarily decoded
        missing = [position for position in range(required) if position not in state["selected"]]
        if missing:
            raise ValueError(f"{request['id']} missing committed positions: {missing[:12]}")
        hidden = torch.empty((required, LAYERS, WIDTH), dtype=torch.bfloat16)
        for position in range(required):
            hidden[position].copy_(torch.frombuffer(bytearray(state["selected"][position]), dtype=torch.bfloat16).reshape(LAYERS, WIDTH))
        token_ids = torch.tensor(expected[:required], dtype=torch.int64)
        path = partial / f"hs_{len(written):06d}.safetensors"
        save_file({"hidden_states": hidden, "token_ids": token_ids}, path,
                  metadata={"sample_id": request["id"], "slot": str(sequence)})
        discarded += state["raw_rows"] - required
        written.append({"index": len(written), "id": request["id"], "slot": sequence,
                        "rows": required, "raw_rows": state["raw_rows"], "path": path.name,
                        "bytes": path.stat().st_size, "sha256": sha256(path)})

    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError(f"interrupted reconciliation exists: {partial}")
    partial.mkdir(parents=True)
    for shard in shards:
        for sequence, position, token, features in iter_full_records(shard):
            if position == 0:
                finish(sequence)
                current[sequence] = start(sequence)
            if sequence not in current:
                raise ValueError(f"slot {sequence} record appeared before position zero")
            state = current[sequence]
            state["raw_rows"] += 1
            expected = state["request"]["expected_tokens"]
            if 0 <= position < len(expected) and token == expected[position]:
                state["selected"][position] = features  # retain latest verified match
    for sequence in sorted(current):
        finish(sequence)
    remaining = {slot: len(queue) for slot, queue in queues.items() if queue}
    if remaining:
        raise ValueError(f"results contain requests absent from capture: {remaining}")
    manifest = {"schema_version": 2, "state": "complete", "samples": written, "sample_count": len(written),
                "committed_rows": sum(item["rows"] for item in written),
                "discarded_speculative_rows": discarded,
                "requests_sha256": sha256(results)}
    if raw_manifest is not None:
        manifest["raw_manifest_sha256"] = sha256(raw_manifest)
    encoded = json.dumps(manifest, indent=2) + "\n"
    (partial / "reconciliation.json").write_text(encoded)
    manifest["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    partial.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=Path, nargs="+", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw-manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.shards, args.results, args.output, args.raw_manifest), indent=2))


if __name__ == "__main__":
    main()
