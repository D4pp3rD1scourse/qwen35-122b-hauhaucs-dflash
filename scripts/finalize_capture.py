#!/usr/bin/env python3
"""Validate rotated QDL shards and atomically write a fingerprinted run manifest."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from inspect_capture import inspect


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(args):
    shards = sorted(args.directory.glob(f"{args.prefix}.*.qdlhs"))
    if not shards:
        raise ValueError("no completed rotated shards found")
    if list(args.directory.glob(f"{args.prefix}.*.qdlhs.partial")):
        raise ValueError("partial shards exist; preserve and audit them before resume")
    entries, total = [], 0
    for index, shard in enumerate(shards):
        expected = f"{args.prefix}.{index:06d}.qdlhs"
        if shard.name != expected:
            raise ValueError(f"non-contiguous shard sequence: expected {expected}, got {shard.name}")
        meta = inspect(shard)
        if meta["dtype"] != "bf16" or meta["hidden_width"] != 3072:
            raise ValueError(f"unexpected tensor contract in {shard.name}")
        entries.append({"index": index, "path": shard.name, "bytes": shard.stat().st_size,
                        "sha256": sha256(shard), "rows": meta["rows"]})
        total += meta["rows"]
    return {
        "schema_version": 2, "state": "complete", "split": args.split, "target": args.target,
        "target_sha256": args.target_sha256, "draft_sha256": args.draft_sha256,
        "tokenizer_sha256": args.tokenizer_sha256, "corpus_sha256": args.corpus_sha256,
        "llama_cpp_commit": args.llama_cpp_commit, "layers": [2, 8, 15, 21, 27, 33, 40, 46],
        "hidden_width": 3072, "dtype": "bf16", "token_count": total, "shards": entries,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--split", choices=["train", "validation", "held_out", "combined"], required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--draft-sha256", required=True)
    parser.add_argument("--tokenizer-sha256", required=True)
    parser.add_argument("--corpus-sha256", required=True)
    parser.add_argument("--llama-cpp-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".partial").exists():
        raise SystemExit("refusing to overwrite existing manifest or partial")
    data = build(args)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(json.dumps(data, indent=2) + "\n")
    with partial.open("rb") as stream:
        os.fsync(stream.fileno())
    partial.rename(args.output)
    print(f"finalized {len(data['shards'])} shards / {data['token_count']} rows -> {args.output}")


if __name__ == "__main__":
    main()
