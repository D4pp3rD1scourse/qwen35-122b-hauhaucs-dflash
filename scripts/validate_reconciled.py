#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate(root):
    manifest_path = root / "reconciliation.json"
    data = json.loads(manifest_path.read_text())
    errors, ids, rows = [], set(), 0
    if data.get("schema_version") != 2 or data.get("state") != "complete":
        errors.append("manifest is not complete schema 2")
    samples = data.get("samples", [])
    for index, item in enumerate(samples):
        if item.get("index") != index:
            errors.append(f"non-contiguous sample index at {index}")
        if item["id"] in ids:
            errors.append(f"duplicate sample id: {item['id']}")
        ids.add(item["id"])
        path = root / item["path"]
        if not path.is_file():
            errors.append(f"missing sample: {item['path']}")
            continue
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            errors.append(f"file fingerprint mismatch: {item['path']}")
            continue
        with safe_open(path, framework="pt", device="cpu") as tensors:
            hs_shape = tensors.get_slice("hidden_states").get_shape()
            token_shape = tensors.get_slice("token_ids").get_shape()
            if hs_shape != [item["rows"], 8, 3072] or token_shape != [item["rows"]]:
                errors.append(f"tensor contract mismatch: {item['path']}")
        rows += item["rows"]
    if len(samples) != data.get("sample_count") or rows != data.get("committed_rows"):
        errors.append("aggregate sample or row count mismatch")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    errors = validate(args.root)
    if errors:
        print("\n".join(errors))
        raise SystemExit(1)
    print("reconciled dataset valid")


if __name__ == "__main__":
    main()
