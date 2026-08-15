#!/usr/bin/env python3
"""Build a deterministic short-form corpus shaped like the live promotion workload."""
import argparse, hashlib, json, re
from collections import Counter
from pathlib import Path

COUNTS = {
    "train": {"reasoning": 144, "structured": 120, "code": 36},
    "validation": {"reasoning": 30, "structured": 26, "code": 8},
    "held_out": {"reasoning": 30, "structured": 26, "code": 8},
}
FAMILIES = {
    "train": {"reasoning": ("crate_adjustment", "reservoir_ratio", "ticket_balance"),
              "structured": ("relay_state", "batch_totals", "device_snapshot"),
              "code": ("first_distinct", "ordered_lookup")},
    "validation": {"reasoning": ("parcel_adjustment", "silo_ratio"),
                   "structured": ("queue_state", "series_totals"),
                   "code": ("unique_prefix",)},
    "held_out": {"reasoning": ("ledger_adjustment", "hopper_ratio"),
                 "structured": ("worker_state", "sequence_totals"),
                 "code": ("sorted_probe",)},
}
FORBIDDEN = ("73 * 48", "84 liters", "3/5 full", "4/5 full", "first five primes",
             '"host"', '"port"', "lighthouse", "desert rain")

def prompt(category, family, i):
    tag = f"unit_{i:04d}"
    a, b, c = 19 + i % 53, 7 + (i * 3) % 31, 4 + (i * 5) % 17
    if category == "reasoning":
        if "ratio" in family:
            den = 7 + i % 4; low = 2 + i % 2; delta = (3 + i % 5) * den
            return (f"A fictional {family.replace('_',' ')} is {low}/{den} filled. Adding {delta} crates makes it "
                    f"{low+1}/{den} filled. Find the full capacity and show concise steps. Reference {tag}.")
        return (f"Compute ({a} * {b}) + {c} - {b} for fictional record {tag}. Show concise steps and end with the result.")
    if category == "structured":
        if "totals" in family:
            vals = [2 + i % 5, 5 + i % 7, 9 + i % 11, 14 + i % 13]
            return f"Return compact JSON only with label {tag}, values {vals}, their count, and their sum."
        return (f"Return compact JSON only for fictional {tag} with node_name, channel {7000+i%400}, "
                f"ready true, retries {i%4}, and state \"{family}\".")
    if "lookup" in family or "probe" in family:
        return f"Write only Python: define {family}(items, target) returning the sorted-list index or -1. Add two asserts. {tag}"
    return f"Write only Python: define {family}(items) preserving order while removing repeats. Add two asserts. {tag}"

def build():
    rows=[]; g=0
    for split, cats in COUNTS.items():
        for category, count in cats.items():
            fs=FAMILIES[split][category]
            for n in range(count):
                family=fs[n % len(fs)]; text=prompt(category, family, g)
                rows.append({"id":f"runtime-v6-{split}-{category}-{family}-{n:04d}", "split":split,
                    "category":category, "prompt_family":family,
                    "provenance":"deterministic_synthetic_runtime_v6",
                    "messages":[{"role":"user","content":text}], "temperature":0.0,
                    "seed":20260715+g, "max_tokens":96, "target_speculative_depth":1+g%6,
                    "prompt_length_band":"runtime_short", "capture_shape":{"name":"paired_runtime","workers":2,
                    "homogeneous_capture_unit":True}})
                g += 1
    return rows

def audit(rows):
    errors=[]; ids=set(); prompts=set(); family_splits={}
    for row in rows:
        text=row["messages"][0]["content"]; norm=" ".join(text.casefold().split())
        if row["id"] in ids: errors.append(f"duplicate id {row['id']}")
        if norm in prompts: errors.append(f"duplicate prompt {row['id']}")
        if any(x.casefold() in norm for x in FORBIDDEN): errors.append(f"benchmark leakage {row['id']}")
        if not 12 <= len(text.split()) <= 42: errors.append(f"non-runtime length {row['id']}")
        ids.add(row["id"]); prompts.add(norm); family_splits.setdefault(row["prompt_family"],set()).add(row["split"])
    for f,s in family_splits.items():
        if len(s)!=1: errors.append(f"family crosses splits {f}: {sorted(s)}")
    return errors

def main():
    p=argparse.ArgumentParser(); p.add_argument("output",type=Path); a=p.parse_args(); rows=build(); errors=audit(rows)
    if errors: raise SystemExit("\n".join(errors))
    if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
    print(json.dumps({"rows":len(rows),"splits":dict(Counter(x['split'] for x in rows)),
      "categories":dict(Counter(x['category'] for x in rows)),"families":len({x['prompt_family'] for x in rows}),
      "sha256":hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2,sort_keys=True))
if __name__ == "__main__": main()
