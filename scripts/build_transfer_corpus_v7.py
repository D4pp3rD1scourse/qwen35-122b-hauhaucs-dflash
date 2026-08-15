#!/usr/bin/env python3
"""Build template-diverse, benchmark-shaped reasoning/JSON transfer data."""
import argparse, hashlib, json
from collections import Counter
from pathlib import Path

COUNTS={"train":{"reasoning":240,"structured":160},
        "validation":{"reasoning":48,"structured":32},
        "held_out":{"reasoning":48,"structured":32}}
FAMILIES={
 "train":{"reasoning":("inventory_arithmetic","cistern_fraction","shipment_arithmetic","bin_fraction"),
          "structured":("service_status","number_summary","worker_status","value_summary")},
 "validation":{"reasoning":("orchard_arithmetic","vat_fraction"),
               "structured":("relay_status","integer_summary")},
 "held_out":{"reasoning":("ticket_arithmetic","hopper_fraction"),
             "structured":("node_status","digit_summary")}}
ARITH=[
 "Calculate {a} * {b} - {c} step by step for case {tag}.",
 "For case {tag}, work out {a} times {b}, then subtract {c}. Show each step.",
 "Solve ({a} × {b}) - {c}. Give concise intermediate arithmetic. Label: {tag}.",
 "Case {tag}: multiply {a} by {b} and reduce the product by {c}; explain briefly.",
 "Find the value of {a}*{b}-{c}, showing the multiplication and subtraction. {tag}",
 "Step through this fictional ledger calculation: {a} multiplied by {b}, less {c}. Ref {tag}.",
]
FRACTION=[
 "A {thing} is {lo}/{den} full. Adding {delta} units makes it {hi}/{den} full. Find its capacity. {tag}",
 "For {tag}, adding {delta} items raises a {thing} from {lo}/{den} to {hi}/{den} full. What is total capacity? Show steps.",
 "A fictional {thing} moves from {lo}/{den} filled to {hi}/{den} after {delta} units are added. Determine full size. Ref {tag}.",
 "Case {tag}: {delta} units represent the increase between {lo}/{den} and {hi}/{den} of a {thing}. Compute capacity.",
 "Find the capacity of a {thing}: it is initially {lo}/{den} full and becomes {hi}/{den} full after adding {delta}. {tag}",
 "Show concise fraction reasoning for {tag}. A {thing} gains {delta} units, changing fullness from {lo}/{den} to {hi}/{den}.",
]
STATUS=[
 "Return compact JSON only with host {host}, port {port}, and healthy true. Tag {tag}.",
 "Output one JSON object only: host={host}, port={port}, healthy=true, tag={tag}.",
 "Encode this fictional service as compact JSON with keys host, port, healthy, tag: {host}, {port}, true, {tag}.",
 "JSON only, no prose. Use host {host}, port {port}, healthy true, and tag {tag}.",
 "Create valid compact JSON for {tag}; fields are host ({host}), port ({port}), healthy (true).",
 "Respond with only a compact object containing host, port, healthy, tag for {host}, {port}, true, {tag}.",
]
SUMMARY=[
 "Return compact JSON only with numbers {nums} and their sum {total}. Tag {tag}.",
 "JSON only: include values {nums}, count {count}, sum {total}, and tag {tag}.",
 "Encode {nums} as compact JSON using keys numbers, count, sum, tag. The sum is {total}; tag {tag}.",
 "Produce one compact JSON object for {tag}: numbers={nums}, count={count}, sum={total}.",
 "No prose. Return valid compact JSON containing sequence {nums}, its count, its sum {total}, and {tag}.",
 "For tag {tag}, output only JSON with values {nums}, total {total}, and count {count}.",
]
FORBIDDEN=("73 * 48 - 219","73*48-219","84 liters","3/5 full","4/5 full","first five primes","spark-3737","8089")

def make(category,family,i):
 tag=f"sample_{i:04d}"
 if category=="reasoning" and "arithmetic" in family:
  a=21+(i*7)%61; b=11+(i*5)%37; c=43+(i*13)%211
  return ARITH[i%len(ARITH)].format(a=a,b=b,c=c,tag=tag)
 if category=="reasoning":
  den=6+i%7; lo=1+i%(den-2); hi=lo+1; unit=7+(i*3)%19; delta=unit*(hi-lo)
  return FRACTION[i%len(FRACTION)].format(thing=("reservoir","container","silo","bay")[i%4],lo=lo,den=den,hi=hi,delta=delta,tag=tag)
 if "status" in family:
  return STATUS[i%len(STATUS)].format(host=f"test-node-{i%97}",port=6100+i%700,tag=tag)
 nums=[3+i%7,8+(i*3)%11,15+(i*5)%17,27+(i*7)%19,41+(i*11)%23]
 return SUMMARY[i%len(SUMMARY)].format(nums=nums,total=sum(nums),count=len(nums),tag=tag)

def build():
 rows=[]; g=0
 for split,cats in COUNTS.items():
  for category,count in cats.items():
   fs=FAMILIES[split][category]
   for n in range(count):
    family=fs[n%len(fs)]; text=make(category,family,g)
    rows.append({"id":f"transfer-v7-{split}-{category}-{family}-{n:04d}","split":split,"category":category,
      "prompt_family":family,"provenance":"deterministic_synthetic_transfer_v7",
      "messages":[{"role":"user","content":text}],"temperature":0.0,"seed":20260716+g,
      "max_tokens":96,"target_speculative_depth":1+g%6,"prompt_length_band":"runtime_short",
      "capture_shape":{"name":"paired_runtime","workers":2,"homogeneous_capture_unit":True}}); g+=1
 return rows

def audit(rows):
 errors=[]; ids=set(); prompts=set(); family_splits={}
 for row in rows:
  text=row["messages"][0]["content"]; norm=" ".join(text.casefold().split())
  if row["id"] in ids or norm in prompts: errors.append(f"duplicate {row['id']}")
  if any(x.casefold() in norm for x in FORBIDDEN): errors.append(f"benchmark leakage {row['id']}")
  ids.add(row["id"]); prompts.add(norm); family_splits.setdefault(row["prompt_family"],set()).add(row["split"])
 for family,splits in family_splits.items():
  if len(splits)!=1: errors.append(f"family crosses splits {family}: {sorted(splits)}")
 return errors

def main():
 p=argparse.ArgumentParser(); p.add_argument("output",type=Path); a=p.parse_args(); rows=build(); errors=audit(rows)
 if errors: raise SystemExit("\n".join(errors))
 if a.output.exists(): raise SystemExit(f"refusing to overwrite {a.output}")
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text("".join(json.dumps(x,sort_keys=True)+"\n" for x in rows))
 print(json.dumps({"rows":len(rows),"splits":dict(Counter(x['split'] for x in rows)),
  "categories":dict(Counter(x['category'] for x in rows)),"families":len({x['prompt_family'] for x in rows}),
  "sha256":hashlib.sha256(a.output.read_bytes()).hexdigest()},indent=2,sort_keys=True))
if __name__=="__main__": main()
