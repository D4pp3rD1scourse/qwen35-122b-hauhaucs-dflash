#!/usr/bin/env python3
"""Create an immutable symlink view for one split from a combined reconciliation."""
import argparse, hashlib, json, os
from pathlib import Path

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--id-prefix", default="runtime-v6")
    p.add_argument("--split",choices=("train","validation","held_out"),required=True); a=p.parse_args()
    if a.output.exists(): raise SystemExit(f"refusing existing output {a.output}")
    payload=json.loads((a.source/"reconciliation.json").read_text()); selected=[]; a.output.mkdir(parents=True)
    marker=f"{a.id_prefix}-{a.split}-"
    for sample in payload["samples"]:
        if not sample["id"].startswith(marker): continue
        src=a.source/sample["path"]
        if sha(src)!=sample["sha256"]: raise SystemExit(f"hash mismatch {src}")
        os.symlink(src,a.output/sample["path"]); selected.append(sample)
    if not selected: raise SystemExit(f"no samples selected for {a.split}")
    manifest={"schema_version":1,"split":a.split,"source":str(a.source),"sample_count":len(selected),"samples":selected}
    (a.output/"selection.json.partial").write_text(json.dumps(manifest,indent=2)+"\n")
    os.replace(a.output/"selection.json.partial",a.output/"selection.json")
    print(json.dumps({"split":a.split,"sample_count":len(selected)}))
if __name__ == "__main__": main()
