#!/bin/bash
set -euo pipefail
ROOT=/home/admin/benchmarks/qwen-dflash
V2=/home/admin/benchmarks/qwen-dflash-v2
SERIES=${SERIES:-$V2/checkpoints/runtime-v6-depth6-gamma060-001}
BASELINE=${BASELINE:-$V2/checkpoints/stretch-v5-targeted-depth6-gamma060-001/epoch-002}
VALIDATION=${VALIDATION:-$V2/corpus-runtime-v6-single-load/validation}
HELDOUT=${HELDOUT:-$V2/corpus-runtime-v6-single-load/held_out}
OUTPUT=${OUTPUT:-$V2/evaluations/runtime-v6-depth6-gamma060-001}
BASELINE_EPOCH=${BASELINE_EPOCH:-2}
EPOCHS=${EPOCHS:-4}
EPOCH_START=${EPOCH_START:-0}
RUNTIME_DEPTH=${RUNTIME_DEPTH:-6}
EARLY_POSITION_GAMMA=${EARLY_POSITION_GAMMA:-0.6}
FROZEN=$ROOT/frozen-verifier
exec 9>/tmp/qdl-stretch-v3-maintenance.lock
flock -n 9 || { echo "another DFlash unit owns the maintenance lock" >&2; exit 2; }
restore() {
  status=$?; trap - EXIT INT TERM
  systemctl --user start qwen35-dflash-backend.service qwen35-dflash-router.service >/dev/null 2>&1 || true
  for _ in $(seq 1 450); do
    if curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && curl -fsS --max-time 2 http://127.0.0.1:8089/health >/dev/null 2>&1; then
      printf '{"production_restored":true,"prior_status":%s}\n' "$status"; return "$status"
    fi; sleep 2
  done
  echo '{"production_restored":false}' >&2; return 1
}
trap restore EXIT INT TERM
test ! -e "$OUTPUT"; test -f "$BASELINE/model.safetensors"; test -d "$VALIDATION"; test -d "$HELDOUT"
models=("$BASELINE/model.safetensors"); candidates=()
for ((epoch=EPOCH_START; epoch<EPOCH_START+EPOCHS; epoch++)); do
  path=$(printf '%s/epoch-%03d/model.safetensors' "$SERIES" "$epoch")
  eval_json=$(printf '%s/epoch-%03d.json' "$OUTPUT" "$epoch")
  test -f "$path"; models+=("$path"); candidates+=("$eval_json")
done
systemctl --user stop qwen35-dflash-router.service qwen35-dflash-backend.service
mkdir -p "$OUTPUT"
export PYTHONPATH=$ROOT/speculators/src:$ROOT/train-venv/lib/python3.12/site-packages
export TORCHDYNAMO_DISABLE=1 PYTHONUNBUFFERED=1
PY=/home/admin/.venv-comfy/bin/python
sha256sum "${models[@]}" > "$OUTPUT/input-model-hashes.before.sha256"
"$PY" "$ROOT/evaluate_training_candidate.py" --checkpoint "$BASELINE" --epoch "$BASELINE_EPOCH" --frozen "$FROZEN" \
  --validation "$VALIDATION" --heldout "$HELDOUT" --runtime-depth "$RUNTIME_DEPTH" --early-position-gamma "$EARLY_POSITION_GAMMA" --output "$OUTPUT/v5-baseline.json"
for ((epoch=EPOCH_START; epoch<EPOCH_START+EPOCHS; epoch++)); do
  checkpoint=$(printf '%s/epoch-%03d' "$SERIES" "$epoch")
  eval_json=$(printf '%s/epoch-%03d.json' "$OUTPUT" "$epoch")
  "$PY" "$ROOT/evaluate_training_candidate.py" --checkpoint "$checkpoint" --epoch "$epoch" --frozen "$FROZEN" \
    --validation "$VALIDATION" --heldout "$HELDOUT" --runtime-depth "$RUNTIME_DEPTH" --early-position-gamma "$EARLY_POSITION_GAMMA" --output "$eval_json"
done
read -r base_val base_held < <("$PY" - "$OUTPUT/v5-baseline.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); print(d["validation"]["eal"],d["heldout"]["eal"])
PY
)
"$PY" "$ROOT/select_training_candidates.py" "${candidates[@]}" \
  --baseline-validation-eal "$base_val" --baseline-heldout-eal "$base_held" --output "$OUTPUT/selection.json"
sha256sum -c "$OUTPUT/input-model-hashes.before.sha256"
sha256sum "$OUTPUT"/*.json > "$OUTPUT/output-json-hashes.sha256"
find "$OUTPUT" -name '*.partial' -print -quit | grep -q . && { echo "partial evaluation artifact remains" >&2; exit 1; }
echo '{"runtime_v6_evaluation_complete":true}'
