#!/bin/bash
set -euo pipefail

ROOT=/home/admin/benchmarks/qwen-dflash
V2_ROOT=/home/admin/benchmarks/qwen-dflash-v2
TRAINER=$ROOT/train_hard_targets.py
SERIES=${SERIES:?set SERIES to an absolute checkpoint-series path}
RESUME=${RESUME:-$ROOT/checkpoints/fullbody-mp-001}
EPOCHS=${EPOCHS:-4}
TRAIN=${TRAIN:-$V2_ROOT/corpus-v2/train/reconciled}
VALIDATION=${VALIDATION:-$V2_ROOT/corpus-v2/validation/reconciled}
RUNTIME_DEPTH=${RUNTIME_DEPTH:-4}
EARLY_POSITION_GAMMA=${EARLY_POSITION_GAMMA:-0.75}
LR=${LR:-1e-5}
EXTRA_ARGS=${EXTRA_ARGS:-}
WARM_START=${WARM_START:-0}
LOCKFILE=/tmp/qdl-stretch-v3-maintenance.lock
exec 9>"$LOCKFILE"
flock -n 9 || { echo "another stretch-v3 unit owns $LOCKFILE" >&2; exit 2; }

restore() {
  systemctl --user start qwen35-dflash-backend.service qwen35-dflash-router.service >/dev/null 2>&1 || true
  for _ in $(seq 1 240); do
    backend=$(curl -fsS --max-time 2 http://127.0.0.1:8090/health 2>/dev/null || true)
    public=$(curl -fsS --max-time 2 http://127.0.0.1:8089/health 2>/dev/null || true)
    if [[ $backend == *'"status":"ok"'* && $public == *'"status":"ok"'* ]]; then
      printf '{"production_restored":true,"backend":%q,"public":%q}\n' "$backend" "$public"
      return 0
    fi
    sleep 2
  done
  echo '{"production_restored":false}' >&2
  return 1
}
trap restore EXIT

if systemctl --user is-active --quiet qdl-corpus-v2-held_out.service \
  || systemctl --user is-active --quiet qdl-corpus-v2-validation.service \
  || systemctl --user is-active --quiet qdl-corpus-v2-train.service; then
  echo "refusing to overlap a corpus-v2 capture" >&2
  exit 1
fi
if systemctl --user list-units --state=running --no-legend 'qdl-corpus-stretch-v3-*.service' | grep -q .; then
  echo "refusing to overlap a stretch-v3 capture" >&2
  exit 1
fi

test -f "$RESUME/model.safetensors"
test -f "$RESUME/training.json"
test -d "$TRAIN"
test -d "$VALIDATION"
if [[ -d "$SERIES" ]] && find "$SERIES" -name '*.partial' -print -quit | grep -q .; then
  echo "refusing to start with partial artifact(s) in $SERIES" >&2
  find "$SERIES" -name '*.partial' -print >&2
  exit 1
fi
systemctl --user stop qwen35-dflash-router.service qwen35-dflash-backend.service

export PYTHONPATH=$ROOT/speculators/src:$ROOT/train-venv/lib/python3.12/site-packages
export TORCHDYNAMO_DISABLE=1
export PYTHONUNBUFFERED=1

# EXTRA_ARGS is set only by the internal operator wrapper and intentionally
# undergoes word splitting for bounded smoke/resume-test flags.
# shellcheck disable=SC2086
WARM_ARGS=()
if [[ "$WARM_START" == 1 ]]; then WARM_ARGS+=(--warm-start); fi
/home/admin/.venv-comfy/bin/python "$TRAINER" \
  --stock "$ROOT/hf/draft" --frozen "$ROOT/frozen-verifier" \
  --train "$TRAIN" --validation "$VALIDATION" \
  --output "$SERIES" --resume "$RESUME" "${WARM_ARGS[@]}" --mode full-body \
  --epochs "$EPOCHS" --lr "$LR" --grad-accum 8 --max-anchors 8 \
  --optimizer adafactor --bucket-width 16 --prefetch 2 --pin-memory \
  --validate-every 1 --early-stop-patience 2 \
  --runtime-depth "$RUNTIME_DEPTH" --early-position-gamma "$EARLY_POSITION_GAMMA" $EXTRA_ARGS

find "$SERIES" -maxdepth 2 -name '*.partial' -print -quit | grep -q . && {
  echo "partial checkpoint remains" >&2
  exit 1
}
echo '{"training_acceleration_complete":true}'
