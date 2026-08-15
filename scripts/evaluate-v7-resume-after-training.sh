#!/bin/bash
set -euo pipefail
V2=/home/admin/benchmarks/qwen-dflash-v2
TRAIN_UNIT=${TRAIN_UNIT:-qdl-transfer-v7-resume-epochs4-7.service}
SERIES=$V2/checkpoints/transfer-v7-depth6-gamma060-001
EPOCH_START=${EPOCH_START:-4}
EPOCHS=${EPOCHS:-4}
BASELINE_EPOCH=${BASELINE_EPOCH:-3}
OUTPUT=${OUTPUT:-$V2/evaluations/transfer-v7-depth6-gamma060-resume-epochs4-7}
while systemctl --user is-active --quiet "$TRAIN_UNIT"; do sleep 15; done
for ((epoch=EPOCH_START; epoch<EPOCH_START+EPOCHS; epoch++)); do
  test -f "$(printf '%s/epoch-%03d/model.safetensors' "$SERIES" "$epoch")"
done
curl -fsS --max-time 3 http://127.0.0.1:8090/health >/dev/null
curl -fsS --max-time 3 http://127.0.0.1:8089/health >/dev/null
SERIES="$SERIES" BASELINE="$(printf '%s/epoch-%03d' "$SERIES" "$BASELINE_EPOCH")" \
BASELINE_EPOCH="$BASELINE_EPOCH" EPOCH_START="$EPOCH_START" EPOCHS="$EPOCHS" \
VALIDATION="$V2/corpus-transfer-v7-single-load/validation" \
HELDOUT="$V2/corpus-transfer-v7-single-load/held_out" \
OUTPUT="$OUTPUT" \
  /bin/bash "$V2/evaluate-runtime-v6.sh"
echo V7_RESUME_EVALUATION_OK
