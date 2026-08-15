#!/bin/bash
set -euo pipefail
ROOT=/home/admin/benchmarks/qwen-dflash-v2
TARGET=/home/admin/.lmstudio/models/HauhauCS/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive/Qwen3.5-122B-A10B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf
DRAFT=/home/admin/benchmarks/qwen-dflash/checkpoints/fullbody-mp-001-Q4_K_M.gguf
CORPUS=${CORPUS:-$ROOT/runtime-shaped-v6.jsonl}
SERVER=/home/admin/benchmarks/dflash-builds/llama.cpp/build/bin/llama-server
DIR=${DIR:-$ROOT/corpus-runtime-v6-single-load}
RAW=$DIR/raw
CAPTURE_PREFIX=${CAPTURE_PREFIX:-runtime-v6}
ID_PREFIX=${ID_PREFIX:-runtime-v6}
PREFIX=$RAW/$CAPTURE_PREFIX
UNIT=${CAPTURE_UNIT:-qdl-corpus-runtime-v6.service}
BACKEND=qwen35-dflash-backend.service
ROUTER=qwen35-dflash-router.service
STARTED=$(date +%s)
exec 9>/tmp/qdl-stretch-v3-maintenance.lock
flock -n 9 || { echo "another DFlash maintenance unit is active" >&2; exit 2; }

restore() {
  status=$?; trap - EXIT INT TERM
  systemctl --user stop "$UNIT" >/dev/null 2>&1 || true
  systemctl --user start "$BACKEND" >/dev/null 2>&1 || true
  for _ in $(seq 1 450); do curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && break; sleep 2; done
  systemctl --user start "$ROUTER" >/dev/null 2>&1 || true
  restored=false
  for _ in $(seq 1 120); do curl -fsS --max-time 2 http://127.0.0.1:8089/health >/dev/null 2>&1 && { restored=true; break; }; sleep 2; done
  printf '{"exit_status":%s,"wall_time_seconds":%s,"production_restored":%s}\n' "$status" "$(( $(date +%s)-STARTED ))" "$restored" | tee "$DIR/restoration.json"
  test "$restored" = true || return 1
  return "$status"
}
trap restore EXIT INT TERM

test -f "$CORPUS"; test -f "$TARGET"; test -f "$DRAFT"
mkdir "$DIR"; mkdir "$RAW"
systemctl --user stop "$ROUTER" "$BACKEND"
systemd-run --user --unit="$UNIT" --property=Restart=no \
  --setenv=QDL_CAPTURE_PATH="$PREFIX" --setenv=QDL_CAPTURE_MAX_TOKENS=1000000 \
  --setenv=QDL_CAPTURE_SHARD_TOKENS=2048 \
  --setenv=LD_LIBRARY_PATH=/home/admin/benchmarks/dflash-builds/llama.cpp/build/bin:/usr/local/cuda/targets/sbsa-linux/lib \
  "$SERVER" -m "$TARGET" --spec-draft-model "$DRAFT" --spec-type draft-dflash --spec-draft-n-max 6 \
  --host 127.0.0.1 --port 8090 -c 262144 --parallel 2 -ngl 999 --spec-draft-ngl 999 --jinja \
  --alias qwen35-122b-uncensored --reasoning off --reasoning-format none \
  --chat-template-kwargs '{"enable_thinking":false}' >/dev/null
for _ in $(seq 1 450); do
  curl -fsS --max-time 2 http://127.0.0.1:8090/health >/dev/null 2>&1 && break
  systemctl --user is-active --quiet "$UNIT" || { journalctl --user -u "$UNIT" -n 120 --no-pager; exit 1; }
  sleep 2
done
for split in train validation held_out; do
  python3 "$ROOT/run_corpus.py" "$CORPUS" --split "$split" --concurrency 2 --results "$DIR/$split-requests.jsonl"
done
systemctl --user stop "$UNIT"
test "$(find "$RAW" -name '*.partial' | wc -l)" = 0
cat "$DIR/train-requests.jsonl" "$DIR/validation-requests.jsonl" "$DIR/held_out-requests.jsonl" > "$DIR/all-requests.jsonl"
TARGET_SHA=$(awk -v p="$TARGET" '$2==p {print $1; exit}' "$ROOT"/runtime-evidence/*/input-hashes.sha256)
test -n "$TARGET_SHA"
DRAFT_SHA=$(sha256sum "$DRAFT" | awk '{print $1}')
CORPUS_SHA=$(sha256sum "$CORPUS" | awk '{print $1}')
TOKENIZER_SHA=$TARGET_SHA
COMMIT=$(git -C /home/admin/benchmarks/dflash-builds/llama.cpp rev-parse HEAD)
python3 "$ROOT/finalize_capture.py" "$RAW" --prefix "$CAPTURE_PREFIX" --split combined --target "$TARGET" \
  --target-sha256 "$TARGET_SHA" --draft-sha256 "$DRAFT_SHA" --tokenizer-sha256 "$TOKENIZER_SHA" \
  --corpus-sha256 "$CORPUS_SHA" --llama-cpp-commit "$COMMIT" --output "$DIR/raw-manifest.json"
python3 "$ROOT/reconcile_capture.py" --shards "$RAW"/*.qdlhs --results "$DIR/all-requests.jsonl" \
  --raw-manifest "$DIR/raw-manifest.json" --output "$DIR/reconciled"
python3 "$ROOT/validate_reconciled.py" "$DIR/reconciled"
for split in train validation held_out; do
  python3 "$ROOT/build_split_dataset_view.py" --source "$DIR/reconciled" --output "$DIR/$split" --split "$split" --id-prefix "$ID_PREFIX"
done
sha256sum "$CORPUS" "$DIR"/*-requests.jsonl "$DIR/raw-manifest.json" "$DIR/reconciled/reconciliation.json" > "$DIR/evidence.sha256"
echo SINGLE_LOAD_CAPTURE_OK
