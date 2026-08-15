#!/bin/bash
set -euo pipefail
LLAMA=/home/admin/benchmarks/dflash-builds/llama.cpp
SOURCE=${SOURCE:-/home/admin/benchmarks/qwen-dflash-v2/checkpoints/runtime-v6-depth6-gamma060-001/epoch-003}
OUT=${OUT:-/home/admin/benchmarks/qwen-dflash-v2/exports/runtime-v6-depth6-gamma060-001}
TARGET_METADATA=/home/admin/benchmarks/qwen-dflash/hf/target-metadata
EXPECTED_SOURCE=${EXPECTED_SOURCE:-df6138e9ac24a8f8ea80c4984c2fd34d1fd4c0c8b6fa1fa7ed23c1ca77405868}
EXPECTED_CONVERTER=c819f18fb22927b49fabc3b35d1c9e21ee638b3817eccd1bd4efbcc7116eeb4d
EXPECTED_QUANTIZER=2f29c0a446937ae3ad8ae4c4f677e18623e15ba9209b2b880183d61083668c05
EPOCH_LABEL=${EPOCH_LABEL:-epoch-003}
BF16=$OUT/$EPOCH_LABEL-BF16.gguf
Q4=$OUT/$EPOCH_LABEL-Q4_K_M.gguf
test "$(sha256sum "$SOURCE/model.safetensors" | cut -d' ' -f1)" = "$EXPECTED_SOURCE"
test "$(sha256sum "$LLAMA/convert_hf_to_gguf.py" | cut -d' ' -f1)" = "$EXPECTED_CONVERTER"
test "$(sha256sum "$LLAMA/build/bin/llama-quantize" | cut -d' ' -f1)" = "$EXPECTED_QUANTIZER"
test -d "$TARGET_METADATA"; test ! -e "$OUT"; mkdir "$OUT"
/home/admin/.venv-comfy/bin/python "$LLAMA/convert_hf_to_gguf.py" "$SOURCE" \
  --target-model-dir "$TARGET_METADATA" --outfile "$BF16.partial" --outtype bf16
mv "$BF16.partial" "$BF16"
"$LLAMA/build/bin/llama-quantize" "$BF16" "$Q4.partial" Q4_K_M
mv "$Q4.partial" "$Q4"
sha256sum "$SOURCE/model.safetensors" "$LLAMA/convert_hf_to_gguf.py" \
  "$LLAMA/build/bin/llama-quantize" "$BF16" "$Q4" > "$OUT/export-hashes.sha256"
find "$OUT" -name '*.partial' -print -quit | grep -q . && { echo "partial export artifact remains" >&2; exit 1; }
echo '{"selected_export_complete":true}'
