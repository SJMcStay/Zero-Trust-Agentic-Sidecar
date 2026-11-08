#!/usr/bin/env bash
# Export a Hugging Face model dir (safetensors) to ONNX for the `ort` runtime.
# Skip if the repo already ships onnx/model.onnx.
#
#   scripts/export_onnx.sh [models/kev]
#   TASK=text-classification (default) — set to match Kev's architecture.
#
# Needs optimum-cli: pip install optimum-onnx   (older releases: pip install "optimum[onnxruntime]")
set -euo pipefail
cd "$(dirname "$0")/.."

SRC="${1:-models/kev}"
TASK="${TASK:-text-classification}"

[ -f "$SRC/config.json" ] || { echo "no $SRC/config.json: run scripts/fetch_model.py first" >&2; exit 1; }
[ -f "$SRC/onnx/model.onnx" ] && { echo "already present: $SRC/onnx/model.onnx"; exit 0; }
command -v optimum-cli >/dev/null || { echo "optimum-cli missing: pip install optimum-onnx" >&2; exit 1; }

optimum-cli export onnx --model "$SRC" --task "$TASK" "$SRC/onnx"
echo "ok: $SRC/onnx/model.onnx (re-run scripts/fetch_model.py --lock to record its hash)"
