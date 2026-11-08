#!/usr/bin/env bash
# Run Kev's reference server on 127.0.0.1:8008 (POST /v1/systemone), fully offline.
# Python; MLX on Apple Silicon, CUDA or CPU elsewhere. Dev bridge until the Rust/candle port lands:
# Rust calls it over HTTP through decision_model::KevHttp.
#
#   scripts/kev_server.sh              # first run: clone Kev, install deps, cache base model (network)
#   KEV_PORT=9008 scripts/kev_server.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEV_COMMIT=2074cc44256200bb37540103fa4572901faa10f3   # github.com/jaredpalmer/kev main, 2026-10-22
DIR="$ROOT/.kev-server"
BASE_REPO=$(python3 -c "import json; print(json.load(open('$ROOT/models.json'))['qwen3.5-0.8b-base']['repo'])")
BASE_REV=$(python3 -c "import json; print(json.load(open('$ROOT/models.json'))['qwen3.5-0.8b-base']['revision'])")
export HF_HUB_DISABLE_TELEMETRY=1

command -v uv >/dev/null || { echo "need uv: brew install uv" >&2; exit 1; }

# --- Setup (network): skipped once done ---
[ -f "$ROOT/models/kev/head.pt" ] || python3 "$ROOT/scripts/fetch_model.py"
[ -d "$DIR/.git" ] || git clone --quiet https://github.com/jaredpalmer/kev.git "$DIR"
git -C "$DIR" checkout --quiet "$KEV_COMMIT"
cd "$DIR"
uv sync --quiet --extra serve
# Kev loads its base by Hub id; put that exact revision in the Hugging Face cache now.
uv run --extra serve python -c "from huggingface_hub import snapshot_download as d; \
d('$BASE_REPO', revision='$BASE_REV', allow_patterns=['*.json', '*.safetensors', '*.pt', '*.txt', '*.jinja'])" >/dev/null

# --- Serve (offline): loopback only, no Hub calls ---
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec uv run --offline --extra serve python -m kev.serve --run "$ROOT/models/kev" --host 127.0.0.1 --port "${KEV_PORT:-8008}"
