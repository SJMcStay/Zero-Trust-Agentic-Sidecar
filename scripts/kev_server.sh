#!/usr/bin/env bash
# Run Kev's reference server on 127.0.0.1:8008 (POST /v1/systemone).
# Python; MLX on Apple Silicon, CUDA or CPU elsewhere. Dev bridge until the Rust/candle port lands:
# Rust calls it over HTTP through decision_model::KevHttp.
#
#   scripts/kev_server.sh              # first run clones Kev and installs deps (~minutes)
#   KEV_PORT=9008 scripts/kev_server.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEV_COMMIT=2074cc44256200bb37540103fa4572901faa10f3   # github.com/jaredpalmer/kev main, 2026-10-22
DIR="$ROOT/.kev-server"

command -v uv >/dev/null || { echo "need uv: brew install uv" >&2; exit 1; }
[ -f "$ROOT/models/kev/head.pt" ] || python3 "$ROOT/scripts/fetch_model.py"

[ -d "$DIR/.git" ] || git clone --quiet https://github.com/jaredpalmer/kev.git "$DIR"
git -C "$DIR" checkout --quiet "$KEV_COMMIT"

cd "$DIR"
uv sync --quiet --extra serve
# Local adapter + head (verified by fetch_model.py). The base model comes from the Hugging Face cache.
exec uv run --extra serve python -m kev.serve --run "$ROOT/models/kev" --port "${KEV_PORT:-8008}"
