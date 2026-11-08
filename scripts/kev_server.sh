#!/usr/bin/env bash
# Kev's reference server on 127.0.0.1:8008 (POST /v1/systemone): the sidecar's default engine
# (KEV_ENGINE=http). Python; MLX on Apple Silicon, CUDA or CPU elsewhere.
#
# Serves fully offline: Kev reads the verified weights in models/ through a link-only Hugging
# Face cache in .kev-server/hf-cache, so nothing is downloaded again.
#
#   scripts/kev_server.sh                     # KEV_MODEL from env, default kev-4b
#   KEV_MODEL=kev-0.8b scripts/kev_server.sh  # smaller Kev for <32 GB machines
#   scripts/kev_server.sh --setup-only        # clone + install deps + link cache, don't serve
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
KEV_MODEL="${KEV_MODEL:-kev-4b}"
KEV_COMMIT=2074cc44256200bb37540103fa4572901faa10f3   # github.com/jaredpalmer/kev main, 2026-10-22
DIR="$ROOT/.kev-server"
CACHE="$DIR/hf-cache"

command -v uv >/dev/null || { echo "need uv: brew install uv" >&2; exit 1; }

# Model and base locations from models.json (values shell-quoted by Python).
MODEL_VARS="$(python3 -c '
import json, shlex, sys
models = json.load(open(sys.argv[1]))
if sys.argv[2] not in models: sys.exit(f"{sys.argv[2]!r} not in models.json")
kev = models[sys.argv[2]]; base = models[kev["requires"][0]]
for key, value in {"KEV_DEST": kev["dest"], "BASE_REPO": base["repo"],
                   "BASE_REV": base["revision"], "BASE_DEST": base["dest"]}.items():
    print(f"{key}={shlex.quote(value)}")
' "$ROOT/models.json" "$KEV_MODEL")"
eval "$MODEL_VARS"

# --- Setup (network on first run only) ---
[ -f "$ROOT/$KEV_DEST/head.pt" ] || python3 "$ROOT/scripts/fetch_model.py" "$KEV_MODEL"
[ -d "$DIR/.git" ] || git clone --quiet https://github.com/jaredpalmer/kev.git "$DIR"
git -C "$DIR" checkout --quiet "$KEV_COMMIT"
(cd "$DIR" && uv sync --quiet --extra serve)

# Kev loads its base by Hub id at the revision stored in head.pt (= models.json pin).
# Point that cache entry at our verified files instead of downloading a second copy.
REPO_CACHE="$CACHE/models--${BASE_REPO//\//--}"
mkdir -p "$REPO_CACHE/snapshots/$BASE_REV" "$REPO_CACHE/refs"
for f in "$ROOT/$BASE_DEST"/*; do ln -sfn "$f" "$REPO_CACHE/snapshots/$BASE_REV/$(basename "$f")"; done
printf '%s' "$BASE_REV" > "$REPO_CACHE/refs/main"

[ "${1:-}" = "--setup-only" ] && { echo "kev server ready: $KEV_MODEL (run scripts/kev_server.sh to serve)"; exit 0; }

# --- Serve (offline): loopback only, no Hub calls, no telemetry ---
export HF_HUB_CACHE="$CACHE" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
cd "$DIR"
exec uv run --offline --extra serve python -m kev.serve \
  --run "$ROOT/$KEV_DEST" --host 127.0.0.1 --port "${KEV_PORT:-8008}"
