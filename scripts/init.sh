#!/usr/bin/env bash
# One-time setup. Safe to re-run: each step skips work already done.
#
#   scripts/init.sh                        # KEV_MODEL=kev-4b (default): ~10 GB disk, 32 GB RAM / 16 GB GPU
#   KEV_MODEL=kev-0.8b scripts/init.sh     # smaller Kev for 16 GB machines: ~2 GB disk
#   ZT_FEATURES=cuda scripts/init.sh       # Linux: build the Candle engine for CUDA (needs the CUDA toolkit)
set -euo pipefail
[ "${ZT_DEBUG:-0}" != 0 ] && set -x   # ZT_DEBUG=1: print each command as it runs
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
KEV_MODEL="${KEV_MODEL:-kev-4b}"

step() { printf '\n==> %s\n' "$*"; }
need() { command -v "$1" >/dev/null || { echo "missing $1: $2" >&2; exit 1; }; }

step "Tools"
need cargo "install Rust from https://rustup.rs, then: . \"\$HOME/.cargo/env\""
need python3 "install Python 3.8+"
need uv "brew install uv (runs Kev's reference server)"
need git "install git"
need openssl "install openssl"

step "Hardware"
if [ "$(uname)" = Darwin ]; then
  ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
  echo "macOS, RAM ${ram_gb} GB (unified with the GPU); Kev runs on MLX"
  if [ "$KEV_MODEL" = kev-4b ] && [ "$ram_gb" -lt 32 ]; then
    echo "warning: kev-4b is sized for 32 GB (about 12 GB while serving). Consider KEV_MODEL=kev-0.8b."
  fi
  CARGO_FLAGS=""                                   # default feature: metal
else
  ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 ))
  if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
    vram_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
    echo "Linux, RAM ${ram_gb} GB, NVIDIA GPU with ${vram_mb} MiB; Kev runs on CUDA"
    if [ "$KEV_MODEL" = kev-4b ] && [ "$vram_mb" -lt 16000 ]; then
      echo "warning: kev-4b needs ~14.3 GB of GPU memory. Consider KEV_MODEL=kev-0.8b."
    fi
  else
    echo "Linux, RAM ${ram_gb} GB, no NVIDIA GPU found; Kev runs on CPU (slow)"
  fi
  CARGO_FLAGS="--no-default-features ${ZT_FEATURES:+--features $ZT_FEATURES}"   # metal is macOS-only
fi

step ".env (fresh secrets, never committed)"
if [ -f .env ]; then
  echo ".env exists; leaving it"
else
  python3 - "$KEV_MODEL" "$(openssl rand -hex 32)" "$(openssl rand -hex 32)" <<'EOF'
import re, sys
model, sidecar, jwt = sys.argv[1:]
env = open(".env.example").read()
values = {"SIDECAR_SECRET": sidecar, "JWT_SECRET": jwt, "KEV_MODEL": model, "POLICY_PATH": "policy.local.toml"}
for key, value in values.items():
    env = re.sub(rf"^{key}=.*$", f"{key}={value}", env, flags=re.M)
open(".env", "w").write(env)
EOF
  chmod 600 .env
  echo "wrote .env (Claude Code's own login is used; set ANTHROPIC_API_KEY to have the sidecar hold a key)"
fi

step "policy.local.toml (your settings; policy.toml keeps the defaults the tests use)"
if [ -f policy.local.toml ]; then
  echo "policy.local.toml exists; leaving it"
else
  # Real Kev needs more than 500 ms; agents need only the new turn scored and a readable block.
  sed -e 's/^model_timeout_ms = .*/model_timeout_ms = 5000/' \
      -e 's/^score = "all"/score = "new"/' \
      -e 's/^on_block = "error"/on_block = "message"/' \
      -e 's/^desktop = false/desktop = true/' policy.toml > policy.local.toml
  echo "wrote policy.local.toml (score = new, on_block = message, desktop alerts, 5 s timeout)"
fi

step "Model weights ($KEV_MODEL, pinned + sha256-verified)"
python3 scripts/fetch_model.py "$KEV_MODEL"

step "Kev reference server (clone, Python deps, link-only cache)"
KEV_MODEL="$KEV_MODEL" scripts/kev_server.sh --setup-only

step "Python venv for the Flask backend and e2e tests (.venv)"
[ -d .venv ] || uv venv --quiet .venv
uv pip install --quiet --python .venv/bin/python -r backend/requirements.txt

step "Build (release) ${CARGO_FLAGS:-with default features}"
# shellcheck disable=SC2086  # CARGO_FLAGS is a list of flags
cargo build --release $CARGO_FLAGS

cat <<EOF

Done. Run (see USAGE.md):
  scripts/kev_server.sh                                              # 1: Kev on 127.0.0.1:8008
  set -a && . ./.env && set +a && .venv/bin/python backend/app.py   # 2: backend on :5000
  set -a && . ./.env && set +a && cargo run --release $CARGO_FLAGS  # 3: sidecar on :8080
  scripts/claude.sh                                                  # 4: Claude Code through the sidecar
EOF
