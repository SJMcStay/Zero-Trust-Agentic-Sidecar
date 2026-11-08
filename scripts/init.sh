#!/usr/bin/env bash
# One-time setup. Safe to re-run: each step skips work already done.
#
#   scripts/init.sh                        # KEV_MODEL=kev-4b (default): ~10 GB disk, 32 GB RAM
#   KEV_MODEL=kev-0.8b scripts/init.sh     # smaller Kev for 16 GB machines: ~2 GB disk
set -euo pipefail
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

step "Memory"
if [ "$(uname)" = Darwin ]; then ram_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
else ram_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1048576 )); fi
echo "RAM: ${ram_gb} GB, model: $KEV_MODEL"
if [ "$KEV_MODEL" = kev-4b ] && [ "$ram_gb" -lt 32 ]; then
  echo "warning: kev-4b is sized for 32 GB (about 12 GB while serving). Consider KEV_MODEL=kev-0.8b."
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
  echo "wrote .env (set ANTHROPIC_API_KEY in it for Route A)"
fi

step "policy.local.toml (real-model timeout; policy.toml keeps 500 ms for the tests)"
if [ -f policy.local.toml ]; then
  echo "policy.local.toml exists; leaving it"
else
  sed 's/^model_timeout_ms = .*/model_timeout_ms = 5000/' policy.toml > policy.local.toml
  echo "wrote policy.local.toml (model_timeout_ms = 5000; raise it if you see model_unavailable)"
fi

step "Model weights ($KEV_MODEL, pinned + sha256-verified)"
python3 scripts/fetch_model.py "$KEV_MODEL"

step "Kev reference server (clone, Python deps, link-only cache)"
KEV_MODEL="$KEV_MODEL" scripts/kev_server.sh --setup-only

step "Python venv for the Flask backend and e2e tests (.venv)"
[ -d .venv ] || uv venv --quiet .venv
uv pip install --quiet --python .venv/bin/python -r backend/requirements.txt

step "Build (release)"
cargo build --release

cat <<EOF

Done. Run (see USAGE.md):
  scripts/kev_server.sh                                   # terminal 1: Kev on 127.0.0.1:8008
  source .venv/bin/activate && python backend/app.py      # terminal 2: backend on :5000
  set -a && . ./.env && set +a && cargo run --release     # terminal 3: sidecar on :8080
EOF
