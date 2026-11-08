#!/usr/bin/env bash
# Delete locally generated files. Never touches tracked files, .env, Rust or uv.
#
#   scripts/cleanup.sh               # builds, Kev server checkout, venv, Python caches
#   scripts/cleanup.sh --models      # also downloaded weights (models/)
#   scripts/cleanup.sh --hf-cache    # also Hugging Face cache copies of models in models.json
#   scripts/cleanup.sh --all         # all of the above
#   add --dry-run to only list, --yes to skip the prompt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_HUB="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}/hub}"

models=0 hf=0 dry=0 yes=0
for arg in "$@"; do
  case "$arg" in
    --models) models=1 ;;
    --hf-cache) hf=1 ;;
    --all) models=1 hf=1 ;;
    --dry-run) dry=1 ;;
    --yes) yes=1 ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

targets=("$ROOT/target" "$ROOT/.kev-server" "$ROOT/.venv" "$ROOT/.pytest_cache" "$ROOT/backend/__pycache__")
[ "$models" = 1 ] && targets+=("$ROOT/models")
if [ "$hf" = 1 ]; then
  # Hub cache layout: models--<org>--<name>
  while read -r repo; do targets+=("$HF_HUB/models--${repo//\//--}"); done < <(
    python3 -c "import json; [print(m['repo']) for m in json.load(open('$ROOT/models.json')).values()]")
fi

found=()
for t in "${targets[@]}"; do
  [ -e "$t" ] && found+=("$t") && printf '%8s  %s\n' "$(du -sh "$t" 2>/dev/null | cut -f1)" "$t"
done
[ ${#found[@]} -eq 0 ] && { echo "nothing to clean"; exit 0; }
[ "$dry" = 1 ] && exit 0

if [ "$yes" != 1 ]; then
  read -r -p "Delete the above? [y/N] " answer
  [[ "$answer" =~ ^[Yy]$ ]] || { echo "aborted"; exit 1; }
fi
for t in "${found[@]}"; do rm -rf -- "$t" && echo "deleted $t"; done
