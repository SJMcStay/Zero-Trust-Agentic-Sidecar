#!/usr/bin/env bash
# Claude Code with its LLM traffic routed through the sidecar. Arguments go to `claude`.
#
#   scripts/claude.sh                          # interactive
#   scripts/claude.sh -p "summarize notes.md"  # one prompt
#
# Uses Claude Code's own login unless the sidecar holds ANTHROPIC_API_KEY (.env).
set -euo pipefail
[ "${ZT_DEBUG:-0}" != 0 ] && set -x
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

addr="${SIDECAR_ADDR:-}"
if [ -z "$addr" ] && [ -f "$ROOT/.env" ]; then
  addr="$(sed -n 's/^SIDECAR_ADDR=//p' "$ROOT/.env" | tail -1)"
fi
addr="${addr:-127.0.0.1:8080}"

command -v claude >/dev/null || { echo "missing claude: https://docs.claude.com/en/docs/claude-code" >&2; exit 1; }
if ! curl -fsS --max-time 2 "http://$addr/healthz" >/dev/null; then
  echo "sidecar not answering on $addr; start it first (USAGE.md)" >&2
  exit 1
fi

export ANTHROPIC_BASE_URL="http://$addr"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1   # no telemetry, error reports or update checks
exec claude "$@"
