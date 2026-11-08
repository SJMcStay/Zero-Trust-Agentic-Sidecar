#!/usr/bin/env bash
# Runs an agent CLI with its LLM traffic sent through the sidecar. The environment comes from
# [agent_cli_mappings.<cli>] in the policy (POLICY_PATH); only the Anthropic protocol is served
# today, so other CLIs are refused rather than run uninspected. Arguments go to the CLI.
#
#   scripts/agent.sh claude                          # Claude Code, interactive
#   scripts/agent.sh claude -p "summarize notes.md"  # one prompt
set -euo pipefail
[ "${ZT_DEBUG:-0}" != 0 ] && set -x
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cli="${1:?usage: scripts/agent.sh <cli> [args...]   (CLIs: [agent_cli_mappings] in the policy)}"
shift

# Only POLICY_PATH is read from .env: the agent must not inherit the sidecar's secrets.
policy="${POLICY_PATH:-}"
if [ -z "$policy" ] && [ -f "$ROOT/.env" ]; then
  policy="$(sed -n 's/^POLICY_PATH=//p' "$ROOT/.env" | tail -1)"
fi
policy="${policy:-policy.toml}"
case "$policy" in /*) ;; *) policy="$ROOT/$policy" ;; esac

bin=""
for candidate in "$ROOT/target/release/zero-trust-agentic-sidecar" "$ROOT/target/debug/zero-trust-agentic-sidecar"; do
  if [ -x "$candidate" ]; then bin="$candidate"; break; fi
done
[ -n "$bin" ] || { echo "build the sidecar first: cargo build --release" >&2; exit 1; }

env_lines="$(POLICY_PATH="$policy" "$bin" env "$cli")"   # prints why and exits 2 if refused
command -v "$cli" >/dev/null || { echo "missing $cli on PATH" >&2; exit 1; }

base_url="$(printf '%s\n' "$env_lines" | head -1 | cut -d= -f2-)"
if ! curl -fsS --max-time 2 "${base_url%/v1}/healthz" >/dev/null; then
  echo "sidecar not answering at $base_url; start it first (USAGE.md)" >&2
  exit 1
fi

while IFS='=' read -r key value; do
  [ -n "$key" ] && export "$key=$value"
done <<< "$env_lines"
exec "$cli" "$@"
