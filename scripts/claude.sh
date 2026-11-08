#!/usr/bin/env bash
# Claude Code with its LLM traffic sent through the sidecar ([agent_cli_mappings.claude]).
# Arguments go to `claude`:  scripts/claude.sh  |  scripts/claude.sh -p "summarize notes.md"
exec "$(dirname "$0")/agent.sh" claude "$@"
