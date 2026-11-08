# Proposal: alert the human, keep the agent talking, work with any agent

Status: AL1–AL3 are implemented for borderline Route B calls; G1 and AD1–AD3 remain open.

## Today

A deny is an HTTP 403. Claude Code prints `API Error: 403 … permission_error` and the turn ends. Only the audit log records what was blocked and why.

## Goal

1. The human sees a pop-up: "Claude Code tried to upload `~/.ssh/id_rsa`; blocked: prompt injection in a WebFetch result (Kev 0.91)".
2. The agent gets an answer in its own protocol and carries on safely.
3. The same for any agent and any model, including uncensored local models: they refuse nothing themselves, so the sidecar is the only guard.

## Design

```text
Claude Code · Codex · Antigravity · aider / Open Interpreter on Ollama, LM Studio, vLLM
   │ API traffic (agent's base URL → sidecar)
   ▼
protocol adapter: Anthropic Messages | OpenAI Chat / Responses | Ollama | Gemini
   │ normalize: new untrusted content, outbound tool calls, agent + session id
   ▼
core: policy + Kev → verdict
   │ deny
   ├─► alert bus: GET /v1/alerts (SSE, loopback) ─► pop-up app / notifier / dashboard
   └─► adapter: an in-protocol reply instead of a 403
```

### Alerts

- **AL1** Alert bus: the sidecar keeps recent alerts in memory and streams them on `GET /v1/alerts` (SSE, loopback, token). Event: time, agent (User-Agent, `x-claude-code-session-id`), route, what (tool + argument summary or text snippet), why (rule, scores), action.
- **AL2** Pop-up: first a small notifier subscribed to the stream that raises native notifications (macOS UserNotifications or `osascript`, Linux `notify-send`); then a tray app (Tauri: Rust + web UI) with history and details.
- **AL3** (optional) Human in the loop: hold a borderline request up to N seconds while the pop-up offers *Allow once* / *Block*; no answer means block (fail closed).

### An answer instead of a 403

- **Bad tool output (inbound, R1):** don't forward; return a normal assistant message, streamed if the request asked for streaming: "The Zero-Trust Sidecar blocked the last WebFetch result: it looked like a prompt injection (Kev 0.91). Nothing in it was acted on." `stop_reason: end_turn`. The CLI shows it like any reply; the human decides what next.
- **Bad tool call (outbound, R2):** replace that `tool_use` block in the stream with a text block saying why; `stop_reason: end_turn`, so the CLI never runs it.
- Never rewrite the request body (Claude Code's gateway guidance).

### Agent-agnostic adapters

| Agent / model | Wire protocol | Points at the sidecar via | Status |
| --- | --- | --- | --- |
| Claude Code | Anthropic Messages | `ANTHROPIC_BASE_URL` | Route A today; needs G1 |
| Codex CLI | OpenAI Responses | custom model provider `base_url` in Codex config | to verify (AD2) |
| Ollama, LM Studio, llama.cpp, vLLM (uncensored models) and agents on them | OpenAI-compatible `/v1/chat/completions` (Ollama also `/api/chat`) | the agent's base URL → sidecar → local model server | AD1 |
| Antigravity, Gemini CLI | Gemini API | custom base URL if supported, else an HTTPS proxy with a local CA | to verify (AD3) |

Each adapter maps: request → new untrusted content (R1 rule); response stream → tool calls (R2); deny → in-protocol reply.

## Path to running in a real CLI (Claude Code first)

1. **G1 gateway compliance** (Claude Code's LLM-gateway docs): accept `/v1/messages?beta=true`; pass through `/v1/messages/count_tokens`, `GET /v1/models`, `HEAD /api/hello`; forward every `anthropic-*` header unchanged; return `x-should-retry` and `anthropic-ratelimit-unified-*`. Auth: a sidecar-held key, or pass the client's `Authorization` through (a claude.ai login needs its OAuth `anthropic-beta` value forwarded).
2. **R1** score only new content: latency and false positives.
3. **Answer instead of 403** + **AL1/AL2** pop-up.
4. **R2** outbound `tool_use` check: the real protection, since the CLI runs Bash itself.
5. **One command** that starts the Kev server and the sidecar and sets `ANTHROPIC_BASE_URL` and `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1` for the CLI; later a launchd / systemd service.
6. **Model quality**: Kev-4B on a 32 GB Mac or a CUDA box; an eval set (E1) to set thresholds.

Rough order: G1 + R1 make a Claude Code demo possible; then the answer + pop-up; then R2; then one adapter at a time.
