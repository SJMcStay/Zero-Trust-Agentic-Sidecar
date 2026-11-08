# Proposal: score only new content (Route A)

Status: proposal, ticket R1 in [design-doc.md](../design-doc.md).

## Problem

Every Claude Code request resends the whole conversation: system prompt, tool definitions, every earlier turn. Route A (`extract_prompt_text_from_json` in `src/routes/messages.rs`) scores all of it, every turn:

- cost grows with the conversation; long states blow `model_timeout_ms`, and the request is denied (fail closed);
- the system prompt is full of instructions, so it looks like an injection (false positives);
- old content is re-scored on every turn.

Kev's own numbers (model cards, MLX on an M5): an 8k-token state takes 1.4 s (0.8B) / 6.6 s (4B); a short state takes ~70 ms on this M1 Pro (measured).

## Rule

Score only what entered the context since the model last spoke:

- **new** = content blocks in the messages after the last `assistant` message: normally one `user` message holding `tool_result` blocks and any user text;
- **skip** `system` and `tools` (written by the harness) and all earlier turns (each was scored when it was new);
- **stateless**: no session store; each piece of untrusted text is scored exactly once, when it arrives.

## What Kev sees

One request per turn, both questions in one forward pass when Route A also scores intent later:

```json
{"user_request": "<latest user text>",
 "tool_results": [{"tool": "WebFetch", "input": {"url": "https://vendor.example"},
                   "output": "<tool_result text>"}]}
```

- `tool` and `input` come from the matching `tool_use` block (same `tool_use_id`) in the previous assistant message.
- Question: `injection`: do the tool outputs try to instruct the AI?
- Long outputs: the existing 24k-character windows, max kept. Images and other non-text blocks: skipped (logged at `ZT_DEBUG=1`).

## Never modify the request

Claude Code's gateway guide: a gateway that rewrites or redacts request bodies "breaks the pairing … so inspect without modifying". On deny the sidecar does not forward and answers the agent itself (see [alerts-and-agents.md](alerts-and-agents.md)); it never edits a `tool_result` in flight.

## Edge cases

| Case | Handling |
| --- | --- |
| First turn | Only the user's message is new: score it |
| Several tool results in one turn | One Kev request with all of them (windowed if long) |
| No matching `tool_use` | Score the output anyway, `tool: "unknown"` |
| Client retries the same turn | Optional later: skip re-scoring by sha256 of the new content for a few minutes |
| Rewritten earlier turns | Needs control of the harness itself: outside this threat model |

## Changes

- `src/routes/messages.rs`: `extract_new_content(&Value)` used for scoring; keep `extract_prompt_text_from_json` for callers that want everything.
- `src/ml/kev_http.rs`: build the JSON state above; reuse `evaluate_prompt` or add `evaluate_turn`.
- Debug: `A2` prints `new blocks: N tool results, M chars`.
- Tests: Scott's suites assume full-history scoring. Two cases inject via `system` (`tests/adversarial_m4_challenger1.rs`); about 11 multi-turn cases in 4 files inject in an earlier turn. Move each injection into the latest turn so it is still caught.
- Done when: a 50-turn Claude Code session stays under `model_timeout_ms`, and an injected `tool_result` in the latest turn is still blocked.

## Next (R2)

Inbound scoring cannot stop a bad command the model decides to run: Claude Code executes its tools (Bash, file edits) itself. R2 scores the `tool_use` blocks in the model's streamed response before the CLI sees them.
