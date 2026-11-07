# Architecture & Key Decisions

Why the sidecar is built the way it is. For contracts and tickets see `design-doc.md`.

## Architectural keys

Invariants. Breaking one breaks the security model.

1. **Single choke point.** Backend accepts only sidecar-signed requests. Agent has no other path in.
2. **Default deny.** Tool not in `policy.toml` = denied. No implicit allows.
3. **Policy decides, model advises.** Model returns scores; fixed rules + thresholds turn them into allow/deny. Same inputs → same decision.
4. **Fail closed.** Model timeout, model error, bad policy file → deny or refuse to start. Never allow on uncertainty.
5. **Model is swappable.** Sidecar depends only on `DecisionModel` trait, never a concrete model.
6. **Contracts live in one place.** `crates/common` holds types, trait, signing. Other crates never redefine them.
7. **Never block the async runtime.** CPU-heavy inference runs in `spawn_blocking`.
8. **Never log argument values.** Args may carry secrets or customer data. Log tool, decision, rule, scores, latency only.

## Critical decisions

### Rust
- **Why:** memory safety without GC, predictable latency, single static binary next to the agent.
- **Cost:** slower to write, steeper async learning curve.

### Plain HTTP JSON (not MCP / OpenAI format)
- **Why:** smallest protocol that proves the idea. Testable with `curl`.
- **Cost:** agents need a small adapter to call us.
- **Revisit:** MCP adapter is the first post-MVP feature.

### Deterministic policy over model verdict
- **Why:** security boundary must be explainable and reproducible. A probability can't be audited; "`to` must match `@ourcompany.com`" can.
- **How:** policy rules run first in decision order, so exact reasons win over model scores when both fail.

### Own TOML policy engine (not Cedar / OPA)
- **Why:** MVP rules are allowlist, required args, regex, max_len. ~150 lines anyone can read fully.
- **Detail:** `regex` crate is linear-time, so hostile args can't trigger catastrophic backtracking.
- **Revisit:** need roles/principals → Cedar (Rust-native).

### HMAC-SHA256 signing (not mTLS / JWT)
- **Proves:** request came from sidecar, body unchanged, made in last 30 s.
- **Does not prove:** request is safe. Only as good as the policy that approved it.
- **Detail:** constant-time compare, otherwise timing leaks matching bytes.
- **Gap:** replay possible inside 30 s window → nonce cache later.
- **Revisit:** sidecar and backend on different machines → mTLS.

### Decision model in-process via `ort` (ONNX Runtime)
- **Why:** no network hop, one binary, ONNX is the most common export format so most small classifiers (Kev, if exportable) run unchanged.
- **Cost:** ONNX Runtime is C++; a crash in it kills the sidecar. Bigger binary.
- **Rejected:** `candle` / `tract` (pure Rust but narrower model support), separate model server (extra process + hop; kept as `HttpModel` fallback).
- **Revisit:** if chosen model runs in `candle`, drop the C++ dependency.

### Parallel policy + model
- **Why:** policy check is microseconds; running it first saves nothing. Parallel also logs both outcomes for threshold tuning.
- **Honest limit:** model is still on the critical path. Sidecar latency ≈ model latency. Measure, don't claim "zero latency".

### Two model inputs, one batched call
- `untrusted_inputs` → `injection_confidence` (does read content try to give orders?)
- `user_prompt + tool + args` → `malicious_intent` (does the action stray from what user asked?)
- **Why batched:** two signals for roughly the cost of one inference.

### Windowed scoring of long input
- **Why:** truncating to first N tokens lets attackers hide injection at the end. Score overlapping windows, keep max.

### Timeouts + concurrency cap
- `tokio::time::timeout` stops *waiting*, not inference. Work continues in background.
- **So:** semaphore caps concurrent inferences, else a request flood piles up unbounded CPU work.

### Cargo workspace, 4 crates
- **Why:** compiler enforces boundaries. Sidecar physically can't reach model internals. Separate folders = fewer merge conflicts.

### Audit = JSON lines on stdout
- **Why:** zero infra, pipe to `jq` / file / collector later.

## Known non-goals (MVP)

Scanning tool results returned to agent · session/taint tracking across calls · per-agent identity · MCP. All real attack surface; deliberately deferred.
