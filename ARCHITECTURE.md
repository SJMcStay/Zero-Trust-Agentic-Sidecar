# Architecture & Key Decisions

Why the sidecar is built the way it is. For contracts and tickets see `design-doc.md`.

## Architectural keys

Invariants. Breaking one breaks the security model.

1. **Single choke point.** Backend accepts only sidecar-signed requests. Agent has no other path in.
2. **Thresholds decide, model advises.** Model returns scores; fixed thresholds turn them into allow/deny. Same scores → same decision.
3. **Fail closed.** Model timeout, model error, bad config → deny or refuse to start. Never allow on uncertainty.
4. **Model is swappable.** Sidecar depends only on `DecisionModel` trait, never a concrete model.
5. **Contracts live in one place.** `crates/common` holds types, trait, signing. Other crates never redefine them.
6. **Never block the async runtime.** CPU-heavy inference runs in `spawn_blocking`.
7. **Never log argument values.** Args may carry secrets or customer data. Log tool, decision, rule, scores, latency only.

## Critical decisions

### Rust
- **Why:** memory safety without GC, predictable latency, single static binary next to the agent.
- **Cost:** slower to write, steeper async learning curve.

### Plain HTTP JSON (not MCP / OpenAI format)
- **Why:** smallest protocol that proves the idea. Testable with `curl`.
- **Cost:** agents need a small adapter to call us.
- **Revisit:** MCP adapter is the first post-MVP feature.

### No policy check for now (model only)
- **Why:** smallest path to a working sidecar; model + thresholds is the core idea.
- **Cost:** no default deny. Any tool passes if scores are low; decisions are only as good as the model.
- **Revisit:** add a tool allowlist + arg rules (own TOML engine first, Cedar if roles needed) as a check before the model.

### HMAC-SHA256 signing (not mTLS / JWT)
- **Proves:** request came from sidecar, body unchanged, made in last 30 s.
- **Does not prove:** request is safe. Only as good as the model check that approved it.
- **Detail:** constant-time compare, otherwise timing leaks matching bytes.
- **Gap:** replay possible inside 30 s window → nonce cache later.
- **Revisit:** sidecar and backend on different machines → mTLS.

### Kev via safetensors + `candle` (not ONNX)
- **Why:** Kev ships safetensors (base, LoRA) and a PyTorch `head.pt`; candle reads both, is pure Rust, and uses Metal on Macs.
- **Rejected:** ONNX/`ort` for Kev. Exporting Qwen3.5's Gated DeltaNet is unproven, and Kev runs one row per question, which a fixed export graph fights. `ort` stays only for `OnnxClassifier`.
- **Cost:** candle has no Qwen3.5, so we port the backbone (tickets K1–K5) with parity tests against Kev's Python reference.
- **Bridge:** until then, Kev's reference server on localhost + `KevHttp` (one local hop).

### Latency
- Model is on the critical path of every call. Sidecar latency ≈ model latency. Measure, don't claim "zero latency".

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
