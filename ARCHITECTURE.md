# Architecture & Key Decisions
**Zero-Trust Agentic Sidecar — Enterprise AI Gateway**

This document defines the architectural invariants, component boundaries, and key engineering decisions for the Zero-Trust Agentic Sidecar as described in [README.md](./README.md). For API schemas, model internals, and implementation tickets, see [design-doc.md](./design-doc.md).

---

## 1. System Topology

```text
                       Route A: POST /v1/messages (Prompt Interception + Kev/Jev ML Eval)
+---------------------+ -----------------------------------------------------------> +----------------------------------+
|                     |                                                              |   Zero-Trust Agentic Sidecar     |
|  Client / Agent CLI |                                                              |    (Rust / Axum / Candle ML)     |
| (Claude Code / Qwen)|                                                              |          localhost:8080          |
+---------------------+ -----------------------------------------------------------> +----------------------------------+
                       Route B: POST /api/execute (JSON Tool-Call Interception)            |                      |
                                                                                           | Safe + Inject        | Authorized + Signed
                                                                                           | x-api-key            | Zero-Trust Policy
                                                                                           v                      v
                                                                                +---------------------+  +----------------------+
                                                                                |   Anthropic API     |  | Target Backend API   |
                                                                                |  (External LLM)     |  | (Python Flask :5000) |
                                                                                +---------------------+  +----------------------+
```

---

## 2. Architectural Invariants

Breaking any of these invariants compromises the zero-trust security posture:

1. **Dual-Route Interception (Inbound Prompts + Outbound Tool Calls)**:
   - **Route A (`POST /v1/messages`)** intercepts raw agent/user prompts *before* they reach the external LLM (`api.anthropic.com` or local Ollama) and evaluates them with the in-process Candle Kev/Jev System One decision model.
   - **Route B (`POST /api/execute`)** intercepts structured JSON tool-calls *exiting* the agent, evaluating both deterministic zero-trust policy rules and Kev/Jev intent/injection probabilities before any backend API is touched.
2. **In-Process Native Rust ML via Candle + Metal (No External ML Microservice)**:
   - The Kev/Jev System One decision model (`.safetensors`) is downloaded directly inside Rust via `hf-hub` and loaded into the proxy's unified memory space at startup using `candle-core` and `candle-nn` on Apple Silicon's Metal backend (`Device::new_metal(0)` with CPU fallback).
   - No Python bridge or network hop is used for ML inference.
3. **Zero Secret Exposure to the Client or Agent**:
   - The upstream LLM credential (`ANTHROPIC_API_KEY`) and the backend signing key (`SIDECAR_SECRET` / `JWT_SECRET`) reside exclusively inside the compiled Rust sidecar's memory space. The agent CLI only ever points `ANTHROPIC_BASE_URL="http://localhost:8080"` at the sidecar.
4. **Deterministic Policy Decides, System One Model Advises**:
   - Kev returns calibrated `noul` probabilities (`injection_confidence` and `malicious_intent` in `[0.0, 1.0]`). Deterministic thresholds and explicit endpoint rules (`policy.toml`) turn those scores and structural constraints into reproducible allow/deny decisions.
5. **Cryptographic Choke Point (Two-Legged Verification)**:
   - **Agent → Sidecar (Route B)**: Verifies caller identity, permissions, and payload integrity via JWT (`Authorization: Bearer <token>`) or client HMAC signature.
   - **Sidecar → Python Flask Backend (`localhost:5000`)**: The sidecar signs every approved tool request with `X-Sidecar-Timestamp` and `X-Sidecar-Signature = hex(HMAC-SHA256(SIDECAR_SECRET, ts + "." + body))`. The Flask backend rejects unsigned, tampered, or stale (`> 30s`) requests with `401 Unauthorized`.
6. **Fail Closed**:
   - Missing secrets, invalid policy configuration, malformed payloads, unverifiable JWTs/signatures, or ML inference errors/timeouts always result in `403 Forbidden` (or startup abort). Uncertainty never defaults to allow.
7. **Never Block the Tokio Async Runtime**:
   - Candle tensor operations on Metal/CPU are executed inside `tokio::task::spawn_blocking` bounded by a concurrency `Semaphore` so HTTP routing and streaming remain non-blocking.
8. **Zero-Trust Audit Logging**:
   - Every Route A and Route B interception emits a structured JSON audit line recording the route, target endpoint, caller sub/role, Kev `injection_confidence` and `malicious_intent` floats, matched rule, decision, and latency—without leaking raw API keys or sensitive argument payloads.

---

## 3. Critical Engineering Decisions

### 3.1 High-Performance Single-Crate Rust Core (`Axum` + `Tokio` + `Reqwest`)
- **Why**: Eradicates garbage collection (GC) latency spikes and I/O bottlenecks compared to Go or Python proxies. Compiles to a single binary (`cargo build --release` / `cargo run --release`) sitting directly alongside the local agent.
- **Streaming Support**: On Route A (`POST /v1/messages`), once Kev clears the prompt, `reqwest` injects `x-api-key` and streams SSE chunks (`text/event-stream`) or JSON responses directly back to Claude Code CLI.

### 3.2 Native Rust ML Engine (`candle-core` + `candle-nn` + `hf-hub` + `.safetensors`)
- **Why**: Hugging Face `candle-core` and `candle-nn` provide native Apple Silicon Metal acceleration inside Rust's unified memory space without a Python runtime or C++ ONNX Runtime dependency.
- **Direct In-Rust Model Download**: At startup, the Rust binary uses `hf-hub` (`hf_hub::api::sync::Api`) to pull the open-source Kev System One checkpoint (`jaredpalmer/kev-0.6b` / `jaredpalmer/kev-0.8b` `.safetensors` adapter, base `.safetensors`, `tokenizer.json`, and `PointerHead` weights) directly from Hugging Face into local cache (`models/`).
- **How Kev System One Inference Works in Candle**:
  - Kev is a prefill-only System One decision model (`output_tokens = 0`): it encodes the state and typed `noul` (yes/no probability) questions using Qwen special delimiter tokens (`<|fim_prefix|>` for state, `<|fim_middle|>` for question instruction, `<|box_start|>` / `<|box_end|>` for `"no"` and `"yes"` options, and `<|fim_suffix|>` for the `<decide>` readout position).
  - The LoRA adapter tensors (`adapter_model.safetensors`, `r=16, alpha=32`) are merged into the base weights in `candle-nn` at load time (`W_merged = W_base + (alpha / r) * (B @ A)`), eliminating runtime LoRA overhead.
  - A single forward pass produces hidden states `H`; the calibrated `PointerHead` (`q` and `k` linear projections of dim `256` with temperature scaling) scores the `<decide>` vector against the `"no"` and `"yes"` `</opt>` vectors via softmax to yield calibrated probabilities in `~4ms`.

### 3.3 Dual-Signal Interrogation (`injection_confidence` + `malicious_intent`)
- **Signal 1 (`injection_confidence`)**: Evaluates whether untrusted text or inbound prompt content attempts to override instructions, spoof system roles, or hijack control flow. Long texts are evaluated across overlapping sliding windows (retaining the maximum window probability) so attackers cannot hide injections past truncation boundaries.
- **Signal 2 (`malicious_intent`)**: Evaluates whether the requested tool execution (`/api/v1/db/drop`, `/api/v1/system/status`, etc.) and its JSON arguments align with authorized user intent or attempt unauthorized destructive/exfiltrative operations.

### 3.4 Deterministic Zero-Trust Policy Engine (`policy.toml`)
- **Why**: Even calibrated System One models can be nudged by adversarial inputs ("Decision Hijacking"). Therefore, Route B pairs Kev's probabilistic scores with hard, auditable policy rules:
  1. **Caller JWT / Signature Verification**: Validates caller identity and role claims (`admin`, `agent_readonly`, etc.).
  2. **Endpoint Allowlist**: Unlisted backend endpoints are denied by default (`/api/v1/db/drop` is blocked for standard agent roles).
  3. **Argument Schema & Regex Constraints**: Enforces required keys, maximum string lengths, and linear-time `regex` patterns before forwarding.
  4. **Kev Threshold Gate**: Enforces `injection_confidence < max_injection_confidence` and `malicious_intent < max_malicious_intent`.

### 3.5 Python Flask Target Backend (`backend/app.py` on `localhost:5000`)
- **Why**: Simulates internal enterprise microservices with both safe endpoints (`/api/v1/system/status`) and high-risk/vulnerable endpoints (`/api/v1/db/drop`).
- **Choke-Point Enforcement**: Includes a Flask `@app.before_request` cryptographic verifier that checks `X-Sidecar-Timestamp` (30-second window) and constant-time `hmac.compare_digest` on `X-Sidecar-Signature`. Direct requests from a compromised agent to `localhost:5000` are rejected with `401 Unauthorized`.
