# Zero-Trust Agentic Sidecar — Technical Design Document

High-performance Rust security proxy (`localhost:8080`) that sits between an AI agent CLI (Claude Code / local Qwen via Ollama) and both external LLM APIs (`api.anthropic.com`) and internal enterprise backends (Python Flask on `localhost:5000`).

---

## 1. End-to-End Request Flows

### Route A: Inbound Message Gateway (`POST /v1/messages`)

```text
Agent CLI (Claude Code / Qwen)
  │  POST http://localhost:8080/v1/messages
  ▼
[Axum Route A Handler]
  ├─► Parse Anthropic/Chat messages payload (extract user & tool_result blocks)
  ├─► Run in-process Candle Kev/Jev (.safetensors on Metal) in parallel:
  │     • injection_confidence (sliding windows over prompt / untrusted blocks)
  │     • malicious_intent     (prompt intent interrogation)
  ├─► Evaluate thresholds (injection_confidence < max_injection, malicious_intent < max_intent)
  │     ├─► FAIL (≥ threshold or model error) ──► 403 Forbidden (JSON error + Audit Log)
  │     └─► PASS (< threshold)
  │           ├─► Inject `x-api-key: $ANTHROPIC_API_KEY` inside Rust memory
  │           └─► Forward via Reqwest to `https://api.anthropic.com/v1/messages` (streams response back)
```

### Route B: Outbound Tool Execution (`POST /api/execute`)

```text
Agent / LLM Tool-Call
  │  POST http://localhost:8080/api/execute
  │  Headers: Authorization: Bearer <JWT> (or X-Agent-Signature)
  ▼
[Axum Route B Handler]
  ├─► Parallel Interrogation (`tokio::join!`):
  │     ├─► 1. Zero-Trust Policy Engine (`policy.toml` + JWT/Signature Auth):
  │     │     • Verify caller JWT / signature & expiration
  │     │     • Check target endpoint allowlist (`/api/v1/system/status` vs `/api/v1/db/drop`)
  │     │     • Check caller role permissions (`allowed_roles`)
  │     │     • Validate required JSON arguments, `max_len`, and linear-time `regex` constraints
  │     └─► 2. In-Process Candle Kev/Jev Model (`.safetensors` on Metal):
  │           • injection_confidence (on `untrusted_inputs` / context)
  │           • malicious_intent     (on `user_prompt` + `endpoint` + `method` + `payload`)
  ├─► Deterministic Decision Order (First Failure Wins):
  │     1. `malformed`          (400 Bad Request)
  │     2. `unauthorized_token` (401/403 — invalid/missing JWT or client signature)
  │     3. `endpoint_denied`    (403 — unlisted endpoint or role not permitted, e.g. `/api/v1/db/drop`)
  │     4. `missing_arg`        (403 — required payload field missing)
  │     5. `constraint_failed`  (403 — regex or max_len violation)
  │     6. `model_unavailable`  (403 — fail-closed on timeout/inference error)
  │     7. `model_risk_high`    (403 — Kev `injection_confidence` or `malicious_intent` ≥ threshold)
  └─► If ALLOWED:
        ├─► Sign request body with `X-Sidecar-Timestamp` + `X-Sidecar-Signature` (HMAC-SHA256)
        └─► Forward via Reqwest to Python Flask Backend (`http://localhost:5000<endpoint>`)
```

---

## 2. Core Contracts & Wire Schemas

### 2.1 Route A: `POST /v1/messages`

**Request** (Anthropic Messages API compatible):
```json
{
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 1024,
  "system": "You are an enterprise assistant.",
  "messages": [
    {
      "role": "user",
      "content": "Check the current system status."
    }
  ],
  "stream": false
}
```

**Blocked Response (`403 Forbidden`)**:
```json
{
  "type": "error",
  "error": {
    "type": "zero_trust_policy_violation",
    "rule": "model_risk_high",
    "message": "Prompt blocked by Kev/Jev System One interrogation",
    "injection_confidence": 0.94,
    "malicious_intent": 0.81
  }
}
```

### 2.2 Route B: `POST /api/execute`

**Request**:
```json
{
  "session_id": "sess_01",
  "endpoint": "/api/v1/system/status",
  "method": "GET",
  "payload": {
    "service": "payments-api"
  },
  "context": {
    "user_prompt": "Check the status of payments-api",
    "untrusted_inputs": [
      "Log line: Everything healthy."
    ]
  }
}
```

**Allowed Response (`200 OK`)**:
```json
{
  "decision": "allow",
  "endpoint": "/api/v1/system/status",
  "kev_scores": {
    "injection_confidence": 0.03,
    "malicious_intent": 0.02
  },
  "backend_response": {
    "status": "healthy",
    "service": "payments-api",
    "uptime_seconds": 86400
  }
}
```

**Denied Response (`403 Forbidden`)**:
```json
{
  "decision": "deny",
  "rule": "endpoint_denied",
  "reason": "Endpoint '/api/v1/db/drop' is denied by zero-trust policy",
  "kev_scores": {
    "injection_confidence": 0.92,
    "malicious_intent": 0.97
  }
}
```

---

## 3. Native Rust ML Engine (`candle-core` + `candle-nn` + Kev `.safetensors`)

### 3.1 Direct Model Download & Loading in Rust (`src/ml/kev.rs`)
At startup, the proxy initializes `KevEngine`:
1. **Device Selection**: Attempts `candle_core::Device::new_metal(0)` on Apple Silicon for native GPU unified-memory acceleration; gracefully falls back to `Device::Cpu` on non-Metal hosts.
2. **Direct Download via `hf-hub`**:
   - Uses `hf_hub::api::sync::Api` inside `src/ml/kev.rs` to download and cache the open-source Kev System One checkpoint from Hugging Face (`jaredpalmer/kev-0.6b` / `jaredpalmer/kev-0.8b`):
     - Base backbone `.safetensors` + `config.json` (`Qwen/Qwen3-0.6B-Base` / `Qwen/Qwen3.5-0.8B-Base`)
     - Kev PEFT LoRA adapter (`adapter_model.safetensors`, `adapter_config.json` with `r=16, lora_alpha=32`)
     - Kev calibrated `PointerHead` weights (`head.safetensors` / `head.pt` projections `q`, `k` of dim `256` and `temperature`)
     - `tokenizer.json` loaded via Hugging Face `tokenizers::Tokenizer`.
3. **LoRA Weight Merging in `candle-nn`**:
   - For each target projection (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`), folds the `.safetensors` LoRA adapter into the base tensor at startup:
     $$W_{\text{merged}} = W_{\text{base}} + \frac{\alpha}{r} (B \cdot A)$$
   - Builds the `candle-nn` transformer layers and `PointerHead` (`q: Linear(d, 256)`, `k: Linear(d, 256)`) in unified memory.

### 3.2 Single-Pass System One Prefill & Pointer Readout
Kev computes exact yes/no (`noul`) probabilities in a single forward pass without autoregressive token generation:
- **Special Delimiters**:
  - `<|fim_prefix|>` (`state` start)
  - `<|fim_middle|>` (`question` start)
  - `<|box_start|>` (`option` start)
  - `<|box_end|>` (`option` end — readout index `opt_idx`)
  - `<|fim_suffix|>` (`decide` token — readout index `decide_idx`)
- **Packed Sequence Construction**:
  ```text
  <|fim_prefix|> {state_tokens}
  <|fim_middle|> {instruction_tokens}
  <|box_start|> no <|box_end|>
  <|box_start|> yes <|box_end|>
  <|fim_suffix|>
  ```
- **Pointer Head Readout**:
  $$z_i = \frac{\langle K(h_{\text{opt}_i}), Q(h_{\text{decide}}) \rangle}{\sqrt{d_p} \cdot T}, \quad P(\text{yes}) = \text{softmax}(z_0, z_1)_1$$
- **Two Parallel Interrogations**:
  1. `injection_confidence`: Scored across overlapping windows of the prompt / `untrusted_inputs` (`max` probability across windows retained).
  2. `malicious_intent`: Scored on `(user_prompt, endpoint, method, payload)` to detect intent hijacking or unauthorized tool invocation.

---

## 4. Zero-Trust Policy Configuration (`policy.toml`)

```toml
[defaults]
max_injection_confidence = 0.70
max_malicious_intent = 0.50
model_timeout_ms = 500
fail_closed = true
require_jwt = true

[endpoints."/api/v1/system/status"]
backend_url = "http://127.0.0.1:5000/api/v1/system/status"
allowed_methods = ["GET", "POST"]
allowed_roles = ["agent", "admin", "readonly"]
required_args = ["service"]
constraints.service = { regex = "^[a-z0-9-]+$", max_len = 64 }

[endpoints."/api/v1/email/send"]
backend_url = "http://127.0.0.1:5000/api/v1/email/send"
allowed_methods = ["POST"]
allowed_roles = ["agent", "admin"]
required_args = ["to", "body"]
constraints.to = { regex = "^[^@]+@ourcompany\\.com$", max_len = 128 }
constraints.body = { max_len = 2000 }

[endpoints."/api/v1/db/drop"]
backend_url = "http://127.0.0.1:5000/api/v1/db/drop"
allowed_methods = ["POST"]
# Explicitly disabled / empty allowed_roles so any agent attempt to call /api/v1/db/drop is blocked
enabled = false
allowed_roles = []
required_args = ["table"]
```

---

## 5. Cryptographic Authorization

1. **Inbound Agent Authentication (Route B)**:
   - Agent requests to `POST /api/execute` carry `Authorization: Bearer <JWT>` signed with `JWT_SECRET` (HS256).
   - Claims verified: `sub` (caller ID), `role` (`agent`, `readonly`, `admin`), `exp` (expiration timestamp), and optional `payload_sha256` binding.
2. **Outbound Sidecar-to-Backend Signing (Rust Proxy → Python Flask `:5000`)**:
   - Every approved request forwarded to Flask carries:
     - `X-Sidecar-Timestamp`: Unix epoch seconds (`ts`)
     - `X-Sidecar-Signature`: `hex(HMAC-SHA256(SIDECAR_SECRET, "{ts}.{raw_body}"))`
   - Flask (`backend/app.py`) verifies `(now - ts) <= 30s` and checks `hmac.compare_digest` before executing any route handler.


Reasons and trade-offs: see `ARCHITECTURE.md`.

## 6. Repository Layout

```text
Zero-Trust-Agentic-Sidecar/
├── Cargo.toml                  # Single Rust package (Axum, Tokio, Reqwest, Candle, hf-hub, JWT/HMAC)
├── policy.toml                 # Deterministic zero-trust endpoint & threshold policy
├── .env.example                # Environment variables template
├── ARCHITECTURE.md             # Architectural invariants & engineering decisions
├── design-doc.md               # Technical design, contracts, and tickets
├── implementation_plan.md      # Step-by-step implementation guide
├── src/
│   ├── main.rs                 # Entrypoint: loads config, policy, Candle/Kev model, starts Axum on :8080
│   ├── lib.rs                  # Library root exporting modules for integration testing
│   ├── config.rs               # Environment & runtime configuration
│   ├── types.rs                # Route A (/v1/messages) & Route B (/api/execute) Serde structs
│   ├── crypto.rs               # JWT verification (Agent->Proxy) & HMAC-SHA256 signing (Proxy->Flask)
│   ├── policy.rs               # TOML policy loader, regex validator, and deterministic rule evaluator
│   ├── audit.rs                # Structured JSONL stdout audit logger (redacts secrets/arg values)
│   ├── routes/
│   │   ├── mod.rs              # Axum Router builder (/healthz, /v1/messages, /api/execute)
│   │   ├── messages.rs         # Route A: Prompt interception, Kev ML eval, x-api-key injection, LLM streaming
│   │   └── execute.rs          # Route B: Tool-call interception, parallel Kev + Policy + JWT, signed Flask forward
│   └── ml/
│       ├── mod.rs              # DecisionModel trait & sliding-window text chunker
│       └── kev.rs              # Native Candle + hf-hub Kev (.safetensors + LoRA merge + PointerHead) engine
├── backend/
│   ├── app.py                  # Python Flask target backend (:5000) with HMAC verification middleware
│   ├── requirements.txt        # Flask, pytest, requests
│   └── test_backend.py         # Pytest suite verifying /api/v1/system/status, /api/v1/db/drop, and 401 HMAC rejection
└── tests/
    └── e2e.rs                  # Rust E2E tests covering Route A, Route B, JWT/HMAC, Policy, and Candle Kev engine
```

Env: `SIDECAR_SECRET` (required) · `POLICY_PATH` · `MODEL_KIND` (`heuristic`|`onnx`|`http`) · `MODEL_DIR` · `MODEL_URL` · `SIDECAR_ADDR` · `BACKEND_ADDR`

## Tickets

| ID | Ticket | Needs | Test |
| --- | --- | --- | --- |
| A1 | Workspace + 4 crates + CI (fmt, clippy, test) | — | `cargo build --workspace` |
| A2 | Shared types, serde round-trip | A1 | `cargo test -p common types` |
| A3 | `DecisionModel` trait + `FixedModel` fake | A2 | `cargo test -p common model` |
| C1 | `sign` / `verify` | A1 | `cargo test -p common signing` |
| C2 | axum verify extractor (401) | C1 | `cargo test -p common signing` |
| B1 | Server, `/healthz`, env config | A2 | `cargo test -p sidecar server` |
| B2 | Policy load + validate | A2 | `cargo test -p sidecar policy` |
| B3 | Policy check (steps 2–4), pure fn | B2 | `cargo test -p sidecar policy` |
| B4 | Pipeline: parallel + timeout + decision order | A3, B3 | `cargo test -p sidecar pipeline` |
| B5 | Forwarder (sign + POST) | C1, B4 | `cargo test -p sidecar forward` |
| B6 | Audit log | B4 | `cargo test -p sidecar audit` |
| E1 | Mock backend | C2 | `cargo test -p mock-backend` |
| E2 | End-to-end test, random ports | E1, B5 | `cargo test --test e2e` |
| D1 | `HeuristicModel` | A3 | `cargo test -p decision-model heuristic` |
| D2 | Pick model (Kev?): format, license, p50/p99 on laptop | — | add result to `ARCHITECTURE.md` |
| D3 | Input builder + windowing | A2 | `cargo test -p decision-model input` |
| D4 | `OnnxModel` | A3, D2 | `cargo test -p decision-model --features onnx` |
| D5 | Eval fixtures, precision/recall | D1 | `cargo run -p decision-model --example eval` |
| D6 | Latency bench | B4, D4 | `cargo run --release --example bench` |

## Open

- Kev: format (ONNX / safetensors / GGUF), license, latency → D2.
- Keep name "Jev" in README or make it generic?
