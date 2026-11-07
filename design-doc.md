# Zero-Trust Agentic Sidecar — Design

Rust proxy between an LLM agent and backend APIs. Each tool-call goes through a policy check and a local decision model at the same time. Only allowed calls are signed and forwarded. The backend rejects unsigned calls.

## Flow

```text
agent ──POST /v1/tool-call──► sidecar :8080
                                ├─ policy check  ─┐  (parallel)
                                └─ decision model ┘
                                ▼
                             decide ── deny ──► 403 {rule, reason}
                                │ allow
                                ▼
                             HMAC sign ──► backend :9000 (verifies sig, else 401)
```

## Contracts (`crates/common`)

**Request** `POST /v1/tool-call`
```json
{ "session_id": "s1", "tool": "send_email",
  "args": { "to": "a@ourcompany.com", "body": "..." },
  "context": { "user_prompt": "...", "untrusted_inputs": ["..."] } }
```

**Response** `200 {"decision":"allow","result":...}` · `403/400/502 {"decision":"deny","rule":"...","reason":"..."}`

Rules: `malformed` `tool_not_allowed` `missing_arg` `constraint_failed` `model_unavailable` `model_risk_high` `backend_error`

**Model trait**
```rust
#[async_trait]
pub trait DecisionModel: Send + Sync {
    fn id(&self) -> &str;
    async fn score(&self, req: &ToolCallRequest) -> anyhow::Result<RiskVerdict>;
}
pub struct RiskVerdict { pub injection_confidence: f32, pub malicious_intent: f32, pub reason: String }
```

**Signing** headers `X-Sidecar-Timestamp`, `X-Sidecar-Signature = hex(HMAC-SHA256(SIDECAR_SECRET, ts + "." + body))`. Constant-time verify, 30 s window.

**Policy** `policy.toml`
```toml
[defaults]
max_injection_confidence = 0.7
max_malicious_intent = 0.5
model_timeout_ms = 300
fail_closed = true

[tools.send_email]
backend = "http://127.0.0.1:9000/tools/send_email"
required_args = ["to", "body"]
constraints.to = { regex = "^[^@]+@ourcompany\\.com$" }
constraints.body = { max_len = 2000 }
# unlisted tool = denied
```

## Decision order (first failure wins)

1. parse → `malformed`
2. tool listed → `tool_not_allowed`
3. required args → `missing_arg`
4. constraints → `constraint_failed`
5. model answered in time → `model_unavailable`
6. scores ≤ thresholds → `model_risk_high`
7. sign + forward → `backend_error`

## Decision model

- Runs in-process on CPU, loaded once, shared via `Arc`.
- Inference in `spawn_blocking` (blocking work stalls tokio otherwise); semaphore caps concurrency (timeout stops waiting, not inference).
- Impls: `HeuristicModel` (default, CI), `OnnxModel` (feature `onnx`, via `ort` + `tokenizers`), `HttpModel` (feature `http`, local server fallback).
- Inputs: `untrusted_inputs` → `injection_confidence`; `user_prompt + tool + args` → `malicious_intent`. One batched call.
- Long input scored in overlapping windows, max score kept (injection hidden at end otherwise missed).

## Decisions

Reasons and trade-offs: see `ARCHITECTURE.md`.

## Layout

```text
crates/common          types, DecisionModel trait, signing
crates/sidecar         server, config, policy, pipeline, forward, audit
crates/decision-model  heuristic, onnx, http, input
crates/mock-backend    test backend, verifies signatures
models/                git-ignored
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
