# Zero-Trust Agentic Sidecar — Design

Rust proxy between an LLM agent and backend APIs. Each tool-call is scored by a local decision model. Calls under the risk thresholds are signed and forwarded. The backend rejects unsigned calls.

## Flow

```text
agent ──POST /v1/tool-call──► sidecar :8080
                                ▼
                             decision model (local)
                                ▼
                             thresholds ── deny ──► 403 {rule, reason}
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

Rules: `malformed` `model_unavailable` `model_risk_high` `backend_error`

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

**Forwarding** `POST {BACKEND_URL}/tools/{tool}` with `args` as body.

## Decision order (first failure wins)

1. parse → `malformed`
2. model answered within `MODEL_TIMEOUT_MS` → `model_unavailable`
3. `injection_confidence ≤ MAX_INJECTION` and `malicious_intent ≤ MAX_INTENT` → `model_risk_high`
4. sign + forward → `backend_error`

## Decision model: Kev

[`jaredpalmer/kev-0.8b`](https://huggingface.co/jaredpalmer/kev-0.8b) v1.0, Apache-2.0. LoRA adapter + pointer head on `Qwen/Qwen3.5-0.8B-Base`. Input: a state (text or JSON) + typed questions (`noul` yes/no, `choice`, `score`). Output: a calibrated probability per option, one forward pass, no generation.

- **Now:** Kev's reference server (Python; MLX on Apple Silicon) on `127.0.0.1:8008`, offline after setup (`HF_HUB_OFFLINE=1`); Rust calls it via `decision_model::KevHttp` (defaults to loopback; built without TLS). Run steps: [`USAGE.md`](USAGE.md).
- **Target:** in-process Rust via `candle` + safetensors (tickets K1–K5). candle has no Qwen3.5 Gated DeltaNet, so we port it.
- **Quality caveat:** card reports prompt-injection accuracy 0.547 (Jev 0.893) and tool-call routing below chance. Measure on our own fixtures (D5) before trusting thresholds.
- Inference is per tool-call (tens–hundreds of ms), not per packet. Blocking inference runs in `spawn_blocking`; semaphore caps concurrency.

`models/kev/`: `adapter_model.safetensors` · `head.pt` (pointer head, T = 2.35) · `tokenizer.json` · `manifest.json`. `models/qwen3.5-0.8b-base/`: `config.json` · weights.

## Decisions

Reasons and trade-offs: see `ARCHITECTURE.md`.

## Layout

Per-file detail: [`FILES.md`](FILES.md).

```text
crates/common          types, DecisionModel trait, signing
crates/sidecar         server, config, pipeline, forward, audit
crates/decision-model  kev_http (done), onnx_classifier (done), kev candle port
crates/mock-backend    test backend, verifies signatures
models.json            which models to fetch; models.lock.json pins revision + sha256
models/                git-ignored weights
scripts/               fetch_model.py, kev_server.sh, cleanup.sh, export_onnx.sh
```

Env: `SIDECAR_SECRET` (required) · `BACKEND_URL` · `MAX_INJECTION` (0.7) · `MAX_INTENT` (0.5) · `MODEL_TIMEOUT_MS` (300) · `KEV_URL` (`http://127.0.0.1:8008`) · `SIDECAR_ADDR`

## Tickets

| ID | Ticket | Needs | Test |
| --- | --- | --- | --- |
| A1 | Workspace + 4 crates + CI (fmt, clippy, test) | — | `cargo build --workspace` |
| A2 | Shared types, serde round-trip | A1 | `cargo test -p common types` |
| A3 | `DecisionModel` trait + `FixedModel` fake | A2 | `cargo test -p common model` |
| C1 | `sign` / `verify` | A1 | `cargo test -p common signing` |
| C2 | axum verify extractor (401) | C1 | `cargo test -p common signing` |
| B1 | Server, `/healthz`, env config | A2 | `cargo test -p sidecar server` |
| B4 | Pipeline: model + timeout + thresholds | A3 | `cargo test -p sidecar pipeline` |
| B5 | Forwarder (sign + POST) | C1, B4 | `cargo test -p sidecar forward` |
| B6 | Audit log | B4 | `cargo test -p sidecar audit` |
| E1 | Mock backend | C2 | `cargo test -p mock-backend` |
| E2 | End-to-end test, random ports | E1, B5 | `cargo test --test e2e` |
| D1 | `HeuristicModel` | A3 | `cargo test -p decision-model heuristic` |
| D2 | ~~Pick model~~ done: Kev-0.8B, safetensors + candle | — | — |
| D3 | Input builder: tool-call → Kev state + questions | A2 | `cargo test -p decision-model input` |
| D4 | ~~`KevHttp` client~~ done | — | `cargo test -p decision-model` |
| D5 | Eval fixtures, precision/recall | D4 | `cargo run -p decision-model --example eval` |
| D6 | Latency bench | B4, D4 | `cargo run --release --example bench` |
| K1 | Kev encoder: special tokens, user-text sanitising, rows per question | — | parity with Kev's `encode()` on fixtures |
| K2 | Qwen3.5 text backbone in candle (Gated DeltaNet + gated attention + MLP) | — | hidden-state parity vs reference |
| K3 | Load base safetensors + merge LoRA (W += 2·B·A) | K2 | weight parity |
| K4 | Pointer head + temperature from `head.pt` | K2 | probability parity ≤ 0.02 |
| K5 | `Kev` in-process, Metal on Mac | K1–K4 | `kev_ask` same answers as `KevHttp` |
| P1 | Proxy pass-through `/v1/messages` (`ANTHROPIC_BASE_URL`) | — | Claude Code works through it |
| P2 | Inbound: score new `tool_result` blocks in request | P1, D4 | fixture test |
| P3 | Outbound: buffer + score `tool_use` in SSE stream, rewrite on deny | P1, D4 | recorded-stream test |
| P4 | Claude Code `PreToolUse` hook endpoint | D4 | `curl` test |

## Open

- Kev-0.8B injection accuracy is low (0.547): fine-tune (`kev.train --init_from`), Kev-4B, or a dedicated classifier (`OnnxClassifier`)?
- README's Route A/B architecture vs this doc's `/v1/tool-call`: align on one.
- Policy check (tool allowlist, arg rules) dropped for now: any tool passes if model scores are low.
