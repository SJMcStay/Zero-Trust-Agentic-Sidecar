# Files

| Path | What it does |
| --- | --- |
| `README.md` | Pitch, architecture, quick start |
| `USAGE.md` | Setup, running, the demo script, policy, tests, cleanup |
| `design-doc.md` | Request flows, contracts, Kev engine, tickets |
| `ARCHITECTURE.md` | Invariants and decisions, with reasons |
| `FILES.md` | This map |
| `Cargo.toml` / `Cargo.lock` | Single Rust crate and pinned dependency versions |
| `policy.toml` | Zero-trust policy (strict schema): thresholds and timeout, Route A scoring and blocking, alerts, Route B endpoints and local tools, agent CLI mappings (defaults; your copy is `policy.local.toml`) |
| `.env.example` | Environment template (`scripts/init.sh` copies it to `.env` with fresh secrets) |
| `models.json` | Kev models and their bases: Hub repo, pinned commit, files, local dir |
| `models.lock.json` | sha256 of every model file; downloads are checked against it |
| **src/** | |
| `src/main.rs` | Entry point: config, audit, policy, decision engine (`KEV_ENGINE`, fail closed), Axum on `:8080`; `env <cli>` prints an agent's mapping |
| `src/lib.rs` | Library root (modules re-exported for the integration tests) |
| `src/config.rs` | Environment config: fail-closed secret checks (16+ bytes), URL validation |
| `src/types.rs` | Route A/B request/response types, `KevVerdict`, policy schema |
| `src/routes/messages.rs` | Route A: pick what to score, ask Kev, block (403 or assistant message) or forward |
| `src/routes/upstream.rs` | Unchanged forwarding to the LLM API (headers, query, streaming); relayed Claude Code endpoints |
| `src/turn.rs` | What is new in a request (tool results, inlined files, typed text, prefill; harness `system` messages skipped); memory of blocked content |
| `src/alerts.rs` | Desktop notification on block (macOS `osascript`, Linux `notify-send`) |
| `src/routes/execute.rs` | Route B: JWT + policy + Kev (concurrently) for endpoints and local tools; sign, then forward (endpoint) or return a signed allow (tool) |
| `src/routes/mod.rs` | Router (`/healthz`, `/v1/messages`, `count_tokens`, `/v1/models`, `/api/hello`, `/api/execute`) and shared state |
| `src/policy.rs` | `policy.toml` loader and validation, rule evaluation for endpoints and tools, threshold overrides, CLI mappings (`agent_env`), reload on change (`LivePolicy`) |
| `src/crypto.rs` | JWT verification (agent → sidecar), HMAC-SHA256 signing (sidecar → backend) |
| `src/audit.rs` | JSON-lines audit log; recursive secret redaction (keys, bearer tokens, JWTs, mapped CLI keys) for logs and error bodies |
| `src/debug.rs` | `ZT_DEBUG` switch for per-step debug output (1 steps, 2 previews, 3 full text) |
| `src/ml/mod.rs` | `DecisionEngine` trait, windowing, verdict pooling |
| `src/ml/kev_http.rs` | `KevHttpEngine`: asks the local Kev server (default engine) |
| `src/ml/kev.rs` | Candle engine (experimental), local artifact resolution, synthetic test engine |
| **proposals/** | |
| `proposals/score-new-content.md` | R1 (done): score only what is new in each turn |
| `proposals/alerts-and-agents.md` | Pop-up alerts and in-protocol answers (done), agent-agnostic adapters, path to a real CLI |
| **scripts/** | |
| `scripts/init.sh` | One-time, re-runnable setup |
| `scripts/fetch_model.py` | Downloads a `models.json` model into `models/`, pinned and sha256-verified |
| `scripts/kev_server.sh` | Kev reference server on `127.0.0.1:8008` (MLX or CUDA), offline, reading `models/` |
| `scripts/cleanup.sh` | Deletes generated files; lists sizes and asks first |
| `scripts/mint_jwt.py` | Mints an agent JWT for calling Route B by hand |
| `scripts/agent.sh` | Runs an agent CLI through the sidecar using `[agent_cli_mappings]`; refuses unserved protocols |
| `scripts/claude.sh` | `agent.sh claude` |
| **demo/** | |
| `demo/notes.md`, `demo/poisoned-notes.md` | Clean and poisoned (inert `.example` payload) files for the demo in `USAGE.md` |
| **backend/** | |
| `backend/app.py` | Flask target backend on `:5000`; rejects unsigned requests |
| `backend/requirements.txt` | Backend and test dependencies (installed into `.venv/` by `init.sh`) |
| `backend/test_*.py` | Backend and adversarial pytest suites |
| **tests/** | |
| `tests/e2e.rs` | End to end through Route A/B with the real Flask backend |
| `tests/claude_code_gateway.rs` | Claude Code through Route A against a recording upstream: new-turn scoring, real Claude Code request shapes, block replies, monitor mode, relayed endpoints, live policy, long-output and prefill regressions |
| `tests/phase2_acceptance.rs` | Policy overrides, timeouts, CLI mappings, redaction, scoped scoring, local tool inspection |
| `tests/test_routes.rs`, `tests/adversarial_*.rs` | Route, policy, crypto and adversarial suites (M1–M9) |

Created locally, git-ignored: `models/` (weights), `.kev-server/` (Kev checkout, Python env, link-only HF cache), `.venv/`, `target/`, `.env`, `policy.local.toml`.
