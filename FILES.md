# Files

| Path | What it does |
| --- | --- |
| `README.md` | Pitch, architecture, quick start |
| `USAGE.md` | Setup, running, tests, cleanup |
| `design-doc.md` | Request flows, contracts, Kev engine, tickets |
| `ARCHITECTURE.md` | Invariants and decisions, with reasons |
| `FILES.md` | This map |
| `Cargo.toml` / `Cargo.lock` | Single Rust crate and pinned dependency versions |
| `policy.toml` | Zero-trust policy: endpoints, roles, argument rules, Kev thresholds, model timeout |
| `.env.example` | Environment template (`scripts/init.sh` copies it to `.env` with fresh secrets) |
| `models.json` | Kev models and their bases: Hub repo, pinned commit, files, local dir |
| `models.lock.json` | sha256 of every model file; downloads are checked against it |
| **src/** | |
| `src/main.rs` | Entry point: config, audit, policy, decision engine (`KEV_ENGINE`, fail closed), Axum on `:8080` |
| `src/lib.rs` | Library root (modules re-exported for the integration tests) |
| `src/config.rs` | Environment config with fail-closed secret checks |
| `src/types.rs` | Route A/B request/response types, `KevVerdict`, policy schema |
| `src/routes/messages.rs` | Route A: score prompt, inject `x-api-key`, forward and stream from the LLM API |
| `src/routes/execute.rs` | Route B: JWT + policy + Kev (concurrently), sign and forward to the backend |
| `src/routes/mod.rs` | Router (`/healthz`, `/v1/messages`, `/api/execute`) and shared state |
| `src/policy.rs` | `policy.toml` loader and deterministic rule evaluation |
| `src/crypto.rs` | JWT verification (agent → sidecar), HMAC-SHA256 signing (sidecar → backend) |
| `src/audit.rs` | JSON-lines audit log with secret redaction |
| `src/ml/mod.rs` | `DecisionEngine` trait, windowing, verdict pooling |
| `src/ml/kev_http.rs` | `KevHttpEngine`: asks the local Kev server (default engine) |
| `src/ml/kev.rs` | Candle engine (experimental), local artifact resolution, synthetic test engine |
| **scripts/** | |
| `scripts/init.sh` | One-time, re-runnable setup |
| `scripts/fetch_model.py` | Downloads a `models.json` model into `models/`, pinned and sha256-verified |
| `scripts/kev_server.sh` | Kev reference server on `127.0.0.1:8008`, offline, reading `models/` |
| `scripts/cleanup.sh` | Deletes generated files; lists sizes and asks first |
| **backend/** | |
| `backend/app.py` | Flask target backend on `:5000`; rejects unsigned requests |
| `backend/requirements.txt` | Backend and test dependencies (installed into `.venv/` by `init.sh`) |
| `backend/test_*.py` | Backend and adversarial pytest suites |
| **tests/** | |
| `tests/e2e.rs` | End to end through Route A/B with the real Flask backend |
| `tests/test_routes.rs`, `tests/adversarial_*.rs` | Route, policy, crypto and adversarial suites |

Created locally, git-ignored: `models/` (weights), `.kev-server/` (Kev checkout, Python env, link-only HF cache), `.venv/`, `target/`, `.env`, `policy.local.toml`.
