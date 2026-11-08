# Files

| Path | What it does |
| --- | --- |
| `README.md` | Project pitch, architecture, quick start |
| `USAGE.md` | How to run Kev and call it from Rust |
| `design-doc.md` | Contracts, decision flow, tickets |
| `ARCHITECTURE.md` | Invariants and key decisions, with reasons |
| `FILES.md` | This map |
| `Cargo.toml` / `Cargo.lock` | Rust workspace (`crates/*`) and pinned dependency versions |
| `models.json` | Which models to download: repo, revision, files, dependencies |
| `models.lock.json` | Pinned commit + sha256 of every model file; everyone gets identical bytes |
| `scripts/fetch_model.py` | Downloads models into `models/`, verifies against the lock (`--lock` to re-pin, `--from-dir` for local copies) |
| `scripts/kev_server.sh` | Runs Kev's reference server on `127.0.0.1:8008`, offline after setup |
| `scripts/cleanup.sh` | Deletes generated files (`target/`, `.kev-server/`; `--models`, `--hf-cache`, `--all`), lists sizes and asks first |
| `scripts/export_onnx.sh` | Converts a safetensors classifier to ONNX for `OnnxClassifier` (not needed for Kev) |
| `crates/decision-model/src/kev_http.rs` | `KevHttp`: async client for Kev's `POST /v1/systemone` |
| `crates/decision-model/src/onnx_classifier.rs` | `OnnxClassifier`: in-process ONNX text classifier (fallback if Kev is too weak) |
| `crates/decision-model/examples/kev_ask.rs` | CLI: score text for prompt injection with Kev |
| `crates/decision-model/examples/classifier_smoke.rs` | CLI: score text with an ONNX classifier |
| `crates/decision-model/tests/` | Fake-server test for `KevHttp`; on-disk test for `OnnxClassifier` (skips without a model) |
| `.gitignore` | Ignores `target/`, `models/`, `.kev-server/`, secrets, personal AI-tool config |

Git-ignored, created locally: `models/` (weights), `.kev-server/` (Kev checkout + Python env), `target/` (Rust builds).
