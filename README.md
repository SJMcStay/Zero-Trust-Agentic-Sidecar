# Zero-Trust Agentic Sidecar
**A Deterministic Execution Proxy for AI Agents**

## Overview
Autonomous LLM agents are highly susceptible to indirect prompt injection, turning unauthorized tool execution into a critical Sev-1 remote code execution (RCE) vulnerability. Filtering prompts with another LLM is non-deterministic and slow. This project adds a deterministic security layer instead.

The **Zero-Trust Agentic Sidecar** is a Rust proxy between a local LLM agent and both the LLM API and internal backend APIs. It scores prompts and tool calls for prompt injection and malicious intent with [Kev](https://huggingface.co/jaredpalmer/kev-4b), an open, Jev-compatible System One decision model running on this machine. Tool calls are then checked against zero-trust policy rules and cryptographically signed. Malicious payloads are dropped before they reach the backend.

## Key Features
- **Kev interrogation (local)**: calibrated `injection_confidence` and `malicious_intent` probabilities from Kev, run on this machine. Policy and model checks run concurrently (`tokio::join!`); the model is still on the critical path, so latency is Kev's latency.
- **Outbound Tool-Call Interception**: Intercepts structured JSON tool-calls exiting the agent rather than relying solely on brittle prompt filtering.
- **Deterministic Policy Enforcement**: Evaluates every requested action against explicit, zero-trust security rules before execution.
- **Cryptographic Authorization**: Verifies caller identity, permissions, and payload integrity prior to forwarding requests.
- **Fail closed**: missing secrets, a bad policy, a model timeout or a model that cannot start all mean deny (or no start).
- **Claude Code as is**: `scripts/claude.sh` points it at the sidecar. Kev scores only what is new each turn (tool results, `@files`, typed text); a block shows up as Claude Code's reply plus a desktop notification. `policy.local.toml` edits apply without a restart.

## Project Architecture: Enterprise AI Gateway

```text
                       Route A: POST /v1/messages (Prompt Interception + Kev Eval)
+---------------------+ -----------------------------------------------------------> +----------------------------------+
|                     |                                                              |   Zero-Trust Agentic Sidecar     |
|  Client / Agent CLI |                                                              |       (Rust / Axum + Kev)        |
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

Kev runs on this machine: its reference server on `127.0.0.1:8008` (default) or an experimental in-process Candle port.

### 1. System Components & Tech Stack
- **Client/Agent Layer**: Claude Code CLI (or local Qwen via Ollama), pointed at the proxy with `ANTHROPIC_BASE_URL="http://localhost:8080"`.
- **Security Proxy (The Core)**: Rust: Axum (HTTP routing), Tokio (async runtime), Reqwest (HTTP client). Listens on `localhost:8080`.
- **Decision Model**: Kev, a LoRA adapter + pointer head on Qwen3.5. Default `kev-4b` (32 GB machines); `kev-0.8b` for 16 GB machines. Weights are pinned and sha256-verified (`models.json`, `models.lock.json`) and never downloaded at runtime.
- **Model Engine** (`KEV_ENGINE`): `http` (default) asks Kev's reference server on `127.0.0.1:8008` (MLX on Apple Silicon, CUDA on NVIDIA); `candle` runs an in-process Rust port that is experimental and not yet Kev-equivalent (see `ARCHITECTURE.md`).
- **Target Backend API**: Python Flask application. Listens on `localhost:5000` (`5001` in `.env.example`: macOS AirPlay holds 5000). Simulates internal enterprise tools with safe endpoints (`/api/v1/system/status`) and vulnerable endpoints (`/api/v1/db/drop`).

### 2. Data Flow & Routing Logic

#### Route A: Inbound Message Gateway (Client -> LLM)
- **Endpoint**: `POST /v1/messages` (Proxy port `8080`)
- **Interception**: Axum catches the chat request from the agent CLI.
- **ML Evaluation**: Kev scores what is new since the model's last turn (`score = "new"`) or the whole prompt (`score = "all"`) for prompt injection.
- **Forwarding**: If safe (probability < threshold), the request goes unchanged to `https://api.anthropic.com/v1/messages` with the sidecar's `x-api-key` (or the agent's own login) and the response streams back. If not, the agent gets a 403 or an assistant message explaining the block (`on_block`).

#### Route B: Outbound Tool Execution (LLM -> Backend)
- **Endpoint**: `POST /api/execute` (Proxy port `8080`)
- **Interception**: Axum catches the JSON tool-call payload generated by the LLM.
- **Policy Enforcement**: JWT caller check, endpoint allowlist, roles, required arguments and `regex`/`max_len` constraints (`policy.toml`), plus Kev's two scores.
- **Execution**: If authorized, signs the request (HMAC-SHA256) and forwards it to the Flask backend (`localhost:5000`). Otherwise returns `403 Forbidden`.

### 3. Deployment & Security Posture
- Runs entirely locally: the proxy, Kev and the backend. Only Route A's forward to the LLM API leaves the machine.
- API secrets are injected inside the Rust process; the agent never sees them. With no key set, the agent's own login passes through.
- Kev-4B needs ~12 GB of memory while serving; use `KEV_MODEL=kev-0.8b` on 16 GB machines.

## Getting Started

```bash
scripts/init.sh                       # once: tools, .env with fresh secrets, verified weights, Kev server, build
scripts/kev_server.sh                 # terminal 1: Kev on 127.0.0.1:8008
set -a && . ./.env && set +a && cargo run --release   # terminal 2: sidecar on :8080
scripts/claude.sh                     # terminal 3: Claude Code through the sidecar
```

| Doc | Read for |
| --- | --- |
| [`USAGE.md`](USAGE.md) | Setup, running, Claude Code, tests, cleanup |
| [`FILES.md`](FILES.md) | What each file does |
| [`design-doc.md`](design-doc.md) | Request flows, contracts, Kev engine, tickets |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | Invariants and decisions, with reasons |
| [`proposals/`](proposals/) | Design notes: scoring only new content (done); alerts and other agents |

## License
MIT
