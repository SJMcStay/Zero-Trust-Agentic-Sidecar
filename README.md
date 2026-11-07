# Zero-Trust Agentic Sidecar
**A Deterministic Execution Proxy for AI Agents**

## Overview
Autonomous LLM agents are highly susceptible to indirect prompt injection, turning unauthorized tool execution into a critical Sev-1 remote code execution (RCE) vulnerability. While current industry approaches attempt to solve this by filtering non-deterministic LLM prompts (which frequently fail and add massive latency), this project introduces a deterministic, enterprise-grade security layer. 

The **Zero-Trust Agentic Sidecar** is a high-performance Rust proxy that sits between a local LLM agent and internal backend APIs. It leverages Jev—a System One decision model—to perform zero-latency parallel interrogations of agent prompts for malicious intent and injection confidence. Outbound JSON tool-calls are then intercepted, evaluated against strict zero-trust policies, and cryptographically verified. Malicious payloads are dropped before they ever reach vulnerable endpoints, bringing robust, microservice-style security to the unpredictable domain of AI workflows.

## Key Features
- **Zero-Latency Parallel Interrogation (Jev)**: Leverages Jev—a System One decision model—to interrogate agent prompts in parallel for malicious intent and injection confidence without adding latency.
- **Outbound Tool-Call Interception**: Intercepts structured JSON tool-calls exiting the agent rather than relying solely on brittle prompt filtering.
- **Deterministic Policy Enforcement**: Evaluates every requested action against explicit, zero-trust security rules before execution.
- **Cryptographic Authorization**: Verifies caller identity, permissions, and payload integrity prior to forwarding requests.
- **High-Performance Rust Core**: Built for minimal latency and memory safety as a lightweight sidecar alongside local or containerized agents.

## Project Architecture: Enterprise AI Gateway

```text
                       Route A: POST /v1/messages (Prompt Interception + Jev ML Eval)
+---------------------+ -----------------------------------------------------------> +----------------------------------+
|                     |                                                              |   Zero-Trust Agentic Sidecar     |
|  Client / Agent CLI |                                                              |    (Rust / Axum / Candle ML)     |
| (Claude Code / Qwen)|                                                              |          localhost:8080          |
+---------------------+ -----------------------------------------------------------> +----------------------------------+
                       Route B: POST /api/execute (JSON Tool-Call Interception)            |                      |
                                                                                           | Safe + Inject        | Authorized
                                                                                           | x-api-key            | Zero-Trust Policy
                                                                                           v                      v
                                                                                +---------------------+  +----------------------+
                                                                                |   Anthropic API     |  | Target Backend API   |
                                                                                |  (External LLM)     |  | (Python Flask :5000) |
                                                                                +---------------------+  +----------------------+
```

### 1. System Components & Tech Stack
- **Client/Agent Layer**: Claude Code CLI (or local Qwen 2.5 14B via Ollama). Configured to route all inbound/outbound HTTP traffic to the proxy via environment variables (e.g., `ANTHROPIC_BASE_URL="http://localhost:8080"`).
- **Security Proxy (The Core)**: High-performance Rust application built with Axum (HTTP routing), Tokio (async runtime), and Reqwest (HTTP client). Listens on `localhost:8080`.
- **Local ML Engine**: Hugging Face `candle-core` and `candle-nn` (Rust native). Uses the Metal API backend for native Apple Silicon hardware acceleration without bridging to Python.
- **Decision Model**: Jev/Kev System One decision model (`.safetensors` format). Loaded directly into the Rust proxy's unified memory space at startup (no external ML microservice, ~4ms inference latency).
- **Target Backend API**: Python Flask application. Listens on `localhost:5000`. Simulates internal enterprise tools with safe endpoints (`/api/v1/system/status`) and vulnerable endpoints (`/api/v1/db/drop`).

### 2. Data Flow & Routing Logic

#### Route A: Inbound Message Gateway (Client -> LLM)
- **Endpoint**: `POST /v1/messages` (Proxy port `8080`)
- **Interception**: Axum middleware catches the chat prompt from the agent CLI.
- **ML Evaluation**: The raw prompt text is passed to the native Candle/Jev model to calculate a prompt injection probability float (parallel evaluation).
- **Forwarding**: If safe (probability < threshold), the proxy uses Reqwest to inject the actual `x-api-key` header and forwards the JSON payload to `https://api.anthropic.com/v1/messages`. It streams the response back to the client.

#### Route B: Outbound Tool Execution (LLM -> Backend)
- **Endpoint**: `POST /api/execute` (Proxy port `8080`)
- **Interception**: Axum catches the JSON tool-call payload generated by the LLM.
- **Policy Enforcement**: Zero-trust policy engine validates the intent, parses the Serde JSON struct, checks mock JWTs/signatures, and evaluates permissions against the requested endpoint.
- **Execution**: If authorized, forwards the tool execution request to the Python Flask backend (`localhost:5000`). If malicious/unauthorized, drops the request and returns a `403 Forbidden` to the agent.

### 3. Deployment & Security Posture
- Runs entirely locally on Apple Silicon (optimizing a 36GB unified memory pool).
- Cryptographic verification and API key injection occur exclusively inside the compiled Rust proxy memory space; API secrets are never exposed to the client terminal or the AI Agent.
- Eradicates garbage collection (GC) latency spikes and I/O bottlenecks by utilizing Rust over Go/Python for the proxy layer.

## Getting Started

### Prerequisites
- [Rust](https://www.rust-lang.org/tools/install) (stable toolchain)
- Cargo

### Build & Run
```bash
# Clone the repository
git clone <repository-url>
cd Zero-Trust-Agentic-Sidecar

# Build the sidecar
cargo build --release

# Run the proxy
cargo run --release
```

## License
MIT
