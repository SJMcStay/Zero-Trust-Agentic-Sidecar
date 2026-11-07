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

## Architecture

```text
+------------------+       Outbound JSON       +-----------------------------+       Verified Request       +--------------------+
|                  |        Tool-Calls         |                             |                              |                    |
|  Local LLM Agent | ------------------------> | Zero-Trust Agentic Sidecar  | ---------------------------> | Internal Backend   |
|                  |                           |        (Rust Proxy)         |                              |       APIs         |
+------------------+                           +-----------------------------+                              +--------------------+
                                                              |
                                                              | Drops Unauthorized /
                                                              v Malicious Payloads
                                                           [BLOCKED]
```

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
