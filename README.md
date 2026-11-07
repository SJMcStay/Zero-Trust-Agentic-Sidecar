# Zero-Trust Agentic Sidecar
**An Outbound Execution Proxy for AI Agents**

## Overview
Autonomous LLM agents are highly susceptible to indirect prompt injection, turning unauthorized tool execution into a critical Sev-1 remote code execution (RCE) vulnerability. While current industry approaches attempt to solve this by filtering non-deterministic LLM prompts (which frequently fail), this project introduces a deterministic, enterprise-grade security layer.

The **Zero-Trust Agentic Sidecar** is a high-performance Rust proxy that sits between a local LLM agent and internal backend APIs. Instead of proxying inbound chat, it intercepts outbound JSON tool-calls. It evaluates the agent's requested actions against strict zero-trust policies, cryptographically verifies authorization, and drops malicious payloads before they ever reach vulnerable endpoints. It brings robust, microservice-style security to the unpredictable domain of AI agentic workflows.

## Key Features
- **Outbound Tool-Call Interception**: Inspects structured JSON tool-calls exiting the agent rather than relying on brittle inbound prompt filtering.
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
