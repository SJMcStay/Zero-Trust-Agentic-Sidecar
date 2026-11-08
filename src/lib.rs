//! Zero-Trust Agentic Sidecar Library Root (`src/lib.rs`)
//!
//! A high-performance execution security proxy for autonomous AI agents.
//! Intercepts LLM prompt flows (Route A: `POST /v1/messages`) and tool
//! execution calls (Route B: `POST /api/execute`) with Kev decision-model scoring
//! (local, see `ml`), two-legged cryptographic authorization, and deterministic
//! policy enforcement.

pub mod audit;
pub mod config;
pub mod crypto;
pub mod debug;
pub mod ml;
pub mod policy;
pub mod routes;
pub mod types;

// Re-exports of primary types and functions
pub use audit::*;
pub use config::{Config, ConfigError, KevEngineKind};
pub use crypto::*;
pub use ml::*;
pub use policy::*;
pub use routes::*;
pub use types::*;
