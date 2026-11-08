//! Decision engines (`src/ml/mod.rs`)
//!
//! Scores raw prompt text (Route A: `POST /v1/messages`) and structured tool execution
//! requests (Route B: `POST /api/execute`) for prompt injection and malicious intent.
//!
//! Engines (selected with `KEV_ENGINE`, see `main.rs`):
//! - [`KevHttpEngine`] (`http`, default): asks Kev's reference server on loopback; real Kev answers.
//! - [`KevCandleEngine`] (`candle`): in-process Candle port; experimental, not yet Kev-equivalent.
//! - [`SyntheticKevEngine`] (`synthetic`): deterministic toy graph for tests and demos only.
//!
//! Also: the [`DecisionEngine`] trait, sliding-window chunking with max-confidence pooling.

pub mod kev;
pub mod kev_http;

pub use kev::*;
pub use kev_http::*;

use crate::types::{KevVerdict, ToolExecuteRequest};
use async_trait::async_trait;
use thiserror::Error;

// ==============================================================================
// 1. Error Types
// ==============================================================================

/// Structured errors emitted during ML model loading, artifact downloading,
/// device initialization, or tensor inference.
#[derive(Debug, Error)]
pub enum MlError {
    #[error("ML inference timed out: {0}")]
    Timeout(String),

    #[error("ML concurrency limit exhausted: {0}")]
    ConcurrencyExhausted(String),

    #[error("Hardware device error: {0}")]
    DeviceError(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Model artifact loading failed: {0}")]
    ModelLoadError(String),

    #[error("Candle tensor operation failed: {0}")]
    TensorError(#[from] candle_core::Error),

    #[error("Inference execution failed: {0}")]
    InferenceFailed(String),

    #[error("Configuration or parsing error: {0}")]
    ConfigError(String),

    #[error("Internal ML engine error: {0}")]
    Internal(String),
}

// ==============================================================================
// 2. Decision Engine Trait
// ==============================================================================

/// Abstract interface for in-process ML safety decision models.
/// Must be Send + Sync + 'static for shared application state.
#[async_trait]
pub trait DecisionEngine: Send + Sync {
    /// Identifier of the model or synthetic engine implementation.
    fn model_id(&self) -> &str;

    /// Evaluates raw prompt text (inbound Route A messages or untrusted contexts).
    async fn evaluate_prompt(&self, prompt: &str) -> Result<KevVerdict, MlError>;

    /// Evaluates structured tool execution requests (outbound Route B calls).
    async fn evaluate_tool_call(&self, req: &ToolExecuteRequest) -> Result<KevVerdict, MlError>;

    /// Scores the new content of one agent turn (Route A with `score = "new"`). Engines that
    /// take a single string get the turn rendered as text; `KevHttpEngine` overrides this to
    /// send Kev a structured state.
    async fn evaluate_turn(&self, turn: &crate::turn::NewTurn) -> Result<KevVerdict, MlError> {
        self.evaluate_prompt(&turn.to_text()).await
    }
}

// ==============================================================================
// 3. Sliding-Window Text & Token Chunking
// ==============================================================================

/// Splits a UTF-8 text string into overlapping character windows.
///
/// Ensures strict UTF-8 validity: slices never divide multi-byte characters.
/// If `text` is empty, returns a single empty string slice.
/// If `text.len() <= window_size` or `stride == 0`, returns the entire text.
pub fn window_text(text: &str, window_size: usize, stride: usize) -> Vec<String> {
    if text.is_empty() {
        return vec![String::new()];
    }

    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= window_size || stride == 0 {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let end = (start + window_size).min(chars.len());
        let chunk: String = chars[start..end].iter().collect();
        chunks.push(chunk);
        if end == chars.len() {
            break;
        }
        start += stride;
    }
    chunks
}

/// Slices text into overlapping string slices using byte offsets aligned to UTF-8 char boundaries.
pub fn chunk_text_windows(text: &str, window_size: usize, stride: usize) -> Vec<String> {
    window_text(text, window_size, stride)
}

/// Splits a slice of token IDs into overlapping windows of fixed token count.
///
/// Default configuration: `window_size = 512`, `stride = 128` (providing 384 tokens overlap).
pub fn chunk_tokens(tokens: &[u32], window_size: usize, stride: usize) -> Vec<Vec<u32>> {
    if tokens.is_empty() {
        return vec![vec![]];
    }
    if tokens.len() <= window_size || stride == 0 {
        return vec![tokens.to_vec()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < tokens.len() {
        let end = (start + window_size).min(tokens.len());
        chunks.push(tokens[start..end].to_vec());
        if end == tokens.len() {
            break;
        }
        start += stride;
    }
    chunks
}

/// Aggregates multiple sliding-window verdicts into a single summary verdict.
///
/// Follows maximum confidence aggregation:
/// `injection_confidence = max(window_injection_confidences)`
/// `malicious_intent = max(window_malicious_intents)`
pub fn aggregate_verdicts(verdicts: &[KevVerdict], model_id: &str) -> KevVerdict {
    if verdicts.is_empty() {
        return KevVerdict::safe(model_id);
    }

    let mut max_injection = 0.0f32;
    let mut max_intent = 0.0f32;
    let mut chosen_reason = String::new();

    for v in verdicts {
        if v.injection_confidence > max_injection {
            max_injection = v.injection_confidence;
            if v.injection_confidence >= 0.70 {
                chosen_reason = v.reason.clone();
            }
        }
        if v.malicious_intent > max_intent {
            max_intent = v.malicious_intent;
            if v.malicious_intent >= 0.50 && chosen_reason.is_empty() {
                chosen_reason = v.reason.clone();
            }
        }
    }

    if chosen_reason.is_empty() {
        chosen_reason = format!(
            "Aggregated {} window(s): max_inj={:.2}, max_intent={:.2}",
            verdicts.len(),
            max_injection,
            max_intent
        );
    }

    KevVerdict::new(max_injection, max_intent, model_id, chosen_reason)
}

// ==============================================================================
// 4. Unit Tests
// ==============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_empty_and_short_text() {
        assert_eq!(window_text("", 512, 128), vec![""]);
        assert_eq!(window_text("hello world", 512, 128), vec!["hello world"]);
        assert_eq!(window_text("exact boundary", 14, 7), vec!["exact boundary"]);
    }

    #[test]
    fn test_sliding_window_overlapping_segments() {
        let text = "abcdefghijklmnopqrstuvwxyz";
        let windows = window_text(text, 10, 5);
        assert!(windows.len() >= 4);
        assert_eq!(windows[0], "abcdefghij");
        assert_eq!(windows[1], "fghijklmno");
        assert_eq!(windows[2], "klmnopqrst");
        assert_eq!(windows[3], "pqrstuvwxy");
        assert_eq!(windows[4], "uvwxyz");
    }

    #[test]
    fn test_sliding_window_utf8_multibyte_safety() {
        // Multi-byte UTF-8 string (emojis and accented letters)
        let text = "🚀🌟🦀🔥 Rust Zero-Trust Gateway 🔐🛡️⚡️";
        let windows = window_text(text, 10, 5);
        assert!(!windows.is_empty());
        for w in &windows {
            // Verify each slice is strictly valid UTF-8
            assert!(std::str::from_utf8(w.as_bytes()).is_ok());
        }
    }

    #[test]
    fn test_token_chunking() {
        let tokens: Vec<u32> = (0..100).collect();
        let chunks = chunk_tokens(&tokens, 30, 20);
        assert_eq!(chunks.len(), 5);
        assert_eq!(chunks[0].len(), 30);
        assert_eq!(chunks[0][0], 0);
        assert_eq!(chunks[1][0], 20);
        assert_eq!(chunks[2][0], 40);
        assert_eq!(chunks[3][0], 60);
        assert_eq!(chunks[4].len(), 20);
        assert_eq!(chunks[4][0], 80);
    }

    #[test]
    fn test_aggregate_verdicts_max_confidence() {
        let v1 = KevVerdict::new(0.10, 0.05, "model_test", "window 1 clean");
        let v2 = KevVerdict::new(0.92, 0.08, "model_test", "window 2 injection detected");
        let v3 = KevVerdict::new(0.40, 0.65, "model_test", "window 3 intent detected");

        let aggregated = aggregate_verdicts(&[v1, v2, v3], "model_test");
        assert_eq!(aggregated.injection_confidence, 0.92);
        assert_eq!(aggregated.malicious_intent, 0.65);
        assert_eq!(aggregated.reason, "window 2 injection detected");
    }

    #[test]
    fn test_aggregate_verdicts_empty() {
        let empty = aggregate_verdicts(&[], "model_test");
        assert!(empty.is_safe(0.70, 0.50));
        assert_eq!(empty.injection_confidence, 0.01);
    }
}
