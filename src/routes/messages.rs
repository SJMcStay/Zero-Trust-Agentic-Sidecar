//! Route A: Inbound Messages Interception Proxy (`src/routes/messages.rs`)
//!
//! Handles `POST /v1/messages`:
//! 1. Parses polymorphic Anthropic Messages API payloads (system text, text blocks, tool results).
//! 2. Evaluates prompt safety using `DecisionEngine::evaluate_prompt` bounded by `model_timeout_ms`.
//! 3. Drops prompt injections (`injection_confidence >= 0.70` or `malicious_intent >= 0.50`) with HTTP 403
//!    and exact Anthropic permission_error JSON schema.
//! 4. Injects `x-api-key` and `anthropic-version: 2023-06-01` in compiled Rust process memory and forwards
//!    authorized calls to upstream LLM.
//! 5. Supports both real-time SSE streaming (`stream: true`) and non-streaming HTTP 200 responses.
//! 6. Emits structured JSONL audit logs via `AuditLogger`.

use axum::{
    body::{Body, Bytes},
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::time::Instant;

use crate::{
    routes::AppState,
    types::{AuditEntry, DenyRule},
};

pub const ANTHROPIC_VERSION_HEADER_VALUE: &str = "2023-06-01";
pub const PROMPT_INJECTION_MESSAGE: &str =
    "Zero-trust security policy violation: prompt injection detected";

/// Standard Anthropic error response envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicErrorEnvelope {
    #[serde(rename = "type")]
    pub response_type: String, // "error"
    pub error: AnthropicErrorDetail,
}

/// Detailed error payload embedded inside `AnthropicErrorEnvelope`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String, // "permission_error", "invalid_request_error", "api_error"
    pub message: String,
}

impl AnthropicErrorEnvelope {
    pub fn permission_error(message: impl Into<String>) -> Self {
        Self {
            response_type: "error".to_string(),
            error: AnthropicErrorDetail {
                error_type: "permission_error".to_string(),
                message: message.into(),
            },
        }
    }

    pub fn invalid_request_error(message: impl Into<String>) -> Self {
        Self {
            response_type: "error".to_string(),
            error: AnthropicErrorDetail {
                error_type: "invalid_request_error".to_string(),
                message: message.into(),
            },
        }
    }

    pub fn api_error(message: impl Into<String>) -> Self {
        Self {
            response_type: "error".to_string(),
            error: AnthropicErrorDetail {
                error_type: "api_error".to_string(),
                message: message.into(),
            },
        }
    }
}

/// Extracts full prompt text across system instructions and conversation messages.
/// Handles polymorphic `system` (string or array of text blocks) and message `content`
/// (string, text blocks, tool_result blocks, and nested content).
pub fn extract_prompt_text_from_json(val: &Value) -> Result<String, String> {
    let mut prompt_parts = Vec::new();

    // 1. Validate required 'model'
    let model = val.get("model").and_then(|v| v.as_str()).unwrap_or("");
    if model.trim().is_empty() {
        return Err("Missing or invalid required field 'model'".to_string());
    }

    // 2. Extract system instructions if present
    if let Some(sys) = val.get("system") {
        if let Some(s) = sys.as_str() {
            if !s.is_empty() {
                prompt_parts.push(s.to_string());
            }
        } else if let Some(arr) = sys.as_array() {
            for block in arr {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        prompt_parts.push(text.to_string());
                    }
                }
            }
        }
    }

    // 3. Validate and extract messages
    let messages = val
        .get("messages")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "Missing or invalid required field 'messages'".to_string())?;

    if messages.is_empty() {
        return Err("Field 'messages' must contain at least one message".to_string());
    }

    for msg in messages {
        let content = msg
            .get("content")
            .ok_or_else(|| "Message missing 'content' field".to_string())?;

        if let Some(text) = content.as_str() {
            if !text.is_empty() {
                prompt_parts.push(text.to_string());
            }
        } else if let Some(blocks) = content.as_array() {
            for block in blocks {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !text.is_empty() {
                        prompt_parts.push(text.to_string());
                    }
                } else if let Some(tool_res) = block.get("content") {
                    if let Some(res_str) = tool_res.as_str() {
                        prompt_parts.push(res_str.to_string());
                    } else if let Some(nested_blocks) = tool_res.as_array() {
                        for nb in nested_blocks {
                            if let Some(t) = nb.get("text").and_then(|t| t.as_str()) {
                                prompt_parts.push(t.to_string());
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(prompt_parts.join("\n"))
}

/// Route A Handler: `POST /v1/messages`
pub async fn messages_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let start_time = Instant::now();

    // 1. Parse JSON body
    let json_val: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log(AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /v1/messages".to_string(),
                session_id: None,
                caller_sub: None,
                caller_role: None,
                target_endpoint: Some("/v1/messages".to_string()),
                method: Some("POST".to_string()),
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::Malformed),
                reason: Some(format!("Invalid JSON payload: {e}")),
                injection_confidence: None,
                malicious_intent: None,
                latency_ms,
            });

            let err = AnthropicErrorEnvelope::invalid_request_error(format!("Malformed JSON: {e}"));
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err).unwrap_or_default(),
            )
                .into_response();
        }
    };

    tracing::debug!(
        bytes = body.len(),
        model = json_val
            .get("model")
            .and_then(|v| v.as_str())
            .unwrap_or("?"),
        messages = json_val
            .get("messages")
            .and_then(|v| v.as_array())
            .map_or(0, |m| m.len()),
        stream = json_val
            .get("stream")
            .and_then(|v| v.as_bool())
            .unwrap_or(false),
        anthropic_beta = headers.contains_key("anthropic-beta"),
        "A1 request received"
    );

    // 2. Extract full prompt text across system and messages
    let prompt_text = match extract_prompt_text_from_json(&json_val) {
        Ok(txt) => txt,
        Err(err_msg) => {
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log(AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /v1/messages".to_string(),
                session_id: None,
                caller_sub: None,
                caller_role: None,
                target_endpoint: Some("/v1/messages".to_string()),
                method: Some("POST".to_string()),
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::Malformed),
                reason: Some(err_msg.clone()),
                injection_confidence: None,
                malicious_intent: None,
                latency_ms,
            });

            let err = AnthropicErrorEnvelope::invalid_request_error(err_msg);
            return (
                StatusCode::BAD_REQUEST,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err).unwrap_or_default(),
            )
                .into_response();
        }
    };

    let stream = json_val
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    tracing::debug!(
        chars = prompt_text.chars().count(),
        "A2 text to score: system prompt + every message (see proposals/score-new-content.md)"
    );
    tracing::trace!(
        preview = %crate::debug::preview(&prompt_text, crate::debug::PREVIEW_CHARS),
        "A2 text preview"
    );

    // 3. Risk Evaluation via DecisionEngine with timeout
    let model_start = Instant::now();
    let timeout_dur =
        std::time::Duration::from_millis(state.policy_engine.defaults().model_timeout_ms);
    let eval_res = tokio::time::timeout(
        timeout_dur,
        state.decision_engine.evaluate_prompt(&prompt_text),
    )
    .await;

    let verdict = match eval_res {
        Err(_) => {
            tracing::debug!(
                timeout_ms = timeout_dur.as_millis() as u64,
                engine = state.decision_engine.model_id(),
                "A3 model timed out -> deny (fail closed)"
            );
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log(AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /v1/messages".to_string(),
                session_id: None,
                caller_sub: None,
                caller_role: None,
                target_endpoint: Some("/v1/messages".to_string()),
                method: Some("POST".to_string()),
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::ModelUnavailable),
                reason: Some("ML engine timed out".to_string()),
                injection_confidence: None,
                malicious_intent: None,
                latency_ms,
            });

            let err = AnthropicErrorEnvelope::permission_error(PROMPT_INJECTION_MESSAGE);
            return (
                StatusCode::FORBIDDEN,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err).unwrap_or_default(),
            )
                .into_response();
        }
        Ok(Err(ml_err)) => {
            tracing::debug!(error = %ml_err, "A3 model error -> deny (fail closed)");
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let _ = state.audit_logger.log(AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /v1/messages".to_string(),
                session_id: None,
                caller_sub: None,
                caller_role: None,
                target_endpoint: Some("/v1/messages".to_string()),
                method: Some("POST".to_string()),
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::ModelUnavailable),
                reason: Some(format!("ML inference error: {ml_err}")),
                injection_confidence: None,
                malicious_intent: None,
                latency_ms,
            });

            let err = AnthropicErrorEnvelope::permission_error(PROMPT_INJECTION_MESSAGE);
            return (
                StatusCode::FORBIDDEN,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err).unwrap_or_default(),
            )
                .into_response();
        }
        Ok(Ok(v)) => {
            tracing::debug!(
                injection = v.injection_confidence,
                intent = v.malicious_intent,
                engine = %v.model_id,
                model_ms = model_start.elapsed().as_millis() as u64,
                "A3 model verdict"
            );
            v
        }
    };

    // 4. Threshold Checking (injection_confidence >= 0.70 or malicious_intent >= 0.50)
    let max_injection = state.policy_engine.defaults().max_injection_confidence;
    let max_intent = state.policy_engine.defaults().max_malicious_intent;

    if verdict.injection_confidence >= max_injection || verdict.malicious_intent >= max_intent {
        tracing::debug!(
            injection = verdict.injection_confidence,
            max_injection,
            intent = verdict.malicious_intent,
            max_intent,
            "A4 deny: score at or above threshold -> 403 to the agent"
        );
        let latency_ms = start_time.elapsed().as_millis() as u64;
        let _ = state.audit_logger.log(AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /v1/messages".to_string(),
            session_id: None,
            caller_sub: None,
            caller_role: None,
            target_endpoint: Some("/v1/messages".to_string()),
            method: Some("POST".to_string()),
            decision: "deny".to_string(),
            matched_rule: Some(DenyRule::ModelRiskHigh),
            reason: Some(PROMPT_INJECTION_MESSAGE.to_string()),
            injection_confidence: Some(verdict.injection_confidence),
            malicious_intent: Some(verdict.malicious_intent),
            latency_ms,
        });

        let err = AnthropicErrorEnvelope::permission_error(PROMPT_INJECTION_MESSAGE);
        return (
            StatusCode::FORBIDDEN,
            [(header::CONTENT_TYPE, "application/json")],
            serde_json::to_string(&err).unwrap_or_default(),
        )
            .into_response();
    }

    // 5. Audit log Allowed Verdict
    tracing::debug!(
        max_injection,
        max_intent,
        "A4 allow: scores below thresholds"
    );
    let latency_ms = start_time.elapsed().as_millis() as u64;
    let _ = state.audit_logger.log(AuditEntry {
        timestamp: chrono::Utc::now(),
        route: "POST /v1/messages".to_string(),
        session_id: None,
        caller_sub: None,
        caller_role: None,
        target_endpoint: Some("/v1/messages".to_string()),
        method: Some("POST".to_string()),
        decision: "allow".to_string(),
        matched_rule: None,
        reason: None,
        injection_confidence: Some(verdict.injection_confidence),
        malicious_intent: Some(verdict.malicious_intent),
        latency_ms,
    });

    // 6. Upstream Forwarding to Anthropic
    let target_url = &state.config.upstream_llm_url;
    let mut req = state
        .http_client
        .post(target_url)
        .header(header::CONTENT_TYPE, "application/json")
        .header("anthropic-version", ANTHROPIC_VERSION_HEADER_VALUE);

    if let Some(key) = &state.config.anthropic_api_key {
        req = req.header("x-api-key", key);
    }
    if let Some(beta) = headers.get("anthropic-beta") {
        req = req.header("anthropic-beta", beta);
    }
    tracing::debug!(
        url = %target_url,
        api_key = if state.config.anthropic_api_key.is_some() { "sidecar key" } else { "none (client credentials are not forwarded)" },
        "A5 forwarding to the LLM API"
    );

    let upstream_resp = match req.body(body).send().await {
        Ok(resp) => resp,
        Err(err) => {
            tracing::debug!(error = %err, "A5 LLM API unreachable -> 502");
            let err_body = AnthropicErrorEnvelope::api_error(format!(
                "Upstream Anthropic connection failure: {err}"
            ));
            return (
                StatusCode::BAD_GATEWAY,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err_body).unwrap_or_default(),
            )
                .into_response();
        }
    };

    let status = upstream_resp.status();
    tracing::debug!(
        status = status.as_u16(),
        stream,
        total_ms = start_time.elapsed().as_millis() as u64,
        "A6 LLM API answered; relaying to the agent"
    );

    // If upstream returns error status code, pass it through directly
    if !status.is_success() {
        let resp_bytes = upstream_resp.bytes().await.unwrap_or_default();
        let axum_status = StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        return (
            axum_status,
            [(header::CONTENT_TYPE, "application/json")],
            resp_bytes,
        )
            .into_response();
    }

    // Success response: stream SSE or return buffered JSON
    if stream {
        let stream_body = Body::from_stream(upstream_resp.bytes_stream());
        Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "text/event-stream")
            .header(header::CACHE_CONTROL, "no-cache")
            .header(header::CONNECTION, "keep-alive")
            .body(stream_body)
            .unwrap_or_else(|_| {
                (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "Failed to build stream response",
                )
                    .into_response()
            })
    } else {
        let resp_bytes = match upstream_resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                let err_body = AnthropicErrorEnvelope::api_error(format!(
                    "Failed reading upstream response: {e}"
                ));
                return (
                    StatusCode::BAD_GATEWAY,
                    [(header::CONTENT_TYPE, "application/json")],
                    serde_json::to_string(&err_body).unwrap_or_default(),
                )
                    .into_response();
            }
        };

        (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            resp_bytes,
        )
            .into_response()
    }
}
