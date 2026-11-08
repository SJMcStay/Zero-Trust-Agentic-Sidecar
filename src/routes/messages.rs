//! Route A: Inbound Messages Interception Proxy (`src/routes/messages.rs`)
//!
//! Handles `POST /v1/messages` (Claude Code sends `/v1/messages?beta=true`):
//! 1. Parses the Anthropic Messages payload.
//! 2. Picks what to score from `[route_a] score`: `all` (system + every message, every turn) or
//!    `new` (only content since the model's last turn: new tool results and user text, with the
//!    user's latest request as context; see `src/turn.rs`).
//! 3. With `remember_blocked`, keeps blocking a conversation that still carries blocked content.
//! 4. Scores it with the `DecisionEngine`, bounded by `model_timeout_ms` (fail closed).
//! 5. `enforce` mode blocks at the thresholds, answering with an Anthropic error (403) or an
//!    assistant message (`on_block`); `monitor` mode only logs and alerts.
//! 6. Forwards allowed requests unchanged (body, query, `anthropic-*` headers) and relays the
//!    upstream response, streaming or not (`src/routes/upstream.rs`).
//! 7. Emits structured JSONL audit logs via `AuditLogger`.

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, Method, StatusCode, Uri},
    response::{IntoResponse, Response},
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{Duration, Instant};

use crate::{
    alerts,
    routes::{upstream, AppState},
    turn::{extract_new_turn, history_hashes, NewTurn},
    types::{AuditEntry, BlockResponse, DenyRule, EnforcementMode, KevVerdict, ScoreScope},
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
    /// Removes known secrets, API keys and bearer tokens from the message.
    pub fn sanitize(&mut self, known_secrets: &[String]) {
        self.error.message = crate::audit::redact_string(&self.error.message, known_secrets);
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

/// What one request is scored on.
enum Scored {
    /// `score = "all"`: system prompt + every message, as one text.
    All(String),
    /// `score = "new"`: only the new turn.
    New(NewTurn),
}

/// Why a request is blocked: drives the audit line, the alert and the agent-facing text.
struct Block {
    rule: DenyRule,
    reason: String,
    /// Text for an `on_block = "message"` answer.
    explanation: String,
    verdict: Option<KevVerdict>,
}

/// Route A Handler: `POST /v1/messages`
pub async fn messages_handler(
    State(state): State<AppState>,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let start_time = Instant::now();
    let policy = state.current_policy();
    let route_a = policy.route_a().clone();

    // 1. Parse JSON body
    let json_val: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => return malformed(&state, start_time, format!("Malformed JSON: {e}")),
    };
    let stream = json_val
        .get("stream")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model_name = json_val
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    tracing::debug!(
        bytes = body.len(),
        model = %model_name,
        messages = json_val.get("messages").and_then(|v| v.as_array()).map_or(0, |m| m.len()),
        stream,
        query = uri.query().unwrap_or(""),
        anthropic_beta = headers.contains_key("anthropic-beta"),
        score = ?route_a.score,
        mode = ?route_a.mode,
        "A1 request received"
    );
    tracing::debug!(shape = %request_shape(&json_val), "A1 message roles and blocks (no content)");

    // 2. What to score
    let scored = match route_a.score {
        ScoreScope::All => match extract_prompt_text_from_json(&json_val) {
            Ok(text) => {
                tracing::debug!(
                    chars = text.chars().count(),
                    "A2 scoring the system prompt + every message (score = \"all\")"
                );
                tracing::trace!(preview = %crate::debug::preview(&text, crate::debug::preview_chars()), "A2 text preview");
                Scored::All(text)
            }
            Err(err_msg) => return malformed(&state, start_time, err_msg),
        },
        ScoreScope::New => {
            if model_name.trim().is_empty() {
                return malformed(
                    &state,
                    start_time,
                    "Missing or invalid required field 'model'".to_string(),
                );
            }
            let turn = match extract_new_turn(&json_val, policy.turn_rules()) {
                Ok(turn) => turn,
                Err(err_msg) => return malformed(&state, start_time, err_msg),
            };
            tracing::debug!(
                tool_results = turn.tool_results.len(),
                tools = ?turn.tool_results.iter().map(|t| t.tool.as_str()).collect::<Vec<_>>(),
                user_text_chars = turn.user_text.chars().count(),
                scored_chars = turn.scored_chars(),
                skipped_blocks = turn.skipped_blocks,
                has_context = turn.user_request.is_some(),
                "A2 scoring only the new turn (score = \"new\")"
            );
            tracing::trace!(preview = %crate::debug::preview(&turn.to_text(), crate::debug::preview_chars()), "A2 new-turn preview");

            if route_a.remember_blocked && route_a.mode == EnforcementMode::Enforce {
                let earlier = state
                    .blocked
                    .lock()
                    .map(|memory| memory.contains_any(&history_hashes(&json_val)))
                    .unwrap_or(false);
                if earlier {
                    tracing::debug!(
                        "A3 conversation still contains content blocked earlier -> block"
                    );
                    let block = Block {
                        rule: DenyRule::ModelRiskHigh,
                        reason: "Conversation still contains content blocked earlier".to_string(),
                        explanation: "[Zero-Trust Sidecar] This conversation still contains content \
                            that was blocked earlier, so nothing was sent to the model. Use /rewind \
                            to drop it or /clear to start over."
                            .to_string(),
                        verdict: None,
                    };
                    return blocked(
                        &state,
                        &route_a,
                        &policy,
                        start_time,
                        stream,
                        &model_name,
                        block,
                        None,
                        &Caller::from(&headers),
                        false,
                    );
                }
            }

            if turn.is_empty() {
                tracing::debug!("A3 nothing new to score -> allow");
                audit(
                    &state,
                    start_time,
                    "allow",
                    None,
                    Some("Nothing new to score".to_string()),
                    None,
                );
                return forward_to_llm(&state, &uri, &headers, body, stream).await;
            }
            Scored::New(turn)
        }
    };

    // 3. Score with the decision engine, bounded by the policy timeout (fail closed)
    let timeout_dur = std::time::Duration::from_millis(policy.defaults().model_timeout_ms);
    let model_start = Instant::now();
    let eval_res = tokio::time::timeout(timeout_dur, async {
        match &scored {
            Scored::All(text) => state.decision_engine.evaluate_prompt(text).await,
            Scored::New(turn) => state.decision_engine.evaluate_turn(turn).await,
        }
    })
    .await;

    let verdict = match eval_res {
        Err(_) => {
            tracing::debug!(
                timeout_ms = timeout_dur.as_millis() as u64,
                engine = state.decision_engine.model_id(),
                "A3 model timed out"
            );
            Err(format!(
                "ML engine timed out after {} ms",
                timeout_dur.as_millis()
            ))
        }
        Ok(Err(ml_err)) => {
            tracing::debug!(error = %ml_err, "A3 model error");
            Err(format!("ML inference error: {ml_err}"))
        }
        Ok(Ok(v)) => {
            tracing::debug!(
                injection = v.injection_confidence,
                intent = v.malicious_intent,
                engine = %v.model_id,
                model_ms = model_start.elapsed().as_millis() as u64,
                "A3 model verdict"
            );
            Ok(v)
        }
    };

    // Route A asks only "is this injected?"; whether an action is harmful is Route B's question.
    let max_injection = policy.defaults().max_injection_confidence;
    let new_turn = match &scored {
        Scored::New(turn) => Some(turn),
        Scored::All(_) => None,
    };

    // 4. Decide
    let block = match verdict {
        Err(reason) => Some(Block {
            rule: DenyRule::ModelUnavailable,
            reason: reason.clone(),
            explanation: format!(
                "[Zero-Trust Sidecar] This request was not sent to the model because it could not \
                 be checked ({reason}). The sidecar fails closed: make sure the Kev server is \
                 running, then retry."
            ),
            verdict: None,
        }),
        Ok(v) if v.injection_confidence >= max_injection => {
            let source = new_turn
                .map(describe_sources)
                .unwrap_or_else(|| "the prompt".to_string());
            Some(Block {
                rule: DenyRule::ModelRiskHigh,
                reason: PROMPT_INJECTION_MESSAGE.to_string(),
                explanation: format!(
                    "[Zero-Trust Sidecar] Blocked: {source} looks like a prompt injection \
                     (Kev injection {:.2}, threshold {max_injection}). It was not sent to the \
                     model. Use /rewind to drop it or /clear to start over.",
                    v.injection_confidence
                ),
                verdict: Some(v),
            })
        }
        Ok(v) => {
            tracing::debug!(max_injection, "A4 allow: injection score below threshold");
            audit(&state, start_time, "allow", None, None, Some(&v));
            None
        }
    };

    if let Some(block) = block {
        let caller = Caller::from(&headers);
        if route_a.mode == EnforcementMode::Monitor {
            tracing::debug!(rule = ?block.rule, reason = %block.reason, "A4 monitor mode: would block, forwarding");
            let mut alert = caller.alert(&block, new_turn, false);
            alert.action = "monitor".into();
            state.alert_hub.publish(alert);
            if policy.alerts().desktop && !state.alert_hub.has_listeners() {
                alerts::desktop(
                    "Zero-Trust Sidecar (monitor)",
                    &format!("Would block: {}", block.reason),
                );
            }
            audit(
                &state,
                start_time,
                "allow",
                None,
                Some(format!("monitor mode, would block: {}", block.reason)),
                block.verdict.as_ref(),
            );
            return forward_to_llm(&state, &uri, &headers, body, stream).await;
        }

        // Just over the threshold: ask the human once (scripts/alert_notifier.py), if listening.
        let defaults = policy.defaults();
        let borderline = matches!(block.rule, DenyRule::ModelRiskHigh)
            && defaults.human_approval_timeout_ms > 0
            && block.verdict.as_ref().is_some_and(|v| {
                alerts::near_threshold(
                    v.injection_confidence,
                    max_injection,
                    defaults.human_approval_band,
                )
            });
        if borderline {
            let alert = caller.alert(&block, new_turn, true);
            let wait = Duration::from_millis(defaults.human_approval_timeout_ms);
            tracing::debug!(
                wait_ms = wait.as_millis() as u64,
                "A4 borderline score: asking the human"
            );
            if state.alert_hub.request_approval(alert, wait).await {
                tracing::debug!("A4 human allowed this request once -> forwarding");
                audit(
                    &state,
                    start_time,
                    "allow",
                    None,
                    Some(alerts::HUMAN_ALLOWED_ONCE.to_string()),
                    block.verdict.as_ref(),
                );
                return forward_to_llm(&state, &uri, &headers, body, stream).await;
            }
        }
        return blocked(
            &state,
            &route_a,
            &policy,
            start_time,
            stream,
            &model_name,
            block,
            new_turn,
            &caller,
            borderline,
        );
    }

    // 5. Forward unchanged
    forward_to_llm(&state, &uri, &headers, body, stream).await
}

/// `user[text:47,text:3400] assistant[tool_use] user[tool_result:63]`: roles, block types and
/// text sizes, for seeing what a harness sends without logging the content.
fn request_shape(body: &Value) -> String {
    let Some(messages) = body.get("messages").and_then(Value::as_array) else {
        return String::new();
    };
    let block = |b: &Value| {
        let kind = b.get("type").and_then(Value::as_str).unwrap_or("untyped");
        let chars = b
            .get("text")
            .and_then(Value::as_str)
            .map(|t| t.chars().count())
            .or_else(|| match b.get("content") {
                Some(Value::String(t)) => Some(t.chars().count()),
                _ => None,
            });
        match chars {
            Some(n) => format!("{kind}:{n}"),
            None => kind.to_string(),
        }
    };
    messages
        .iter()
        .map(|m| {
            let role = m.get("role").and_then(Value::as_str).unwrap_or("?");
            let blocks = match m.get("content") {
                Some(Value::String(t)) => format!("string:{}", t.chars().count()),
                Some(Value::Array(items)) => items.iter().map(block).collect::<Vec<_>>().join(","),
                _ => String::new(),
            };
            format!("{role}[{blocks}]")
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The calling agent, for alerts.
struct Caller {
    agent: String,
    session: Option<String>,
}

impl Caller {
    fn from(headers: &HeaderMap) -> Self {
        let get = |name: &str| {
            headers
                .get(name)
                .and_then(|v| v.to_str().ok())
                .map(str::to_owned)
        };
        Self {
            agent: get("user-agent").unwrap_or_else(|| "unknown".to_string()),
            session: get("x-claude-code-session-id"),
        }
    }

    /// Alert for `block`; the target names what carried the injection (`Read {"file_path":...}`).
    fn alert(&self, block: &Block, turn: Option<&NewTurn>, approval: bool) -> alerts::Alert {
        let target = match turn {
            Some(t) if !t.tool_results.is_empty() => t
                .tool_results
                .iter()
                .map(|r| {
                    if r.input.is_empty() {
                        r.tool.clone()
                    } else {
                        format!("{} {}", r.tool, crate::turn::truncate(&r.input, 120))
                    }
                })
                .collect::<Vec<_>>()
                .join(", "),
            Some(_) => "the new user text".to_string(),
            None => "the prompt".to_string(),
        };
        let rule = serde_json::to_value(block.rule)
            .ok()
            .and_then(|v| v.as_str().map(str::to_owned))
            .unwrap_or_default();
        alerts::new_alert(
            &self.agent,
            self.session.clone(),
            "POST /v1/messages",
            &target,
            &rule,
            &block.reason,
            block.verdict.as_ref().map(|v| v.injection_confidence),
            block.verdict.as_ref().map(|v| v.malicious_intent),
            approval,
        )
    }
}

/// "the Read result (/repo/notes.md)" / "the new user text" for block explanations.
fn describe_sources(turn: &NewTurn) -> String {
    let tools: Vec<&str> = turn.tool_results.iter().map(|t| t.tool.as_str()).collect();
    match tools.as_slice() {
        [] => "the new user text".to_string(),
        [one] => format!("the {one} result"),
        many => format!("the {} results", many.join(", ")),
    }
}

async fn forward_to_llm(
    state: &AppState,
    uri: &Uri,
    headers: &HeaderMap,
    body: Bytes,
    stream: bool,
) -> Response {
    let url = upstream::with_query(&state.config.upstream_llm_url, uri);
    tracing::debug!(
        url = %url,
        api_key = if state.config.anthropic_api_key.is_some() { "sidecar key" } else { "client credentials passed through" },
        "A5 forwarding to the LLM API"
    );
    upstream::forward(state, Method::POST, &url, headers, body, stream).await
}

fn malformed(state: &AppState, start_time: Instant, message: String) -> Response {
    tracing::debug!(reason = %message, "A1 malformed request -> 400");
    audit(
        state,
        start_time,
        "deny",
        Some(DenyRule::Malformed),
        Some(message.clone()),
        None,
    );
    let mut err = AnthropicErrorEnvelope::invalid_request_error(message);
    err.sanitize(state.audit_logger.known_secrets());
    (
        StatusCode::BAD_REQUEST,
        [(header::CONTENT_TYPE, "application/json")],
        serde_json::to_string(&err).unwrap_or_default(),
    )
        .into_response()
}

/// Logs, alerts, remembers and answers a block (`on_block`: 403 error or assistant message).
#[allow(clippy::too_many_arguments)]
fn blocked(
    state: &AppState,
    route_a: &crate::types::RouteAPolicy,
    policy: &crate::policy::PolicyEngine,
    start_time: Instant,
    stream: bool,
    model: &str,
    block: Block,
    turn: Option<&NewTurn>,
    caller: &Caller,
    alerted: bool,
) -> Response {
    tracing::debug!(rule = ?block.rule, reason = %block.reason, on_block = ?route_a.on_block, "A4 deny");
    let explanation =
        crate::audit::redact_string(&block.explanation, state.audit_logger.known_secrets());
    audit(
        state,
        start_time,
        "deny",
        Some(block.rule),
        Some(block.reason.clone()),
        block.verdict.as_ref(),
    );
    if let (Some(turn), true) = (turn, route_a.remember_blocked) {
        if let Ok(mut memory) = state.blocked.lock() {
            memory.remember(&turn.piece_hashes);
        }
    }
    if !alerted {
        state.alert_hub.publish(caller.alert(&block, turn, false));
    }
    if policy.alerts().desktop && !state.alert_hub.has_listeners() {
        alerts::desktop("Zero-Trust Sidecar blocked a request", &explanation);
    }
    match route_a.on_block {
        BlockResponse::Error => {
            let err = AnthropicErrorEnvelope::permission_error(PROMPT_INJECTION_MESSAGE);
            (
                StatusCode::FORBIDDEN,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::to_string(&err).unwrap_or_default(),
            )
                .into_response()
        }
        BlockResponse::Message => assistant_message(model, stream, &explanation),
    }
}

/// A complete assistant reply carrying `text`, as JSON or as an SSE stream, so the agent shows
/// it like any answer and ends its turn.
pub fn assistant_message(model: &str, stream: bool, text: &str) -> Response {
    let id = format!(
        "msg_zt_{:x}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let usage = json!({"input_tokens": 0, "output_tokens": 0});
    if !stream {
        let body = json!({
            "id": id, "type": "message", "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn", "stop_sequence": null, "usage": usage
        });
        return (
            StatusCode::OK,
            [(header::CONTENT_TYPE, "application/json")],
            body.to_string(),
        )
            .into_response();
    }
    let events = [
        (
            "message_start",
            json!({"type": "message_start", "message": {
            "id": id, "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": null, "stop_sequence": null, "usage": usage}}),
        ),
        (
            "content_block_start",
            json!({"type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}),
        ),
        (
            "content_block_delta",
            json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": text}}),
        ),
        (
            "content_block_stop",
            json!({"type": "content_block_stop", "index": 0}),
        ),
        (
            "message_delta",
            json!({"type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 0}}),
        ),
        ("message_stop", json!({"type": "message_stop"})),
    ];
    let sse: String = events
        .iter()
        .map(|(event, data)| format!("event: {event}\ndata: {data}\n\n"))
        .collect();
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "text/event-stream"),
            (header::CACHE_CONTROL, "no-cache"),
        ],
        sse,
    )
        .into_response()
}

fn audit(
    state: &AppState,
    start_time: Instant,
    decision: &str,
    rule: Option<DenyRule>,
    reason: Option<String>,
    verdict: Option<&KevVerdict>,
) {
    let _ = state.audit_logger.log(AuditEntry {
        timestamp: chrono::Utc::now(),
        route: "POST /v1/messages".to_string(),
        session_id: None,
        caller_sub: None,
        caller_role: None,
        target_endpoint: Some("/v1/messages".to_string()),
        method: Some("POST".to_string()),
        decision: decision.to_string(),
        matched_rule: rule,
        reason,
        injection_confidence: verdict.map(|v| v.injection_confidence),
        malicious_intent: verdict.map(|v| v.malicious_intent),
        latency_ms: start_time.elapsed().as_millis() as u64,
    });
}
