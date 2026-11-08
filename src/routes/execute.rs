//! Route B: Outbound Tool Interception & Execution Proxy (`src/routes/execute.rs`)
//!
//! Handles `POST /api/execute`:
//! 1. Parses raw `ToolExecuteRequest` JSON (HTTP 400 on malformed payloads).
//! 2. Authenticates `Authorization: Bearer <JWT>` header via `crypto::verify_agent_jwt_with_payload`
//!    (HTTP 401 on missing or invalid auth).
//! 3. Evaluates deterministic zero-trust policy rules and ML risk classification in parallel via `tokio::join!`.
//! 4. Enforces the strict 8-step Deny Hierarchy:
//!    - Priority 1: Malformed request (HTTP 400)
//!    - Priority 2: Missing / Invalid Auth (HTTP 401)
//!    - Priority 3: Endpoint denied (HTTP 403: {"error": "EndpointDenied", "rule": "endpoint"})
//!    - Priority 4: Method not allowed (HTTP 403: {"error": "MethodNotAllowed", "rule": "method"})
//!    - Priority 5: Role unauthorized (HTTP 403: {"error": "UnauthorizedRole", "rule": "role"})
//!    - Priority 6: Missing required argument (HTTP 403: {"error": "MissingArg", "rule": "required_args"})
//!    - Priority 7: Constraint failed (HTTP 403: {"error": "ConstraintFailed", "rule": "constraints"})
//!    - Priority 8: ML risk score high (HTTP 403: {"error": "ModelRiskHigh", "rule": "ml_risk"})
//! 5. Cryptographically signs approved requests with HMAC-SHA256 (`X-Sidecar-Timestamp`, `X-Sidecar-Signature`)
//!    and forwards them to downstream backend microservices.
//! 6. Emits sanitized structured JSONL audit logs for every request.

use axum::{
    body::Bytes,
    extract::State,
    http::{header, HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::{
    crypto::{self, HEADER_SIGNATURE, HEADER_TIMESTAMP},
    policy::PolicyEngine,
    routes::AppState,
    types::{
        AgentClaims, AuditEntry, DenyRule, KevScores, RouteBAllowResponse, ToolExecuteRequest,
    },
};

/// Unified error response payload satisfying both prompt specifications
/// and crate `RouteBDenyResponse` schemas.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ExecuteErrorResponse {
    pub error: String,
    pub rule: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    pub decision: String, // "deny"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kev_scores: Option<KevScores>,
}

impl ExecuteErrorResponse {
    pub fn new(
        error: impl Into<String>,
        rule: impl Into<String>,
        message: impl Into<String>,
        reason: Option<String>,
        kev_scores: Option<KevScores>,
    ) -> Self {
        Self {
            error: error.into(),
            rule: rule.into(),
            message: message.into(),
            reason,
            decision: "deny".to_string(),
            kev_scores,
        }
    }

    /// Sanitizes message and reason against known secrets, API keys, and sensitive tokens.
    pub fn sanitize(&mut self, known_secrets: &[String]) {
        self.message = crate::audit::redact_string(&self.message, known_secrets);
        if let Some(r) = &mut self.reason {
            *r = crate::audit::redact_string(r, known_secrets);
        }
    }
}

/// Specialized HTTP 401 Unauthorized response matching exact prompt requirement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UnauthorizedResponse {
    pub error: String,  // "unauthorized"
    pub reason: String, // "Missing or invalid bearer token"
    pub rule: String,   // "unauthorized_token"
    pub message: String,
    pub decision: String, // "deny"
}

/// Route B Handler: `POST /api/execute`
pub async fn execute_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body_bytes: Bytes,
) -> Response {
    let start_time = Instant::now();

    // =========================================================================
    // Priority 1: Malformed Request Check (HTTP 400 Bad Request)
    // =========================================================================
    let req: ToolExecuteRequest = match serde_json::from_slice(&body_bytes) {
        Ok(parsed) => parsed,
        Err(e) => {
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let mut err_resp = ExecuteErrorResponse::new(
                "Malformed",
                "malformed",
                "Malformed request payload",
                Some(format!(
                    "Failed to parse JSON body as ToolExecuteRequest: {e}"
                )),
                None,
            );
            err_resp.sanitize(state.audit_logger.known_secrets());

            let entry = AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /api/execute".to_string(),
                session_id: None,
                caller_sub: None,
                caller_role: None,
                target_endpoint: None,
                method: None,
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::Malformed),
                reason: Some("Malformed request payload".to_string()),
                injection_confidence: None,
                malicious_intent: None,
                latency_ms,
            };
            let _ = state.audit_logger.log(entry);

            return (StatusCode::BAD_REQUEST, Json(err_resp)).into_response();
        }
    };

    tracing::debug!(
        bytes = body_bytes.len(),
        session = %req.session_id,
        endpoint = %req.endpoint,
        method = %req.method,
        payload_keys = ?req.payload.keys().collect::<Vec<_>>(),
        untrusted_inputs = req.context.untrusted_inputs.len(),
        "B1 tool call received"
    );
    let mut payload_for_preview = serde_json::Value::Object(req.payload.clone());
    state.audit_logger.redact_value(&mut payload_for_preview);
    tracing::trace!(
        payload = %crate::debug::preview(
            &serde_json::to_string(&payload_for_preview).unwrap_or_default(),
            crate::debug::preview_chars()
        ),
        "B1 payload preview"
    );

    // =========================================================================
    // Priority 2: Missing / Invalid Auth Check (HTTP 401 Unauthorized)
    // =========================================================================
    let auth_header = headers.get(header::AUTHORIZATION);
    let bearer_token = match auth_header.and_then(|h| h.to_str().ok()) {
        Some(val) if val.starts_with("Bearer ") || val.starts_with("bearer ") => {
            let t = val[7..].trim();
            if t.is_empty() {
                None
            } else {
                Some(t)
            }
        }
        _ => None,
    };

    let claims: AgentClaims = match bearer_token {
        Some(token) => match crypto::verify_agent_jwt_with_payload(
            token,
            state.config.jwt_secret.as_bytes(),
            Some(&body_bytes),
        ) {
            Ok(c) => c,
            Err(e) => {
                tracing::debug!(error = %e, "B2 JWT rejected -> 401");
                return emit_unauthorized(&state, &req, start_time);
            }
        },
        None => {
            tracing::debug!("B2 no Bearer token -> 401");
            return emit_unauthorized(&state, &req, start_time);
        }
    };
    tracing::debug!(sub = %claims.sub, role = %claims.role, "B2 JWT ok");

    // =========================================================================
    // Step 3: Parallel Evaluation via tokio::join!
    // =========================================================================
    // Snapshot of the live policy (reloaded when the file changes), used for the whole request.
    let policy_snapshot = state.current_policy();
    let policy_engine = Arc::clone(&policy_snapshot);
    let decision_engine = Arc::clone(&state.decision_engine);
    let claims_clone = claims.clone();
    let req_policy = req.clone();
    let req_ml = req.clone();

    let is_tool = req.resolved_tool_name(&policy_engine).is_some()
        || policy_engine.get_tool(&req.endpoint).is_some();
    let (effective_max_inj, effective_max_intent) = if is_tool {
        let tool_name = req
            .resolved_tool_name(&policy_engine)
            .unwrap_or(&req.endpoint);
        policy_engine.effective_thresholds_for_tool(tool_name)
    } else {
        policy_engine.effective_thresholds_for_endpoint(&req.endpoint)
    };

    let timeout_ms = if is_tool {
        let tool_name = req
            .resolved_tool_name(&policy_engine)
            .unwrap_or(&req.endpoint);
        policy_engine.effective_timeout_for_tool(tool_name)
    } else {
        policy_engine.effective_timeout_for_endpoint(&req.endpoint)
    };
    let timeout_dur = Duration::from_millis(timeout_ms);

    let checks_start = Instant::now();
    let (policy_result, ml_result) = tokio::join!(
        async move { evaluate_target_detailed(&policy_engine, &claims_clone, &req_policy) },
        async move {
            tokio::time::timeout(timeout_dur, decision_engine.evaluate_tool_call(&req_ml)).await
        }
    );

    match &policy_result {
        Ok(()) => tracing::debug!("B3 policy: endpoint, method, role, args, constraints all pass"),
        Err(deny) => tracing::debug!(rule = deny.rule, reason = %deny.message, "B3 policy: denied"),
    }

    // Extract ML verdict and scores if available
    let (ml_verdict, ml_scores, ml_timed_out_or_failed) = match ml_result {
        Ok(Ok(verdict)) => {
            tracing::debug!(
                injection = verdict.injection_confidence,
                intent = verdict.malicious_intent,
                engine = %verdict.model_id,
                checks_ms = checks_start.elapsed().as_millis() as u64,
                "B4 model verdict (ran alongside the policy)"
            );
            let scores = KevScores::from(&verdict);
            (Some(verdict), Some(scores), false)
        }
        Ok(Err(ml_err)) => {
            tracing::debug!(error = %ml_err, "B4 model error");
            (None, None, true)
        }
        Err(_timeout) => {
            tracing::debug!(timeout_ms, "B4 model timed out");
            (None, None, true)
        }
    };

    // =========================================================================
    // Priority 3 through 7: Deterministic Policy Failures
    // =========================================================================
    if let Err(policy_deny) = policy_result {
        state.alert_hub.publish(crate::alerts::new_alert(
            headers
                .get(header::USER_AGENT)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown"),
            Some(req.session_id.clone()),
            "POST /api/execute",
            &req.endpoint,
            policy_deny.rule,
            &policy_deny.message,
            ml_verdict.as_ref().map(|v| v.injection_confidence),
            ml_verdict.as_ref().map(|v| v.malicious_intent),
            false,
        ));
        tracing::debug!(rule = policy_deny.rule, "B5 deny: policy rule -> 403");
        let latency_ms = start_time.elapsed().as_millis() as u64;
        let mut err_resp = ExecuteErrorResponse::new(
            policy_deny.error,
            policy_deny.rule,
            policy_deny.message.clone(),
            Some(policy_deny.message.clone()),
            ml_scores,
        );
        err_resp.sanitize(state.audit_logger.known_secrets());

        let entry = AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some(req.session_id.clone()),
            caller_sub: Some(claims.sub.clone()),
            caller_role: Some(claims.role.clone()),
            target_endpoint: Some(req.endpoint.clone()),
            method: Some(req.method.clone()),
            decision: "deny".to_string(),
            matched_rule: Some(policy_deny.deny_rule),
            reason: Some(policy_deny.message),
            injection_confidence: ml_verdict.as_ref().map(|v| v.injection_confidence),
            malicious_intent: ml_verdict.as_ref().map(|v| v.malicious_intent),
            latency_ms,
        };
        let _ = state.audit_logger.log(entry);

        return (policy_deny.status, Json(err_resp)).into_response();
    }

    // =========================================================================
    // Priority 8: ML Risk Score High / Model Unavailable
    // =========================================================================
    let defaults = policy_snapshot.defaults();
    let mut approved_by_human = false;

    if ml_timed_out_or_failed && defaults.fail_closed {
        state.alert_hub.publish(crate::alerts::new_alert(
            headers
                .get(header::USER_AGENT)
                .and_then(|v| v.to_str().ok())
                .unwrap_or("unknown"),
            Some(req.session_id.clone()),
            "POST /api/execute",
            &req.endpoint,
            "model_unavailable",
            "Model evaluation timed out or failed",
            None,
            None,
            false,
        ));
        tracing::debug!("B5 deny: model unavailable and fail_closed -> 403");
        let latency_ms = start_time.elapsed().as_millis() as u64;
        let mut err_resp = ExecuteErrorResponse::new(
            "ModelUnavailable",
            "model_unavailable",
            "Model evaluation timed out or failed",
            Some("Model evaluation timed out or failed".to_string()),
            None,
        );
        err_resp.sanitize(state.audit_logger.known_secrets());

        let entry = AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some(req.session_id.clone()),
            caller_sub: Some(claims.sub.clone()),
            caller_role: Some(claims.role.clone()),
            target_endpoint: Some(req.endpoint.clone()),
            method: Some(req.method.clone()),
            decision: "deny".to_string(),
            matched_rule: Some(DenyRule::ModelUnavailable),
            reason: Some("Model evaluation timed out or failed".to_string()),
            injection_confidence: None,
            malicious_intent: None,
            latency_ms,
        };
        let _ = state.audit_logger.log(entry);

        return (StatusCode::FORBIDDEN, Json(err_resp)).into_response();
    }

    if let Some(verdict) = &ml_verdict {
        if verdict.injection_confidence >= effective_max_inj
            || verdict.malicious_intent >= effective_max_intent
        {
            // Just over the threshold: ask the human (scripts/alert_notifier.py) when one is
            // listening. Every score that is over must be within the band.
            let band = defaults.human_approval_band;
            let within = |score: f32, max: f32| {
                score < max || crate::alerts::near_threshold(score, max, band)
            };
            let approval_timeout = defaults.human_approval_timeout_ms;
            let ask = approval_timeout > 0
                && within(verdict.injection_confidence, effective_max_inj)
                && within(verdict.malicious_intent, effective_max_intent);
            let alert = crate::alerts::new_alert(
                headers
                    .get(header::USER_AGENT)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("unknown"),
                Some(req.session_id.clone()),
                "POST /api/execute",
                &req.endpoint,
                "model_risk_high",
                "Kev score is at or above the configured threshold",
                Some(verdict.injection_confidence),
                Some(verdict.malicious_intent),
                ask,
            );
            let approved_once = if ask {
                state
                    .alert_hub
                    .request_approval(alert, Duration::from_millis(approval_timeout))
                    .await
            } else {
                state.alert_hub.publish(alert);
                false
            };
            if !approved_once {
                tracing::debug!(
                    injection = verdict.injection_confidence,
                    max_injection = effective_max_inj,
                    intent = verdict.malicious_intent,
                    max_intent = effective_max_intent,
                    "B5 deny: model score at or above threshold -> 403"
                );
                let latency_ms = start_time.elapsed().as_millis() as u64;
                let mut err_resp = ExecuteErrorResponse::new(
                    "ModelRiskHigh",
                    "ml_risk",
                    "Model risk threshold exceeded",
                    Some("Model risk threshold exceeded".to_string()),
                    ml_scores,
                );
                err_resp.sanitize(state.audit_logger.known_secrets());

                let entry = AuditEntry {
                    timestamp: chrono::Utc::now(),
                    route: "POST /api/execute".to_string(),
                    session_id: Some(req.session_id.clone()),
                    caller_sub: Some(claims.sub.clone()),
                    caller_role: Some(claims.role.clone()),
                    target_endpoint: Some(req.endpoint.clone()),
                    method: Some(req.method.clone()),
                    decision: "deny".to_string(),
                    matched_rule: Some(DenyRule::ModelRiskHigh),
                    reason: Some("Model risk threshold exceeded".to_string()),
                    injection_confidence: Some(verdict.injection_confidence),
                    malicious_intent: Some(verdict.malicious_intent),
                    latency_ms,
                };
                let _ = state.audit_logger.log(entry);

                return (StatusCode::FORBIDDEN, Json(err_resp)).into_response();
            }
            tracing::debug!("B5 human allowed this borderline call once");
            approved_by_human = true;
        }
    }

    // =========================================================================
    // Step 5: Request Signing & Outbound Execution
    // =========================================================================
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    let outbound_payload_bytes =
        serde_json::to_vec(&req.payload).unwrap_or_else(|_| b"{}".to_vec());

    let signature = crypto::sign_payload(
        state.config.sidecar_secret.as_bytes(),
        timestamp,
        &outbound_payload_bytes,
    );

    // If target is an approved local CLI tool (no backend_url configured):
    if is_tool && policy_snapshot.backend_url_for(&req.endpoint).is_none() {
        let latency_ms = start_time.elapsed().as_millis() as u64;
        let entry = AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some(req.session_id.clone()),
            caller_sub: Some(claims.sub.clone()),
            caller_role: Some(claims.role.clone()),
            target_endpoint: Some(req.endpoint.clone()),
            method: Some(req.method.clone()),
            decision: "allow".to_string(),
            matched_rule: None,
            reason: approved_by_human.then(|| crate::alerts::HUMAN_ALLOWED_ONCE.to_string()),
            injection_confidence: ml_verdict.as_ref().map(|v| v.injection_confidence),
            malicious_intent: ml_verdict.as_ref().map(|v| v.malicious_intent),
            latency_ms,
        };
        let _ = state.audit_logger.log(entry);

        let allow_resp = RouteBAllowResponse {
            decision: "allow".to_string(),
            endpoint: req.endpoint.clone(),
            kev_scores: ml_scores.unwrap_or(KevScores {
                injection_confidence: 0.01,
                malicious_intent: 0.01,
            }),
            backend_response: serde_json::json!({
                "status": "approved",
                "tool": req.endpoint,
                "timestamp": timestamp,
                "signature": signature,
            }),
        };

        let mut response_headers = HeaderMap::new();
        response_headers.insert(header::CONTENT_TYPE, "application/json".parse().unwrap());
        response_headers.insert(HEADER_TIMESTAMP, timestamp.to_string().parse().unwrap());
        response_headers.insert(HEADER_SIGNATURE, signature.parse().unwrap());

        return (StatusCode::OK, response_headers, Json(allow_resp)).into_response();
    }

    let target_path = policy_snapshot
        .backend_url_for(&req.endpoint)
        .and_then(|u| {
            if let Some(pos) = u.find("://") {
                let rest = &u[pos + 3..];
                rest.find('/').map(|p| &rest[p..])
            } else {
                Some(u)
            }
        })
        .unwrap_or(req.endpoint.as_str());

    let target_url = if state.config.flask_backend_url != "http://127.0.0.1:5000" {
        format!(
            "{}{}",
            state.config.flask_backend_url.trim_end_matches('/'),
            if target_path.starts_with('/') {
                target_path.to_string()
            } else {
                format!("/{target_path}")
            }
        )
    } else if let Some(url) = policy_snapshot.backend_url_for(&req.endpoint) {
        url.to_string()
    } else {
        format!(
            "{}{}",
            state.config.flask_backend_url.trim_end_matches('/'),
            if target_path.starts_with('/') {
                target_path.to_string()
            } else {
                format!("/{target_path}")
            }
        )
    };

    let method = match req.method.to_ascii_uppercase().as_str() {
        "GET" => reqwest::Method::GET,
        "POST" => reqwest::Method::POST,
        "PUT" => reqwest::Method::PUT,
        "DELETE" => reqwest::Method::DELETE,
        "PATCH" => reqwest::Method::PATCH,
        other => reqwest::Method::from_bytes(other.as_bytes()).unwrap_or(reqwest::Method::POST),
    };

    tracing::debug!(
        url = %target_url,
        method = %method,
        signed_ts = timestamp,
        bytes = outbound_payload_bytes.len(),
        "B5 allow -> B6 HMAC-signed request to the backend"
    );
    let backend_req = state
        .http_client
        .request(method, &target_url)
        .header(HEADER_TIMESTAMP, timestamp.to_string())
        .header(HEADER_SIGNATURE, signature)
        .header(header::CONTENT_TYPE, "application/json")
        .body(outbound_payload_bytes);

    let backend_resp = match backend_req.send().await {
        Ok(resp) => resp,
        Err(e) => {
            tracing::debug!(error = %e, "B6 backend unreachable -> 502");
            let latency_ms = start_time.elapsed().as_millis() as u64;
            let mut err_resp = ExecuteErrorResponse::new(
                "BackendError",
                "backend_error",
                format!("Downstream backend connection failed: {e}"),
                Some(format!("Downstream backend connection failed: {e}")),
                ml_scores,
            );
            err_resp.sanitize(state.audit_logger.known_secrets());

            let entry = AuditEntry {
                timestamp: chrono::Utc::now(),
                route: "POST /api/execute".to_string(),
                session_id: Some(req.session_id.clone()),
                caller_sub: Some(claims.sub.clone()),
                caller_role: Some(claims.role.clone()),
                target_endpoint: Some(req.endpoint.clone()),
                method: Some(req.method.clone()),
                decision: "deny".to_string(),
                matched_rule: Some(DenyRule::BackendError),
                reason: Some(format!("Backend error: {e}")),
                injection_confidence: ml_verdict.as_ref().map(|v| v.injection_confidence),
                malicious_intent: ml_verdict.as_ref().map(|v| v.malicious_intent),
                latency_ms,
            };
            let _ = state.audit_logger.log(entry);

            return (StatusCode::BAD_GATEWAY, Json(err_resp)).into_response();
        }
    };

    let backend_status =
        StatusCode::from_u16(backend_resp.status().as_u16()).unwrap_or(StatusCode::OK);
    tracing::debug!(
        status = backend_status.as_u16(),
        total_ms = start_time.elapsed().as_millis() as u64,
        "B7 backend answered; relaying to the agent"
    );

    let mut response_headers = HeaderMap::new();
    if let Some(ct) = backend_resp.headers().get(header::CONTENT_TYPE) {
        response_headers.insert(header::CONTENT_TYPE, ct.clone());
    }

    let resp_bytes = backend_resp.bytes().await.unwrap_or_default();
    let resp_bytes = if !backend_status.is_success() {
        if let Ok(mut val) = serde_json::from_slice::<serde_json::Value>(&resp_bytes) {
            state.audit_logger.redact_value(&mut val);
            Bytes::from(serde_json::to_vec(&val).unwrap_or_else(|_| resp_bytes.to_vec()))
        } else {
            let text = String::from_utf8_lossy(&resp_bytes);
            Bytes::from(state.audit_logger.redact_string(&text).into_bytes())
        }
    } else {
        resp_bytes
    };
    let latency_ms = start_time.elapsed().as_millis() as u64;

    // Log successful execution
    let entry = AuditEntry {
        timestamp: chrono::Utc::now(),
        route: "POST /api/execute".to_string(),
        session_id: Some(req.session_id.clone()),
        caller_sub: Some(claims.sub.clone()),
        caller_role: Some(claims.role.clone()),
        target_endpoint: Some(req.endpoint.clone()),
        method: Some(req.method.clone()),
        decision: "allow".to_string(),
        matched_rule: None,
        reason: approved_by_human.then(|| crate::alerts::HUMAN_ALLOWED_ONCE.to_string()),
        injection_confidence: ml_verdict.as_ref().map(|v| v.injection_confidence),
        malicious_intent: ml_verdict.as_ref().map(|v| v.malicious_intent),
        latency_ms,
    };
    let _ = state.audit_logger.log(entry);

    (backend_status, response_headers, resp_bytes).into_response()
}

/// Emits HTTP 401 Unauthorized response and records audit log.
fn emit_unauthorized(state: &AppState, req: &ToolExecuteRequest, start_time: Instant) -> Response {
    state.alert_hub.publish(crate::alerts::new_alert(
        "unknown",
        Some(req.session_id.clone()),
        "POST /api/execute",
        &req.endpoint,
        "unauthorized_token",
        "Missing or invalid bearer token",
        None,
        None,
        false,
    ));
    let latency_ms = start_time.elapsed().as_millis() as u64;
    let resp = UnauthorizedResponse {
        error: "unauthorized".to_string(),
        reason: "Missing or invalid bearer token".to_string(),
        rule: "unauthorized_token".to_string(),
        message: "Missing or invalid bearer token".to_string(),
        decision: "deny".to_string(),
    };

    let entry = AuditEntry {
        timestamp: chrono::Utc::now(),
        route: "POST /api/execute".to_string(),
        session_id: Some(req.session_id.clone()),
        caller_sub: None,
        caller_role: None,
        target_endpoint: Some(req.endpoint.clone()),
        method: Some(req.method.clone()),
        decision: "deny".to_string(),
        matched_rule: Some(DenyRule::UnauthorizedToken),
        reason: Some("Missing or invalid bearer token".to_string()),
        injection_confidence: None,
        malicious_intent: None,
        latency_ms,
    };
    let _ = state.audit_logger.log(entry);

    (StatusCode::UNAUTHORIZED, Json(resp)).into_response()
}

/// Internal struct capturing detailed policy denial info.
struct PolicyDenyInfo {
    status: StatusCode,
    error: &'static str,
    rule: &'static str,
    message: String,
    deny_rule: DenyRule,
}

/// Evaluates policy rules for either a local CLI tool or a microservice endpoint.
fn evaluate_target_detailed(
    engine: &PolicyEngine,
    claims: &AgentClaims,
    req: &ToolExecuteRequest,
) -> Result<(), PolicyDenyInfo> {
    if let Some(tool_name) = req.resolved_tool_name(engine) {
        evaluate_tool_detailed(engine, claims, tool_name, &req.payload)
    } else if engine.get_tool(&req.endpoint).is_some() {
        evaluate_tool_detailed(engine, claims, &req.endpoint, &req.payload)
    } else {
        evaluate_policy_detailed(engine, claims, req)
    }
}

/// Evaluates tool-specific deterministic policy rules (enabled, roles, required_args, constraints).
fn evaluate_tool_detailed(
    engine: &PolicyEngine,
    claims: &AgentClaims,
    tool_name: &str,
    payload: &serde_json::Map<String, serde_json::Value>,
) -> Result<(), PolicyDenyInfo> {
    // Priority 3: Tool allowlist & enabled check
    let tool_rule = match engine.get_tool(tool_name) {
        Some(rule) if rule.enabled => rule,
        _ => {
            return Err(PolicyDenyInfo {
                status: StatusCode::FORBIDDEN,
                error: "EndpointDenied",
                rule: "endpoint",
                message: format!("Tool '{tool_name}' is not allowlisted or disabled"),
                deny_rule: DenyRule::EndpointDenied,
            });
        }
    };

    // Priority 5: Caller role authorization against allowed_roles
    let role_authorized = tool_rule.allowed_roles.iter().any(|r| r == &claims.role);
    if !role_authorized {
        return Err(PolicyDenyInfo {
            status: StatusCode::FORBIDDEN,
            error: "UnauthorizedRole",
            rule: "role",
            message: format!(
                "Role '{}' is unauthorized for tool '{tool_name}'",
                claims.role
            ),
            deny_rule: DenyRule::UnauthorizedRole,
        });
    }

    // Priority 6: Required arguments validation
    for req_arg in &tool_rule.required_args {
        match payload.get(req_arg) {
            Some(val) if !val.is_null() => {}
            _ => {
                return Err(PolicyDenyInfo {
                    status: StatusCode::FORBIDDEN,
                    error: "MissingArg",
                    rule: "required_args",
                    message: format!("Missing required argument: {req_arg}"),
                    deny_rule: DenyRule::MissingArg,
                });
            }
        }
    }

    // Priority 7: Argument constraints validation
    validate_constraints_detailed(payload, &tool_rule.constraints)
}

/// Evaluates deterministic zero-trust policy rules in strict priority order (Steps 3-7).
fn evaluate_policy_detailed(
    engine: &PolicyEngine,
    claims: &AgentClaims,
    req: &ToolExecuteRequest,
) -> Result<(), PolicyDenyInfo> {
    // Priority 3: Endpoint allowlist & enabled check
    let endpoint_rule = match engine.get_endpoint(&req.endpoint) {
        Some(rule) if rule.enabled => rule,
        _ => {
            return Err(PolicyDenyInfo {
                status: StatusCode::FORBIDDEN,
                error: "EndpointDenied",
                rule: "endpoint",
                message: "Endpoint is not allowlisted".to_string(),
                deny_rule: DenyRule::EndpointDenied,
            });
        }
    };

    // Priority 4: HTTP Method validation
    let method_allowed = endpoint_rule
        .allowed_methods
        .iter()
        .any(|m| m.eq_ignore_ascii_case(&req.method));
    if !method_allowed {
        return Err(PolicyDenyInfo {
            status: StatusCode::FORBIDDEN,
            error: "MethodNotAllowed",
            rule: "method",
            message: format!(
                "Method '{}' is not allowed for endpoint '{}'",
                req.method, req.endpoint
            ),
            deny_rule: DenyRule::EndpointDenied,
        });
    }

    // Priority 5: Caller role authorization
    let role_authorized = endpoint_rule
        .allowed_roles
        .iter()
        .any(|r| r == &claims.role);
    if !role_authorized {
        return Err(PolicyDenyInfo {
            status: StatusCode::FORBIDDEN,
            error: "UnauthorizedRole",
            rule: "role",
            message: format!(
                "Role '{}' is unauthorized for endpoint '{}'",
                claims.role, req.endpoint
            ),
            deny_rule: DenyRule::UnauthorizedRole,
        });
    }

    // Priority 6: Required arguments validation
    for req_arg in &endpoint_rule.required_args {
        match req.payload.get(req_arg) {
            Some(val) if !val.is_null() => {}
            _ => {
                return Err(PolicyDenyInfo {
                    status: StatusCode::FORBIDDEN,
                    error: "MissingArg",
                    rule: "required_args",
                    message: format!("Missing required argument: {req_arg}"),
                    deny_rule: DenyRule::MissingArg,
                });
            }
        }
    }

    // Priority 7: Argument constraints validation
    validate_constraints_detailed(&req.payload, &endpoint_rule.constraints)
}

/// Priority 7: Argument constraints validation (max_len & regex)
fn validate_constraints_detailed(
    payload: &serde_json::Map<String, serde_json::Value>,
    constraints: &std::collections::HashMap<String, crate::policy::CompiledArgConstraint>,
) -> Result<(), PolicyDenyInfo> {
    let mut sorted_keys: Vec<_> = constraints.keys().collect();
    sorted_keys.sort();

    for arg_name in sorted_keys {
        let constraint = &constraints[arg_name];
        if let Some(val) = payload.get(arg_name) {
            if val.is_null() {
                return Err(PolicyDenyInfo {
                    status: StatusCode::FORBIDDEN,
                    error: "ConstraintFailed",
                    rule: "constraints",
                    message: format!("Constraint validation failed for argument: {arg_name}"),
                    deny_rule: DenyRule::ConstraintFailed,
                });
            }

            let str_val = match val.as_str() {
                Some(s) => s,
                None => {
                    return Err(PolicyDenyInfo {
                        status: StatusCode::FORBIDDEN,
                        error: "ConstraintFailed",
                        rule: "constraints",
                        message: format!("Constraint validation failed for argument: {arg_name}"),
                        deny_rule: DenyRule::ConstraintFailed,
                    });
                }
            };

            if let Some(max_len) = constraint.max_len {
                if str_val.chars().count() > max_len {
                    return Err(PolicyDenyInfo {
                        status: StatusCode::FORBIDDEN,
                        error: "ConstraintFailed",
                        rule: "constraints",
                        message: format!("Constraint validation failed for argument: {arg_name}"),
                        deny_rule: DenyRule::ConstraintFailed,
                    });
                }
            }

            if let Some(re) = &constraint.regex {
                if !re.is_match(str_val) {
                    return Err(PolicyDenyInfo {
                        status: StatusCode::FORBIDDEN,
                        error: "ConstraintFailed",
                        rule: "constraints",
                        message: format!("Constraint validation failed for argument: {arg_name}"),
                        deny_rule: DenyRule::ConstraintFailed,
                    });
                }
            }
        }
    }

    Ok(())
}
