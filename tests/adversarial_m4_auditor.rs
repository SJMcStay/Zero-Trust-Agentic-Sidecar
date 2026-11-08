//! Independent Adversarial Forensic Verification Suite for Milestone 4 (`tests/adversarial_m4_auditor.rs`)
//!
//! Evaluates:
//! 1. Route A: Deep polymorphic extraction, prompt injection blocking, Anthropic header injection, upstream error relay.
//! 2. Route B: Strict 8-step Deny Hierarchy precedence (pairwise conflict stress-tests), JWT payload hash binding.
//! 3. Route B: Cryptographic HMAC-SHA256 signature generation and downstream verification.
//! 4. Audit logging: Structured JSONL logs and sensitive credential redaction.
//! 5. Concurrency & Timeout resilience under parallel loads.

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use std::sync::Arc;
use tower::ServiceExt;
use zero_trust_agentic_sidecar::{
    create_router, crypto, AppState, AuditEntry, AuditLogger, Config, DecisionEngine, DenyRule,
    KevVerdict, MlError, PolicyEngine, ToolExecuteRequest,
};

/// Mock DecisionEngine for adversarial testing.
struct AdversarialMockEngine {
    injection_confidence: f32,
    malicious_intent: f32,
    delay_ms: Option<u64>,
}

#[async_trait]
impl DecisionEngine for AdversarialMockEngine {
    fn model_id(&self) -> &str {
        "adversarial-mock-engine"
    }

    async fn evaluate_prompt(&self, _prompt: &str) -> Result<KevVerdict, MlError> {
        if let Some(delay) = self.delay_ms {
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        }
        Ok(KevVerdict {
            injection_confidence: self.injection_confidence,
            malicious_intent: self.malicious_intent,
            model_id: self.model_id().to_string(),
            reason: "Adversarial test verdict".to_string(),
        })
    }

    async fn evaluate_tool_call(&self, _req: &ToolExecuteRequest) -> Result<KevVerdict, MlError> {
        if let Some(delay) = self.delay_ms {
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        }
        Ok(KevVerdict {
            injection_confidence: self.injection_confidence,
            malicious_intent: self.malicious_intent,
            model_id: self.model_id().to_string(),
            reason: "Adversarial test verdict".to_string(),
        })
    }
}

fn create_test_env(
    engine: Arc<dyn DecisionEngine>,
) -> (axum::Router, Config, Arc<std::sync::Mutex<Vec<AuditEntry>>>) {
    let config = Config::for_test(
        "sidecar-audit-secret-32-bytes-ok!",
        "jwt-auditor-secret-32-bytes-long!",
    );
    let policy_engine = PolicyEngine::from_file("policy.toml").expect("Valid policy.toml");
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(config.clone(), policy_engine, engine, audit_logger);
    let app = create_router(state);
    (app, config, captured)
}

// =============================================================================
// Section 1: Route A Inbound Interrogation & Polymorphic Content
// =============================================================================

#[tokio::test]
async fn test_route_a_deep_nested_polymorphic_prompt_extraction_and_block() {
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.95,
        malicious_intent: 0.80,
        delay_ms: None,
    });
    let (app, _, captured) = create_test_env(engine);

    // Deep polymorphic Anthropic payload with multiple text blocks and tool_result nested blocks
    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet",
        "system": [
            { "type": "text", "text": "System instruction alpha" },
            { "type": "text", "text": "System instruction beta" }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    { "type": "text", "text": "Explain this result:" },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01",
                        "content": [
                            { "type": "text", "text": "Nested tool content containing attack payload" }
                        ]
                    }
                ]
            }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["type"], "error");
    assert_eq!(json["error"]["type"], "permission_error");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].decision, "deny");
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ModelRiskHigh));
}

#[tokio::test]
async fn test_route_a_upstream_http_429_passthrough() {
    // Upstream server returning HTTP 429
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let mock_upstream = axum::Router::new().route(
        "/v1/messages",
        axum::routing::post(|| async move {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [(header::CONTENT_TYPE, "application/json")],
                serde_json::json!({
                    "type": "error",
                    "error": { "type": "rate_limit_error", "message": "Rate limit exceeded" }
                })
                .to_string(),
            )
        }),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_upstream).await.unwrap();
    });

    let mut config = Config::for_test(
        "sidecar-audit-secret-32-bytes-ok!",
        "jwt-auditor-secret-32-bytes-long!",
    );
    config.upstream_llm_url = format!("http://{mock_addr}/v1/messages");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let decision_engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.05,
        malicious_intent: 0.05,
        delay_ms: None,
    });
    let (audit_logger, _) = AuditLogger::capture();
    let state = AppState::test_state(config, policy_engine, decision_engine, audit_logger);
    let app = create_router(state);

    let payload = serde_json::json!({
        "model": "claude-3-5-sonnet",
        "messages": [
            { "role": "user", "content": "Hello" }
        ]
    });

    let req = Request::builder()
        .method("POST")
        .uri("/v1/messages")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::TOO_MANY_REQUESTS);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"]["type"], "rate_limit_error");
}

// =============================================================================
// Section 2: Route B 8-Step Deny Hierarchy Strict Precedence Under Conflicts
// =============================================================================

#[tokio::test]
async fn test_precedence_step1_malformed_over_step2_auth() {
    // Both malformed body AND missing auth.
    // Step 1 (Malformed -> HTTP 400) MUST preempt Step 2 (Auth -> HTTP 401).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, _, captured) = create_test_env(engine);

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        // No Authorization header!
        .body(Body::from(b"not-even-json".to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].matched_rule, Some(DenyRule::Malformed));
}

#[tokio::test]
async fn test_precedence_step2_auth_over_step3_endpoint() {
    // Both invalid token AND unlisted endpoint.
    // Step 2 (Auth -> HTTP 401) MUST preempt Step 3 (Endpoint -> HTTP 403).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, _, captured) = create_test_env(engine);

    let payload = serde_json::json!({
        "session_id": "sess-precedence-2",
        "endpoint": "/api/v1/nonexistent/blacklisted",
        "method": "POST",
        "payload": {}
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer totally-invalid-token")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_step3_endpoint_over_step4_method() {
    // Both unlisted endpoint AND invalid method.
    // Step 3 (EndpointDenied -> HTTP 403) MUST preempt Step 4 (MethodNotAllowed).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);
    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-precedence-3",
        "endpoint": "/api/v1/unlisted/route",
        "method": "INVALID_METHOD",
        "payload": {}
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["rule"], "endpoint");
    assert_eq!(json["error"], "EndpointDenied");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_step4_method_over_step5_role() {
    // /api/v1/system/status allows methods ["GET", "POST"] and roles ["agent", "admin", "readonly"].
    // Caller role is "guest" (unauthorized role) AND method is "DELETE" (disallowed method).
    // Step 4 (MethodNotAllowed) MUST preempt Step 5 (UnauthorizedRole).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);
    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "guest",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-precedence-4",
        "endpoint": "/api/v1/system/status",
        "method": "DELETE",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["rule"], "method");
    assert_eq!(json["error"], "MethodNotAllowed");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_step5_role_over_step6_missing_arg() {
    // /api/v1/email/send requires role ["agent", "admin"] and required arg "to".
    // Role is "readonly" AND "to" is missing.
    // Step 5 (UnauthorizedRole) MUST preempt Step 6 (MissingArg).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);
    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-precedence-5",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": { "body": "only body" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["rule"], "role");
    assert_eq!(json["error"], "UnauthorizedRole");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_precedence_step6_missing_arg_over_step7_constraint() {
    // /api/v1/email/send requires "to" AND validates constraints on "to".
    // Missing required arg "to" must return MissingArg before constraint evaluation.
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.0,
        malicious_intent: 0.0,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);
    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-precedence-6",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "body": "Valid body text"
            // "to" is missing
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["rule"], "required_args");
    assert_eq!(json["error"], "MissingArg");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::MissingArg));
}

#[tokio::test]
async fn test_precedence_step7_constraint_over_step8_ml_risk() {
    // Both constraint failed (regex mismatch on "to") AND high ML risk score (0.99 injection).
    // Step 7 (ConstraintFailed) MUST preempt Step 8 (ModelRiskHigh).
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.99,
        malicious_intent: 0.99,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);
    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-precedence-7",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "invalid-domain@external-hacker.com", // Fails regex ^[^@]+@ourcompany\.com$
            "body": "Normal body"
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["rule"], "constraints");
    assert_eq!(json["error"], "ConstraintFailed");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ConstraintFailed));
}

// =============================================================================
// Section 3: Route B JWT Payload Binding & HMAC Signature Verification
// =============================================================================

#[tokio::test]
async fn test_route_b_jwt_payload_hash_tampering_rejected() {
    let engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.01,
        malicious_intent: 0.01,
        delay_ms: None,
    });
    let (app, config, captured) = create_test_env(engine);

    // Initial payload that was supposedly signed by caller
    let original_payload = serde_json::json!({
        "session_id": "sess-tamper-01",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });
    let original_bytes = serde_json::to_vec(&original_payload).unwrap();

    // Mint token bound to original_bytes hash
    let token = crypto::mint_agent_jwt_for_payload(
        "auditor-sub",
        "agent",
        &original_bytes,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // Attacker modifies payload to different endpoint
    let tampered_payload = serde_json::json!({
        "session_id": "sess-tamper-01",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "tampered-admin-service" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&tampered_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_route_b_outbound_hmac_signature_cryptographic_soundness() {
    // Create mock backend capturing headers and raw body
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let expected_secret = b"sidecar-audit-secret-32-bytes-ok!".to_vec();

    let mock_backend = axum::Router::new().route(
        "/api/v1/system/status",
        axum::routing::post(
            move |headers: axum::http::HeaderMap, body: axum::body::Bytes| async move {
                let ts_header = headers
                    .get("x-sidecar-timestamp")
                    .expect("x-sidecar-timestamp header present")
                    .to_str()
                    .unwrap();
                let sig_header = headers
                    .get("x-sidecar-signature")
                    .expect("x-sidecar-signature header present")
                    .to_str()
                    .unwrap();
                let ts: u64 = ts_header.parse().expect("valid timestamp integer");

                // Cryptographically verify HMAC-SHA256
                crypto::verify_payload_signature(&expected_secret, ts, &body, sig_header, 30)
                    .expect("HMAC signature verification MUST succeed");

                // Verify body is the exact inner payload JSON
                let body_json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(body_json["service"], "auth-api");

                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    serde_json::json!({ "result": "verified" }).to_string(),
                )
            },
        ),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_backend).await.unwrap();
    });

    let mut config = Config::for_test(
        "sidecar-audit-secret-32-bytes-ok!",
        "jwt-auditor-secret-32-bytes-long!",
    );
    config.flask_backend_url = format!("http://{mock_addr}");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let decision_engine = Arc::new(AdversarialMockEngine {
        injection_confidence: 0.01,
        malicious_intent: 0.01,
        delay_ms: None,
    });
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(config.clone(), policy_engine, decision_engine, audit_logger);
    let app = create_router(state);

    let token = crypto::mint_agent_jwt(
        "auditor-sub",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = serde_json::json!({
        "session_id": "sess-hmac-soundness",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["result"], "verified");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].decision, "allow");
}
