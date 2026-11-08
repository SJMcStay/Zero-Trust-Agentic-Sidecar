//! Empirical Adversarial Verification Suite — Challenger 2 (Milestone 4: Ticket T7)
//!
//! Focus: Route B (`POST /api/execute`) Attack Surface:
//! 1. 8-step Deny Hierarchy Precedence Matrix (Strict pairwise & multi-rule cascades)
//! 2. Authentication Attack Vectors (Missing/invalid schemes, expired/tampered JWTs, bound SHA256 mismatches, unauthorized roles)
//! 3. HMAC-SHA256 Signing Verification & Downstream Forwarding Validation

use async_trait::async_trait;
use axum::{
    body::Body,
    http::{header, Request, StatusCode},
};
use serde_json::json;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tower::ServiceExt;
use zero_trust_agentic_sidecar::{
    create_router, crypto, AppState, AuditEntry, AuditLogger, Config, DecisionEngine, DenyRule,
    KevVerdict, MlError, PolicyEngine, ToolExecuteRequest,
};

const TEST_SIDECAR_SECRET: &str = "test-sidecar-secret-32-bytes-long!";
const TEST_JWT_SECRET: &str = "test-jwt-secret-32-bytes-long-key!";
const ALT_JWT_SECRET: &str = "wrong-jwt-secret-32-bytes-long-key!";

/// Configurable Mock DecisionEngine for adversarial testing.
struct AdversarialScoringEngine {
    injection_confidence: f32,
    malicious_intent: f32,
    delay_ms: Option<u64>,
}

impl AdversarialScoringEngine {
    fn benign() -> Self {
        Self {
            injection_confidence: 0.01,
            malicious_intent: 0.01,
            delay_ms: None,
        }
    }

    fn high_risk() -> Self {
        Self {
            injection_confidence: 0.95,
            malicious_intent: 0.85,
            delay_ms: None,
        }
    }
}

#[async_trait]
impl DecisionEngine for AdversarialScoringEngine {
    fn model_id(&self) -> &str {
        "adversarial-scoring-engine"
    }

    async fn evaluate_prompt(&self, _prompt: &str) -> Result<KevVerdict, MlError> {
        if let Some(delay) = self.delay_ms {
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
        }
        Ok(KevVerdict {
            injection_confidence: self.injection_confidence,
            malicious_intent: self.malicious_intent,
            model_id: self.model_id().to_string(),
            reason: "Adversarial test prompt verdict".to_string(),
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
            reason: "Adversarial test tool verdict".to_string(),
        })
    }
}

/// Helper to spin up a test router with an in-memory audit capture and customizable ML engine.
fn setup_challenger_app(
    engine: AdversarialScoringEngine,
) -> (axum::Router, Config, Arc<std::sync::Mutex<Vec<AuditEntry>>>) {
    let config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    let policy_engine = PolicyEngine::from_file("policy.toml").expect("Valid policy.toml");
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(engine),
        audit_logger,
    );
    let app = create_router(state);
    (app, config, captured)
}

/// Helper to mint standard valid agent JWT for tests.
fn valid_agent_jwt(config: &Config) -> String {
    crypto::mint_agent_jwt(
        "agent-adversarial",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .expect("Valid JWT minting")
}

// ==============================================================================
// SECTION 1: STRICT 8-STEP DENY HIERARCHY PRECEDENCE MATRIX
// Priority 1: Malformed Request (HTTP 400, rule: "malformed")
// Priority 2: Missing / Invalid Auth (HTTP 401, rule: "unauthorized_token")
// Priority 3: Endpoint Denied (HTTP 403, rule: "endpoint")
// Priority 4: Method Not Allowed (HTTP 403, rule: "method")
// Priority 5: Role Unauthorized (HTTP 403, rule: "role")
// Priority 6: Missing Required Arg (HTTP 403, rule: "required_args")
// Priority 7: Constraint Failed (HTTP 403, rule: "constraints")
// Priority 8: ML Risk Score High (HTTP 403, rule: "ml_risk")
// ==============================================================================

// --- Rule 1 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r2_missing_auth() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Violates Rule 1 (malformed JSON) AND Rule 2 (no Authorization header)
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(b"INVALID_JSON_BODY".to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Malformed");
    assert_eq!(json["rule"], "malformed");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].matched_rule, Some(DenyRule::Malformed));
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r2_invalid_jwt() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Violates Rule 1 (malformed JSON) AND Rule 2 (garbage JWT)
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer bogus.tampered.jwt")
        .body(Body::from(b"{\"bad_json: missing_brace".to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Malformed");
    assert_eq!(json["rule"], "malformed");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::Malformed));
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r3_unallowlisted_endpoint() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // Violates Rule 1 (malformed JSON) even with valid JWT and unallowlisted endpoint in raw text
    let raw_payload = b"{\"endpoint\": \"/api/v1/forbidden\", truncated...";
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(raw_payload.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Malformed");
    assert_eq!(json["rule"], "malformed");
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r4_disallowed_method() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    let raw_payload = b"{\"method\": \"DELETE\", invalid_json...";
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(raw_payload.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r5_unauthorized_role() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Role "readonly" on email send
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let raw_payload = b"{not_a_valid_json: 123";
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(raw_payload.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r6_missing_arg() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // Missing arg and corrupted JSON
    let raw_payload = b"{\"session_id\": \"s1\", \"endpoint\": \"/api/v1/system/status\", ";
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(raw_payload.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r7_constraint_violation() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // Malformed JSON with regex violation text
    let raw_payload = b"{\"payload\": {\"to\": \"bad@attacker.com\"}, ";
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(raw_payload.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn test_precedence_r1_malformed_beats_r8_high_ml_risk() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    // Violates Rule 1 (malformed JSON) and engine has 95% injection confidence
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(b"{malformed".to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Malformed");
    assert_eq!(json["rule"], "malformed");
}

// --- Rule 2 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r2_invalid_jwt_beats_r3_unallowlisted_endpoint() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Valid JSON structure, but invalid token AND unallowlisted endpoint
    let payload = json!({
        "session_id": "sess-precedence-2-3",
        "endpoint": "/api/v1/unallowlisted/endpoint",
        "method": "POST",
        "payload": {}
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer invalid-jwt-token")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "unauthorized");
    assert_eq!(json["rule"], "unauthorized_token");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_missing_auth_beats_r3_disabled_endpoint() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Valid JSON targeting disabled endpoint (/api/v1/db/drop), but no Authorization header
    let payload = json!({
        "session_id": "sess-precedence-2-3-dis",
        "endpoint": "/api/v1/db/drop",
        "method": "POST",
        "payload": { "table": "users" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "unauthorized");
    assert_eq!(json["rule"], "unauthorized_token");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_invalid_jwt_beats_r4_disallowed_method() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Disallowed method (PATCH on /api/v1/system/status) + invalid JWT
    let payload = json!({
        "session_id": "sess-precedence-2-4",
        "endpoint": "/api/v1/system/status",
        "method": "PATCH",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer invalid-jwt")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_wrong_secret_beats_r5_unauthorized_role() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // JWT minted with WRONG secret for role "analyst" (unauthorized role)
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "analyst",
        None,
        3600,
        ALT_JWT_SECRET.as_bytes(),
    )
    .expect("Mint token with alt secret");

    let payload = json!({
        "session_id": "sess-precedence-2-5",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": { "to": "alice@ourcompany.com", "body": "Hello" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // Rule 2 takes precedence over Rule 5
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_expired_jwt_beats_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Expired JWT (-3600s) + missing required arg "service" for /api/v1/system/status
    let expired_token = crypto::mint_agent_jwt(
        "agent-001",
        "agent",
        None,
        -3600,
        config.jwt_secret.as_bytes(),
    )
    .expect("Mint expired token");

    let payload = json!({
        "session_id": "sess-precedence-2-6",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": {}
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {expired_token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_tampered_sig_beats_r7_constraint_violation() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());

    // Valid JWT, then tamper with the signature part
    let valid_token = valid_agent_jwt(&config);
    let parts: Vec<&str> = valid_token.split('.').collect();
    assert_eq!(parts.len(), 3);
    let tampered_sig = format!("{}tampered", &parts[2][..parts[2].len() - 8]);
    let tampered_token = format!("{}.{}.{}", parts[0], parts[1], tampered_sig);

    // Payload violates regex constraint: "to" address does not end with @ourcompany.com
    let payload = json!({
        "session_id": "sess-precedence-2-7",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "evil@attacker.xyz",
            "body": "Constraint violation text"
        }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {tampered_token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_precedence_r2_invalid_jwt_beats_r8_high_ml_risk() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());

    let payload = json!({
        "session_id": "sess-precedence-2-8",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer bogus")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

// --- Rule 3 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r3_endpoint_denied_beats_r4_disallowed_method() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // Unallowlisted endpoint AND disallowed method (PUT)
    let payload = json!({
        "session_id": "sess-precedence-3-4",
        "endpoint": "/api/v1/nonexistent/service",
        "method": "PUT",
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_endpoint_denied_beats_r5_unauthorized_role() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Mint token with unauthorized role "guest"
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "guest",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // Endpoint not allowlisted
    let payload = json!({
        "session_id": "sess-precedence-3-5",
        "endpoint": "/api/v1/random/action",
        "method": "POST",
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_endpoint_denied_beats_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // Unallowlisted endpoint + empty payload (missing required args if it were a valid endpoint)
    let payload = json!({
        "session_id": "sess-precedence-3-6",
        "endpoint": "/api/v1/nonexistent/endpoint",
        "method": "POST",
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_endpoint_denied_beats_r7_constraint_violation() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-precedence-3-7",
        "endpoint": "/api/v1/nonexistent/service",
        "method": "POST",
        "payload": { "to": "attacker@evil.com" }
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
    assert_eq!(json["error"], "EndpointDenied");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_disabled_endpoint_beats_r4_disallowed_method() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // /api/v1/db/drop is disabled in policy.toml, and GET is disallowed
    let payload = json!({
        "session_id": "sess-precedence-3-4-dis",
        "endpoint": "/api/v1/db/drop",
        "method": "GET",
        "payload": { "table": "users" }
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_disabled_endpoint_beats_r5_unauthorized_role() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // /api/v1/db/drop is disabled in policy.toml, caller is role "analyst"
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "analyst",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-precedence-3-5-dis",
        "endpoint": "/api/v1/db/drop",
        "method": "POST",
        "payload": { "table": "financials" }
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_disabled_endpoint_beats_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // /api/v1/db/drop requires "table", here payload is empty
    let payload = json!({
        "session_id": "sess-precedence-3-6-dis",
        "endpoint": "/api/v1/db/drop",
        "method": "POST",
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
    assert_eq!(json["error"], "EndpointDenied");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_disabled_endpoint_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-precedence-3-8-dis",
        "endpoint": "/api/v1/db/drop",
        "method": "POST",
        "payload": { "table": "users" }
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
    assert_eq!(json["error"], "EndpointDenied");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r3_endpoint_denied_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-precedence-3-8",
        "endpoint": "/api/v1/unknown/resource",
        "method": "POST",
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

// --- Rule 4 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r4_method_not_allowed_beats_r5_unauthorized_role() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // /api/v1/email/send allows ["POST"] and roles ["agent", "admin"]
    // Here: method is DELETE (disallowed) AND role is "readonly" (unauthorized)
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-precedence-4-5",
        "endpoint": "/api/v1/email/send",
        "method": "DELETE",
        "payload": { "to": "alice@ourcompany.com", "body": "msg" }
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
    assert_eq!(json["error"], "MethodNotAllowed");
    assert_eq!(json["rule"], "method");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r4_method_not_allowed_beats_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // /api/v1/system/status allows ["GET", "POST"] and requires "service"
    // Here: method is PUT (disallowed) AND "service" is missing
    let payload = json!({
        "session_id": "sess-precedence-4-6",
        "endpoint": "/api/v1/system/status",
        "method": "PUT",
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
    assert_eq!(json["error"], "MethodNotAllowed");
    assert_eq!(json["rule"], "method");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r4_method_not_allowed_beats_r7_constraint_violation() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // /api/v1/email/send: method DELETE (disallowed) + regex failure on "to"
    let payload = json!({
        "session_id": "sess-precedence-4-7",
        "endpoint": "/api/v1/email/send",
        "method": "DELETE",
        "payload": {
            "to": "invalid-address",
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
    assert_eq!(json["error"], "MethodNotAllowed");
    assert_eq!(json["rule"], "method");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_precedence_r4_method_not_allowed_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-precedence-4-8",
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
    assert_eq!(json["error"], "MethodNotAllowed");
    assert_eq!(json["rule"], "method");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

// --- Rule 5 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r5_unauthorized_role_beats_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // /api/v1/email/send requires ["to", "body"] and allows roles ["agent", "admin"]
    // Caller has role "readonly" (unauthorized) AND payload is missing "body"
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-precedence-5-6",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": { "to": "alice@ourcompany.com" }
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
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_precedence_r5_unauthorized_role_beats_r7_constraint_violation() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Caller has role "readonly" on /api/v1/email/send, AND regex on "to" violates pattern
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-precedence-5-7",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "malicious@external.com",
            "body": "Sensitive corporate dump"
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
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_precedence_r5_unauthorized_role_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-precedence-5-8",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "bob@ourcompany.com",
            "body": "Safe content"
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
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

// --- Rule 6 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r6_missing_arg_beats_r7_constraint_violation() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    // /api/v1/email/send requires ["to", "body"]
    // "to" violates regex constraint (external email) AND "body" is completely missing
    let payload = json!({
        "session_id": "sess-precedence-6-7",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "hacker@evil.org"
            // "body" missing!
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
    assert_eq!(json["error"], "MissingArg");
    assert_eq!(json["rule"], "required_args");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::MissingArg));
}

#[tokio::test]
async fn test_precedence_r6_missing_arg_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-precedence-6-8",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": {} // "service" missing
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
    assert_eq!(json["error"], "MissingArg");
    assert_eq!(json["rule"], "required_args");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::MissingArg));
}

// --- Rule 7 beats lower priority rules ---

#[tokio::test]
async fn test_precedence_r7_constraint_regex_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    // Regex violation on "to" AND high ML risk score (95% injection)
    let payload = json!({
        "session_id": "sess-precedence-7-8-regex",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "bob@unauthorized-external-domain.net",
            "body": "System status update"
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
    assert_eq!(json["error"], "ConstraintFailed");
    assert_eq!(json["rule"], "constraints");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ConstraintFailed));
}

#[tokio::test]
async fn test_precedence_r7_constraint_maxlen_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    // Max length on "body" is 2000 chars. Pass 2500 chars AND high ML risk
    let long_body = "x".repeat(2500);
    let payload = json!({
        "session_id": "sess-precedence-7-8-maxlen",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "charlie@ourcompany.com",
            "body": long_body
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
    assert_eq!(json["error"], "ConstraintFailed");
    assert_eq!(json["rule"], "constraints");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ConstraintFailed));
}

#[tokio::test]
async fn test_precedence_r7_non_string_constraint_beats_r8_high_ml_risk() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    // "service" must be a string matching ^[a-z0-9-]+$. Pass an integer `12345`.
    let payload = json!({
        "session_id": "sess-precedence-7-8-nonstr",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": {
            "service": 12345
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
    assert_eq!(json["error"], "ConstraintFailed");
    assert_eq!(json["rule"], "constraints");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ConstraintFailed));
}

// ==============================================================================
// SECTION 2: MULTI-RULE CASCADES (SIMULTANEOUS VIOLATIONS)
// ==============================================================================

#[tokio::test]
async fn test_cascade_all_8_rules_returns_r1_malformed() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());

    // 1: Malformed JSON, 2: No auth, 3: Bad endpoint, 4: Bad method, 5: Bad role, 6: Missing arg, 7: Constraint, 8: High ML
    let raw_body = b"NOT_VALID_JSON{{{";

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(raw_body.to_vec()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "Malformed");
    assert_eq!(json["rule"], "malformed");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::Malformed));
}

#[tokio::test]
async fn test_cascade_rules_2_through_8_returns_r2_unauthorized() {
    let (app, _, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());

    // Valid JSON, but:
    // 2: Invalid JWT
    // 3: Unallowlisted endpoint
    // 4: Disallowed method (PUT)
    // 5: Role (not applicable due to bad token)
    // 6: Missing required arg
    // 7: Regex violation
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-2-8",
        "endpoint": "/api/v1/rogue/execute",
        "method": "PUT",
        "payload": { "to": "bad@format@email" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer bogus.jwt.sig")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "unauthorized");
    assert_eq!(json["rule"], "unauthorized_token");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedToken));
}

#[tokio::test]
async fn test_cascade_rules_3_through_8_returns_r3_endpoint_denied() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    // Valid JWT with role "readonly"
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // 3: Unallowlisted endpoint
    // 4: Disallowed method (PATCH)
    // 5: Role "readonly"
    // 6: Missing args
    // 7: Constraint failure
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-3-8",
        "endpoint": "/api/v1/forbidden/action",
        "method": "PATCH",
        "payload": { "random": "data" }
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
    assert_eq!(json["error"], "EndpointDenied");
    assert_eq!(json["rule"], "endpoint");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_cascade_rules_4_through_8_returns_r4_method_not_allowed() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    // Role "readonly" (unauthorized for email/send)
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // /api/v1/email/send:
    // 4: Method PUT (disallowed, only POST)
    // 5: Role "readonly" (unauthorized)
    // 6: Missing arg "body"
    // 7: Regex constraint failure on "to"
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-4-8",
        "endpoint": "/api/v1/email/send",
        "method": "PUT",
        "payload": { "to": "notanemail" }
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
    assert_eq!(json["error"], "MethodNotAllowed");
    assert_eq!(json["rule"], "method");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::EndpointDenied));
}

#[tokio::test]
async fn test_cascade_rules_5_through_8_returns_r5_unauthorized_role() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "readonly",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // /api/v1/email/send:
    // Method POST is allowed!
    // 5: Role "readonly" is unauthorized
    // 6: Missing arg "body"
    // 7: Regex failure on "to"
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-5-8",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": { "to": "attacker@evil.com" }
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
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_cascade_rules_6_through_8_returns_r6_missing_arg() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    // Role "agent" is authorized
    let token = valid_agent_jwt(&config);

    // /api/v1/email/send:
    // 6: Missing arg "body"
    // 7: Regex failure on "to"
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-6-8",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": { "to": "attacker@external.com" }
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
    assert_eq!(json["error"], "MissingArg");
    assert_eq!(json["rule"], "required_args");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::MissingArg));
}

#[tokio::test]
async fn test_cascade_rules_7_through_8_returns_r7_constraint_failed() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::high_risk());
    let token = valid_agent_jwt(&config);

    // /api/v1/email/send:
    // All required args present ("to", "body")
    // 7: Regex constraint failure on "to"
    // 8: High ML risk
    let payload = json!({
        "session_id": "sess-cascade-7-8",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "unapproved@competitor.com",
            "body": "Normal body content"
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
    assert_eq!(json["error"], "ConstraintFailed");
    assert_eq!(json["rule"], "constraints");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::ConstraintFailed));
}

// ==============================================================================
// SECTION 3: AUTHENTICATION ATTACK VECTORS
// ==============================================================================

#[tokio::test]
async fn test_auth_missing_header_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-1",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_empty_header_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-2",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_whitespace_only_header_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-3",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "    ")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_scheme_basic_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-4",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Basic dXNlcjpwYXNzd29yZA==")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_scheme_digest_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-5",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(
            header::AUTHORIZATION,
            "Digest username=\"admin\", realm=\"sec\"",
        )
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_scheme_custom_token_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-6",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Token secret-agent-token-12345")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_bearer_no_space_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-auth-7",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    // "Bearer" without a space before the token
    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer{token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_bearer_empty_token_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-8",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer ")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_bearer_whitespace_token_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-9",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer    \t\t   ")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_expired_jwt_relative_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Expired 120 seconds ago
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "agent",
        None,
        -120,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-10",
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
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_expired_jwt_zero_leeway_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let now = chrono::Utc::now().timestamp() as usize;
    // Exactly 1 second in the past (strict zero-leeway check)
    let token = crypto::mint_agent_jwt_with_exp(
        "agent-001",
        "agent",
        None,
        now - 1,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-11",
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
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_tampered_signature_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let valid_token = valid_agent_jwt(&config);
    let parts: Vec<&str> = valid_token.split('.').collect();
    let corrupted_sig = format!("{}XYZ99", &parts[2][..parts[2].len() - 5]);
    let corrupted_token = format!("{}.{}.{}", parts[0], parts[1], corrupted_sig);

    let payload = json!({
        "session_id": "sess-auth-12",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {corrupted_token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_tampered_claims_payload_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token_agent = crypto::mint_agent_jwt(
        "agent-adversarial",
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();
    let token_admin = crypto::mint_agent_jwt(
        "admin-user",
        "admin",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // Splicing: Take header & sig from token_agent, but payload segment from token_admin
    let parts_agent: Vec<&str> = token_agent.split('.').collect();
    let parts_admin: Vec<&str> = token_admin.split('.').collect();
    let spliced_token = format!("{}.{}.{}", parts_agent[0], parts_admin[1], parts_agent[2]);

    let payload = json!({
        "session_id": "sess-auth-13",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {spliced_token}"))
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_wrong_secret_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Minted with wrong secret key
    let token = crypto::mint_agent_jwt("agent-001", "agent", None, 3600, ALT_JWT_SECRET.as_bytes())
        .unwrap();

    let payload = json!({
        "session_id": "sess-auth-14",
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
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_malformed_token_segments_returns_401() {
    let (app, _, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let payload = json!({
        "session_id": "sess-auth-15",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });

    // Test: random garbage
    let req1 = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer !!NOT_A_JWT!!")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let resp1 = app.clone().oneshot(req1).await.unwrap();
    assert_eq!(resp1.status(), StatusCode::UNAUTHORIZED);

    // Test: only two segments
    let req2 = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer header.payload")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let resp2 = app.clone().oneshot(req2).await.unwrap();
    assert_eq!(resp2.status(), StatusCode::UNAUTHORIZED);

    // Test: four segments
    let req3 = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, "Bearer header.payload.sig.extra")
        .body(Body::from(serde_json::to_vec(&payload).unwrap()))
        .unwrap();
    let resp3 = app.oneshot(req3).await.unwrap();
    assert_eq!(resp3.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_empty_sub_in_claims_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = crypto::mint_agent_jwt(
        "", // Empty sub
        "agent",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-16",
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
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_empty_role_in_claims_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = crypto::mint_agent_jwt(
        "agent-001",
        "", // Empty role
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-17",
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
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_payload_sha256_binding_tampered_body_returns_401() {
    let (app, config, _) = setup_challenger_app(AdversarialScoringEngine::benign());

    let original_payload_bytes = serde_json::to_vec(&json!({
        "session_id": "sess-bound-1",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    }))
    .unwrap();

    // Mint token bound to original_payload_bytes
    let bound_token = crypto::mint_agent_jwt_for_payload(
        "agent-001",
        "agent",
        &original_payload_bytes,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    // Tamper with the body before sending (e.g. changing service parameter)
    let tampered_payload = json!({
        "session_id": "sess-bound-1",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "payment-api" }
    });

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {bound_token}"))
        .body(Body::from(serde_json::to_vec(&tampered_payload).unwrap()))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    // Payload hash mismatch must trigger HTTP 401 Unauthorized
    assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
}

#[tokio::test]
async fn test_auth_payload_sha256_binding_matching_body_succeeds() {
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let mock_backend = axum::Router::new().route(
        "/api/v1/system/status",
        axum::routing::post(|| async move {
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, "application/json")],
                json!({ "status": "bound_verified" }).to_string(),
            )
        }),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_backend).await.unwrap();
    });

    let mut config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    config.flask_backend_url = format!("http://{mock_addr}");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let (audit_logger, _) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(AdversarialScoringEngine::benign()),
        audit_logger,
    );
    let app = create_router(state);

    let payload_val = json!({
        "session_id": "sess-bound-ok",
        "endpoint": "/api/v1/system/status",
        "method": "POST",
        "payload": { "service": "auth-api" }
    });
    let payload_bytes = serde_json::to_vec(&payload_val).unwrap();

    let bound_token = crypto::mint_agent_jwt_for_payload(
        "agent-001",
        "agent",
        &payload_bytes,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let req = Request::builder()
        .method("POST")
        .uri("/api/execute")
        .header(header::CONTENT_TYPE, "application/json")
        .header(header::AUTHORIZATION, format!("Bearer {bound_token}"))
        .body(Body::from(payload_bytes))
        .unwrap();

    let resp = app.oneshot(req).await.unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn test_auth_role_analyst_denied_on_email_send_403() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // Role "analyst" is not in allowed_roles for /api/v1/email/send
    let token = crypto::mint_agent_jwt(
        "analyst-user",
        "analyst",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-role-1",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "compliance@ourcompany.com",
            "body": "Quarterly audit report"
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
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");
    assert_eq!(json["decision"], "deny");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_auth_role_analyst_denied_on_system_status_403() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    // /api/v1/system/status allows ["agent", "admin", "readonly"]. "analyst" is forbidden.
    let token = crypto::mint_agent_jwt(
        "analyst-user",
        "analyst",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-role-2",
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
    assert_eq!(resp.status(), StatusCode::FORBIDDEN);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "UnauthorizedRole");
    assert_eq!(json["rule"], "role");

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

#[tokio::test]
async fn test_auth_role_operator_denied_on_email_send_403() {
    let (app, config, captured) = setup_challenger_app(AdversarialScoringEngine::benign());
    let token = crypto::mint_agent_jwt(
        "operator-user",
        "operator",
        None,
        3600,
        config.jwt_secret.as_bytes(),
    )
    .unwrap();

    let payload = json!({
        "session_id": "sess-auth-role-3",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "bob@ourcompany.com",
            "body": "Reboot notification"
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

    let entries = captured.lock().unwrap();
    assert_eq!(entries[0].matched_rule, Some(DenyRule::UnauthorizedRole));
}

// ==============================================================================
// SECTION 4: HMAC SIGNING HEADERS ON APPROVED FORWARDING
// ==============================================================================

#[tokio::test]
async fn test_hmac_forwarding_approved_call_headers_and_signature_verification() {
    // 1. Setup mock backend to inspect and verify incoming sidecar headers
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let sidecar_secret_bytes = TEST_SIDECAR_SECRET.as_bytes();

    let mock_backend = axum::Router::new().route(
        "/api/v1/system/status",
        axum::routing::post(
            move |headers: axum::http::HeaderMap, body: axum::body::Bytes| async move {
                // Header assertions
                let ts_header = headers
                    .get("x-sidecar-timestamp")
                    .expect("x-sidecar-timestamp header missing")
                    .to_str()
                    .unwrap();
                let sig_header = headers
                    .get("x-sidecar-signature")
                    .expect("x-sidecar-signature header missing")
                    .to_str()
                    .unwrap();

                let timestamp: u64 = ts_header.parse().expect("Timestamp must be integer");
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_secs();

                // 1. Freshness check: timestamp must be within 5 seconds
                assert!(
                    (timestamp as i64 - now as i64).abs() <= 5,
                    "Timestamp {timestamp} skewed relative to {now}"
                );

                // 2. Signature match: recalculate expected signature
                let expected_sig = crypto::sign_payload(sidecar_secret_bytes, timestamp, &body);
                assert_eq!(
                    sig_header, expected_sig,
                    "Forwarded HMAC signature does not match crypto::sign_payload!"
                );

                // 3. Verify via crypto::verify_payload_signature
                crypto::verify_payload_signature(
                    sidecar_secret_bytes,
                    timestamp,
                    &body,
                    sig_header,
                    30,
                )
                .expect("verify_payload_signature passed");

                // 4. Verify body is exact payload json
                let json_body: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(json_body["service"], "auth-api");

                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    json!({
                        "status": "healthy",
                        "service": "auth-api",
                        "timestamp": timestamp
                    })
                    .to_string(),
                )
            },
        ),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_backend).await.unwrap();
    });

    // 2. Setup sidecar pointing to mock backend
    let mut config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    config.flask_backend_url = format!("http://{mock_addr}");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(AdversarialScoringEngine::benign()),
        audit_logger,
    );
    let app = create_router(state);

    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-hmac-forward-1",
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
    assert_eq!(json["status"], "healthy");
    assert_eq!(json["service"], "auth-api");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].decision, "allow");
    assert_eq!(entries[0].matched_rule, None);
    assert_eq!(
        entries[0].target_endpoint.as_deref(),
        Some("/api/v1/system/status")
    );
}

#[tokio::test]
async fn test_hmac_forwarding_post_email_exact_signature() {
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let sidecar_secret_bytes = TEST_SIDECAR_SECRET.as_bytes();

    let mock_backend = axum::Router::new().route(
        "/api/v1/email/send",
        axum::routing::post(
            move |headers: axum::http::HeaderMap, body: axum::body::Bytes| async move {
                let ts: u64 = headers
                    .get("x-sidecar-timestamp")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let sig = headers
                    .get("x-sidecar-signature")
                    .unwrap()
                    .to_str()
                    .unwrap();

                // Strict signature validation
                crypto::verify_payload_signature(sidecar_secret_bytes, ts, &body, sig, 10)
                    .expect("Signature valid");

                let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
                assert_eq!(json["to"], "security@ourcompany.com");
                assert_eq!(json["body"], "Deployment approved.");

                (
                    StatusCode::ACCEPTED,
                    [(header::CONTENT_TYPE, "application/json")],
                    json!({ "queued": true, "message_id": "msg-xyz-987" }).to_string(),
                )
            },
        ),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_backend).await.unwrap();
    });

    let mut config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    config.flask_backend_url = format!("http://{mock_addr}");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(AdversarialScoringEngine::benign()),
        audit_logger,
    );
    let app = create_router(state);

    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-hmac-email-1",
        "endpoint": "/api/v1/email/send",
        "method": "POST",
        "payload": {
            "to": "security@ourcompany.com",
            "body": "Deployment approved."
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
    assert_eq!(resp.status(), StatusCode::ACCEPTED);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["queued"], true);
    assert_eq!(json["message_id"], "msg-xyz-987");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].decision, "allow");
}

#[tokio::test]
async fn test_hmac_forwarding_get_method_with_payload() {
    let mock_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let mock_addr = mock_listener.local_addr().unwrap();

    let sidecar_secret_bytes = TEST_SIDECAR_SECRET.as_bytes();

    let mock_backend = axum::Router::new().route(
        "/api/v1/system/status",
        axum::routing::get(
            move |headers: axum::http::HeaderMap, body: axum::body::Bytes| async move {
                let ts: u64 = headers
                    .get("x-sidecar-timestamp")
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .parse()
                    .unwrap();
                let sig = headers
                    .get("x-sidecar-signature")
                    .unwrap()
                    .to_str()
                    .unwrap();

                // Signature verification on GET request with payload
                crypto::verify_payload_signature(sidecar_secret_bytes, ts, &body, sig, 10)
                    .expect("Valid signature on GET");

                (
                    StatusCode::OK,
                    [(header::CONTENT_TYPE, "application/json")],
                    json!({ "status": "get_ok" }).to_string(),
                )
            },
        ),
    );

    tokio::spawn(async move {
        axum::serve(mock_listener, mock_backend).await.unwrap();
    });

    let mut config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    config.flask_backend_url = format!("http://{mock_addr}");

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let (audit_logger, _) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(AdversarialScoringEngine::benign()),
        audit_logger,
    );
    let app = create_router(state);

    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-hmac-get",
        "endpoint": "/api/v1/system/status",
        "method": "GET",
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
}

#[tokio::test]
async fn test_hmac_forwarding_downstream_connection_refusal_returns_502() {
    let mut config = Config::for_test(TEST_SIDECAR_SECRET, TEST_JWT_SECRET);
    // Bind to an unreachable port
    config.flask_backend_url = "http://127.0.0.1:59997".to_string();

    let policy_engine = PolicyEngine::from_file("policy.toml").unwrap();
    let (audit_logger, captured) = AuditLogger::capture();
    let state = AppState::test_state(
        config.clone(),
        policy_engine,
        Arc::new(AdversarialScoringEngine::benign()),
        audit_logger,
    );
    let app = create_router(state);

    let token = valid_agent_jwt(&config);

    let payload = json!({
        "session_id": "sess-hmac-fail-conn",
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
    assert_eq!(resp.status(), StatusCode::BAD_GATEWAY);

    let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["error"], "BackendError");
    assert_eq!(json["rule"], "backend_error");

    let entries = captured.lock().unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(entries[0].matched_rule, Some(DenyRule::BackendError));
}
