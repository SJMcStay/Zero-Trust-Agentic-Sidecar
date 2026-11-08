//! Comprehensive Adversarial Challenge Suite for Milestone 2 (Tickets T3 & T5)
//!
//! Empirical Challenger 1 verification:
//! 1. Policy Rule Priority Matrix & Conflicting Deny Conditions
//! 2. Regex Constraints: ReDoS resistance, boundary cases, case sensitivity, trailing newlines, special Unicode
//! 3. Audit Redaction: Embed actual secrets into diagnostics and test serialized JSONL and entry sanitization

use chrono::Utc;
use serde_json::json;
use std::sync::Arc;
use std::time::Instant;
use zero_trust_agentic_sidecar::*;

/// Helper to create a standard AgentClaims instance.
fn make_claims(sub: &str, role: &str) -> AgentClaims {
    AgentClaims {
        sub: sub.to_string(),
        role: role.to_string(),
        exp: 2_000_000_000,
        payload_sha256: None,
    }
}

/// Helper to create a ToolExecuteRequest instance.
fn make_request(endpoint: &str, method: &str, payload: serde_json::Value) -> ToolExecuteRequest {
    ToolExecuteRequest {
        session_id: "sess-adversarial-001".to_string(),
        endpoint: endpoint.to_string(),
        method: method.to_string(),
        payload: payload.as_object().cloned().unwrap_or_default(),
        context: ToolContext::default(),
    }
}

/// Robust multi-endpoint test policy TOML for adversarial evaluation.
const ADVERSARIAL_POLICY_TOML: &str = r#"
[defaults]
max_injection_confidence = 0.70
max_malicious_intent = 0.50
model_timeout_ms = 500
fail_closed = true
require_jwt = true

[endpoints."/api/v1/system/status"]
backend_url = "http://127.0.0.1:5000/api/v1/system/status"
enabled = true
allowed_methods = ["GET", "POST"]
allowed_roles = ["agent", "admin"]
required_args = ["service"]
constraints.service = { regex = "^[a-z0-9-]+$", max_len = 64 }

[endpoints."/api/v1/email/send"]
backend_url = "http://127.0.0.1:5000/api/v1/email/send"
enabled = true
allowed_methods = ["POST"]
allowed_roles = ["agent", "admin"]
required_args = ["to", "body"]
constraints.to = { regex = "^[^@]+@ourcompany\\.com$", max_len = 128 }
constraints.body = { max_len = 2000 }

[endpoints."/api/v1/db/drop"]
backend_url = "http://127.0.0.1:5000/api/v1/db/drop"
enabled = false
allowed_methods = ["POST"]
allowed_roles = ["admin"]
required_args = ["table"]
constraints.table = { regex = "^[a-zA-Z0-9_]+$", max_len = 32 }

[endpoints."/api/v1/audit/search"]
backend_url = "http://127.0.0.1:5000/api/v1/audit/search"
enabled = true
allowed_methods = ["POST"]
allowed_roles = ["auditor", "admin"]
required_args = ["query"]
constraints.query = { max_len = 100 }
constraints.filter = { regex = "^(info|warn|error)$", max_len = 10 }
"#;

// ==============================================================================
// 1. POLICY RULE PRIORITY MATRIX & CONFLICTING DENY CONDITIONS
// ==============================================================================

#[test]
fn test_priority_1_disabled_endpoint_wins_over_all_other_violations() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // /api/v1/db/drop is enabled = false.
    // We intentionally violate:
    // - Method: "DELETE" (not allowed: allowed is "POST")
    // - Role: "unauthorized_role" (allowed is "admin")
    // - Missing required arg: "table" is absent
    // - Invalid argument constraints: "invalid_key" present
    let claims = make_claims("bad-actor", "unauthorized_role");
    let req = make_request("/api/v1/db/drop", "DELETE", json!({"unknown": "val"}));

    let result = engine.evaluate(&claims, &req);
    assert_eq!(
        result,
        Err(DenyRule::EndpointDenied),
        "Step 1 (EndpointDenied for disabled endpoint) must win over method, role, and argument errors"
    );
}

#[test]
fn test_priority_1_unlisted_endpoint_wins_over_all_other_violations() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // /api/v1/nonexistent is unlisted.
    let claims = make_claims("bad-actor", "unauthorized_role");
    let req = make_request("/api/v1/nonexistent", "PATCH", json!({"foo": "bar"}));

    let result = engine.evaluate(&claims, &req);
    assert_eq!(
        result,
        Err(DenyRule::EndpointDenied),
        "Step 1 (EndpointDenied for unlisted endpoint) must win first"
    );
}

#[test]
fn test_priority_2_invalid_method_wins_over_role_and_arguments() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // /api/v1/email/send is enabled = true, but allowed_methods = ["POST"].
    // We call with method = "GET", invalid role = "readonly", missing required args = ["to", "body"].
    let claims = make_claims("attacker", "readonly");
    let req = make_request("/api/v1/email/send", "GET", json!({}));

    let result = engine.evaluate(&claims, &req);
    assert_eq!(
        result,
        Err(DenyRule::EndpointDenied),
        "Step 2 (EndpointDenied for method mismatch) must win over role mismatch and missing args"
    );
}

#[test]
fn test_priority_2_method_case_insensitivity_and_spacing() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("valid-agent", "agent");

    // "post", "POST", "Post", "pOsT" should all be accepted
    for method in &["post", "POST", "Post", "pOsT"] {
        let req = make_request(
            "/api/v1/email/send",
            method,
            json!({
                "to": "test@ourcompany.com",
                "body": "Hello world"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req),
            Ok(()),
            "Method comparison must be case-insensitive for '{}'",
            method
        );
    }

    // Trailing spaces, leading spaces, or null bytes must NOT match
    for bad_method in &["POST ", " POST", "POST\0", "POST\n", "GET/POST"] {
        let req = make_request(
            "/api/v1/email/send",
            bad_method,
            json!({
                "to": "test@ourcompany.com",
                "body": "Hello world"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req),
            Err(DenyRule::EndpointDenied),
            "Method with whitespace/null byte '{}' must fail with EndpointDenied",
            bad_method
        );
    }
}

#[test]
fn test_priority_3_role_mismatch_wins_over_missing_args_and_constraint_violations() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // /api/v1/email/send allows ["agent", "admin"].
    // Caller has role = "guest".
    // Payload is missing required args "to" and "body".
    let claims = make_claims("guest-user", "guest");
    let req = make_request("/api/v1/email/send", "POST", json!({}));

    let result = engine.evaluate(&claims, &req);
    assert_eq!(
        result,
        Err(DenyRule::UnauthorizedRole),
        "Step 3 (UnauthorizedRole) must win over MissingArg and ConstraintFailed"
    );
}

#[test]
fn test_priority_4_missing_arg_wins_over_constraint_violations() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // Endpoint requires "to" and "body".
    // Caller provides "to" with an invalid regex and length overflow, but omits "body".
    let claims = make_claims("agent-1", "agent");
    let req = make_request(
        "/api/v1/email/send",
        "POST",
        json!({
            "to": "invalid-email-violating-regex-and-too-long-".repeat(10)
            // "body" is missing!
        }),
    );

    let result = engine.evaluate(&claims, &req);
    assert_eq!(
        result,
        Err(DenyRule::MissingArg),
        "Step 4 (MissingArg) must win when a required argument is missing, even if present arguments violate constraints"
    );
}

#[test]
fn test_priority_4_null_required_arg_triggers_missing_arg() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // "to" is null in JSON
    let req = make_request(
        "/api/v1/email/send",
        "POST",
        json!({
            "to": null,
            "body": "Valid body text"
        }),
    );

    assert_eq!(
        engine.evaluate(&claims, &req),
        Err(DenyRule::MissingArg),
        "Explicit JSON null for a required argument must trigger MissingArg"
    );
}

#[test]
fn test_priority_5_constraint_failure_when_all_prior_steps_pass() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // Endpoint, Method, Role are valid. Required args "to" and "body" are both present.
    // But "to" fails the regex constraint (^[^@]+@ourcompany\.com$).
    let req = make_request(
        "/api/v1/email/send",
        "POST",
        json!({
            "to": "attacker@evil-domain.com",
            "body": "Valid body text"
        }),
    );

    assert_eq!(
        engine.evaluate(&claims, &req),
        Err(DenyRule::ConstraintFailed),
        "Step 5 (ConstraintFailed) must be returned when arguments fail constraints"
    );
}

#[test]
fn test_combinatorial_priority_matrix_exhaustive() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();

    struct TestCase {
        name: &'static str,
        endpoint: &'static str,
        method: &'static str,
        role: &'static str,
        payload: serde_json::Value,
        expected: Result<(), DenyRule>,
    }

    let cases = vec![
        TestCase {
            name: "1. Disabled endpoint + bad method + bad role + bad args",
            endpoint: "/api/v1/db/drop",
            method: "DELETE",
            role: "hacker",
            payload: json!({}),
            expected: Err(DenyRule::EndpointDenied),
        },
        TestCase {
            name: "2. Disabled endpoint + good method + good role + good args",
            endpoint: "/api/v1/db/drop",
            method: "POST",
            role: "admin",
            payload: json!({"table": "users"}),
            expected: Err(DenyRule::EndpointDenied),
        },
        TestCase {
            name: "3. Enabled endpoint + bad method + bad role + bad args",
            endpoint: "/api/v1/system/status",
            method: "DELETE",
            role: "intruder",
            payload: json!({}),
            expected: Err(DenyRule::EndpointDenied),
        },
        TestCase {
            name: "4. Enabled endpoint + good method + bad role + bad args",
            endpoint: "/api/v1/system/status",
            method: "GET",
            role: "unauthorized_role",
            payload: json!({}),
            expected: Err(DenyRule::UnauthorizedRole),
        },
        TestCase {
            name: "5. Enabled endpoint + good method + good role + missing args",
            endpoint: "/api/v1/system/status",
            method: "GET",
            role: "agent",
            payload: json!({}),
            expected: Err(DenyRule::MissingArg),
        },
        TestCase {
            name: "6. Enabled endpoint + good method + good role + present args but constraint failure",
            endpoint: "/api/v1/system/status",
            method: "GET",
            role: "agent",
            payload: json!({"service": "INVALID_UPPERCASE_SERVICE!"}),
            expected: Err(DenyRule::ConstraintFailed),
        },
        TestCase {
            name: "7. Enabled endpoint + good method + good role + all constraints satisfied",
            endpoint: "/api/v1/system/status",
            method: "GET",
            role: "agent",
            payload: json!({"service": "valid-service-123"}),
            expected: Ok(()),
        },
    ];

    for tc in cases {
        let claims = make_claims("tester", tc.role);
        let req = make_request(tc.endpoint, tc.method, tc.payload);
        let res = engine.evaluate(&claims, &req);
        assert_eq!(
            res, tc.expected,
            "Failed test case '{}': expected {:?}, got {:?}",
            tc.name, tc.expected, res
        );
    }
}

// ==============================================================================
// 2. REGEX CONSTRAINTS: REDOS, BOUNDARY CASES, CASE SENSITIVITY, UNICODE
// ==============================================================================

#[test]
fn test_redos_resistance_linear_time_guarantee() {
    // Construct a policy with pathological regex patterns known to trigger exponential ReDoS
    // in backtracking engines (NFA/PikeVM guarantees linear time).
    let redos_policy = r#"
    [defaults]
    [endpoints."/api/v1/test/redos"]
    backend_url = "http://localhost:5000"
    enabled = true
    allowed_methods = ["POST"]
    allowed_roles = ["agent"]
    required_args = ["payload"]
    constraints.payload = { regex = "^(a+)+$", max_len = 200000 }
    "#;

    let engine = PolicyEngine::from_str(redos_policy).unwrap();
    let claims = make_claims("agent-1", "agent");

    // Attack input: 50,000 'a' characters ending with '!' (triggers massive backtracking in naive engines)
    let attack_str = format!("{}!", "a".repeat(50_000));
    let req = make_request("/api/v1/test/redos", "POST", json!({"payload": attack_str}));

    let start = Instant::now();
    let result = engine.evaluate(&claims, &req);
    let duration = start.elapsed();

    // In a backtracking engine, this would hang for years or crash the stack.
    // With Rust regex (linear DFA / PikeVM), it completes within milliseconds.
    assert_eq!(
        result,
        Err(DenyRule::ConstraintFailed),
        "Attack string must fail constraint matching"
    );
    assert!(
        duration.as_millis() < 250,
        "ReDoS attack took {}ms; Rust regex must execute in linear time (< 250ms)",
        duration.as_millis()
    );
}

#[test]
fn test_trailing_newlines_and_crlf_rejection() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // In many regex engines without strict anchor semantics (such as Python or PCRE),
    // $ matches before \n at the end of the string.
    // In Rust regex, $ matches strictly at the end of the haystack in default mode.
    // Therefore, an email with a trailing newline or CRLF must fail the regex ^[^@]+@ourcompany\.com$
    let trailing_variations = vec![
        "alice@ourcompany.com\n",
        "alice@ourcompany.com\r\n",
        "alice@ourcompany.com\r",
        "alice@ourcompany.com ",
        "alice@ourcompany.com\t",
    ];

    for email in trailing_variations {
        let req = make_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": email,
                "body": "Normal notification body"
            }),
        );
        let res = engine.evaluate(&claims, &req);
        assert_eq!(
            res,
            Err(DenyRule::ConstraintFailed),
            "Email '{}' with trailing newline or whitespace must be rejected by regex constraint",
            email.escape_debug()
        );
    }

    // Observation: Leading newline '\nalice@ourcompany.com' matches [^@]+ because
    // [^@] matches any character except '@', including '\n'.
    let leading_newline_email = "\nalice@ourcompany.com";
    let req_leading = make_request(
        "/api/v1/email/send",
        "POST",
        json!({
            "to": leading_newline_email,
            "body": "Normal notification body"
        }),
    );
    let res_leading = engine.evaluate(&claims, &req_leading);
    assert_eq!(
        res_leading,
        Ok(()),
        "Leading newline matches [^@]+ because '\\n' != '@'"
    );
}

#[test]
fn test_regex_case_sensitivity_enforcement() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // /api/v1/system/status regex is ^[a-z0-9-]+$ (lowercase only)
    let bad_cases = vec![
        "AUTH-SERVICE",
        "Auth-Service",
        "authService",
        "DATABASE_V1",
        "SYSTEM_STATUS",
    ];

    for service in bad_cases {
        let req = make_request("/api/v1/system/status", "GET", json!({"service": service}));
        let res = engine.evaluate(&claims, &req);
        assert_eq!(
            res,
            Err(DenyRule::ConstraintFailed),
            "Service '{}' containing uppercase must fail lowercase-only regex",
            service
        );
    }

    // Email domain case sensitivity: @ourcompany.com is lowercase in regex
    let bad_domain_cases = vec![
        "alice@OURCOMPANY.COM",
        "alice@OurCompany.com",
        "alice@ourcompany.Com",
    ];

    for email in bad_domain_cases {
        let req = make_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": email,
                "body": "Notification"
            }),
        );
        let res = engine.evaluate(&claims, &req);
        assert_eq!(
            res,
            Err(DenyRule::ConstraintFailed),
            "Email '{}' with uppercase domain must fail regex",
            email
        );
    }
}

#[test]
fn test_special_unicode_homoglyphs_and_injection() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // 1. Cyrillic small 'а' (U+0430) instead of ASCII 'a' (U+0061) in service name
    // Regex is ^[a-z0-9-]+$ which is ASCII [a-z]
    let cyrillic_service = "\u{0430}uth-service";
    let req1 = make_request(
        "/api/v1/system/status",
        "GET",
        json!({"service": cyrillic_service}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req1),
        Err(DenyRule::ConstraintFailed),
        "Cyrillic homoglyph in service name must not match ASCII [a-z]"
    );

    // 2. Zero-width spaces, RTL overrides, null bytes
    let sneaky_services = vec![
        "auth\u{200B}service", // Zero-width space
        "auth\u{202E}service", // RTL override
        "auth\0service",       // Embedded null byte
        "auth\u{FEFF}service", // Zero-width no-break space (BOM)
    ];

    for svc in sneaky_services {
        let req = make_request("/api/v1/system/status", "GET", json!({"service": svc}));
        assert_eq!(
            engine.evaluate(&claims, &req),
            Err(DenyRule::ConstraintFailed),
            "Service with invisible or control character '{}' must fail regex",
            svc.escape_debug()
        );
    }
}

#[test]
fn test_unicode_character_count_vs_byte_length() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    // /api/v1/audit/search allows ["auditor", "admin"]
    let claims = make_claims("auditor-1", "auditor");

    // Endpoint: /api/v1/audit/search has constraint query: max_len = 100
    // Test that multi-byte UTF-8 characters count as 1 character per code point, not bytes.
    // 🦀 is 4 bytes. 100 emojis = 400 bytes, but 100 characters.
    let hundred_emojis = "🦀".repeat(100);
    let req_valid = make_request(
        "/api/v1/audit/search",
        "POST",
        json!({"query": hundred_emojis}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_valid),
        Ok(()),
        "100 emojis (400 bytes) should satisfy max_len = 100 characters"
    );

    // 101 emojis = 101 characters -> must fail
    let hundred_one_emojis = "🦀".repeat(101);
    let req_invalid = make_request(
        "/api/v1/audit/search",
        "POST",
        json!({"query": hundred_one_emojis}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_invalid),
        Err(DenyRule::ConstraintFailed),
        "101 emojis (101 characters) must violate max_len = 100"
    );
}

#[test]
fn test_type_confusion_in_payload_arguments() {
    let engine = PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap();
    let claims = make_claims("agent-1", "agent");

    // Passing numbers, booleans, objects, arrays where a constrained string is expected
    let type_confusion_payloads = vec![
        json!({"service": 12345}),
        json!({"service": 42.5}),
        json!({"service": true}),
        json!({"service": false}),
        json!({"service": ["auth-service"]}),
        json!({"service": {"name": "auth-service"}}),
    ];

    for payload in type_confusion_payloads {
        let req = make_request("/api/v1/system/status", "GET", payload.clone());
        let res = engine.evaluate(&claims, &req);
        assert_eq!(
            res,
            Err(DenyRule::ConstraintFailed),
            "Non-string payload '{:?}' for string constraint must fail with ConstraintFailed",
            payload
        );
    }
}

// ==============================================================================
// 3. AUDIT REDACTION: EMBED ACTUAL SECRETS & VERIFY RAW EXCLUSION
// ==============================================================================

#[test]
fn test_audit_redaction_anthropic_api_keys_in_all_fields() {
    let (logger, buffer) = AuditLogger::capture();

    // Actual Anthropic API key format: sk-ant-api03-...
    let raw_key_1 = "sk-ant-api03-abcdef1234567890abcdef1234567890abcdef12";
    let raw_key_2 = "sk-ant-admin-9876543210fedcba9876543210fedcba11223344";

    let entry = AuditEntry {
        timestamp: Utc::now(),
        route: "POST /v1/messages".to_string(),
        session_id: Some(format!("sess-{}", raw_key_1)),
        caller_sub: Some(format!("user-{}", raw_key_2)),
        caller_role: Some("agent".to_string()),
        target_endpoint: Some(format!("/v1/messages?key={}", raw_key_1)),
        method: Some("POST".to_string()),
        decision: "deny".to_string(),
        matched_rule: Some(DenyRule::ModelRiskHigh),
        reason: Some(format!(
            "Prompt injection detected while invoking with key {}",
            raw_key_1
        )),
        injection_confidence: Some(0.95),
        malicious_intent: Some(0.85),
        latency_ms: 42,
    };

    logger.log(entry.clone()).unwrap();

    let captured = buffer.lock().unwrap();
    assert_eq!(captured.len(), 1);
    let logged = &captured[0];

    // Verify sanitized entry fields do NOT contain raw keys
    assert!(!logged.session_id.as_ref().unwrap().contains(raw_key_1));
    assert!(!logged.caller_sub.as_ref().unwrap().contains(raw_key_2));
    assert!(!logged.target_endpoint.as_ref().unwrap().contains(raw_key_1));
    assert!(!logged.reason.as_ref().unwrap().contains(raw_key_1));

    // Verify replacement token
    assert!(logged
        .session_id
        .as_ref()
        .unwrap()
        .contains("[REDACTED_API_KEY]"));
    assert!(logged
        .caller_sub
        .as_ref()
        .unwrap()
        .contains("[REDACTED_API_KEY]"));
    assert!(logged
        .target_endpoint
        .as_ref()
        .unwrap()
        .contains("[REDACTED_API_KEY]"));
    assert!(logged
        .reason
        .as_ref()
        .unwrap()
        .contains("[REDACTED_API_KEY]"));

    // Verify serialized JSONL does not contain raw key anywhere
    let jsonl = logger.format_jsonl(logged).unwrap();
    assert!(!jsonl.contains(raw_key_1));
    assert!(!jsonl.contains(raw_key_2));
    assert!(jsonl.ends_with('\n'));
}

#[test]
fn test_audit_redaction_hs256_jwt_tokens_bearer_and_naked() {
    let (logger, buffer) = AuditLogger::capture();

    // Valid format HS256 JWT tokens
    let raw_jwt_1 = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
    let raw_bearer = format!("Bearer {}", raw_jwt_1);
    let raw_jwt_2 = "eyJhbGciOiJIUzI1NiJ9.eyJyb2xlIjoiYWRtaW4iLCJleHAiOjE4MDAwMDAwMDB9.5zZ_L7W0xXf4sP7R2a1b3c4d5e6f7g8h9i0j1k2l3m4";

    let entry = AuditEntry {
        timestamp: Utc::now(),
        route: "POST /api/execute".to_string(),
        session_id: Some("sess-002".to_string()),
        caller_sub: Some("attacker".to_string()),
        caller_role: Some("guest".to_string()),
        target_endpoint: Some("/api/v1/email/send".to_string()),
        method: Some("POST".to_string()),
        decision: "deny".to_string(),
        matched_rule: Some(DenyRule::UnauthorizedToken),
        reason: Some(format!(
            "Failed authentication header: {} and naked token was {}",
            raw_bearer, raw_jwt_2
        )),
        injection_confidence: None,
        malicious_intent: None,
        latency_ms: 2,
    };

    logger.log(entry).unwrap();

    let captured = buffer.lock().unwrap();
    let logged = &captured[0];
    let reason = logged.reason.as_ref().unwrap();

    // Raw JWT strings must be absent
    assert!(!reason.contains(raw_jwt_1));
    assert!(!reason.contains(raw_jwt_2));
    assert!(!reason.contains(&raw_bearer));

    // Should contain redaction indicators
    assert!(reason.contains("Bearer [REDACTED]"));
    assert!(reason.contains("[REDACTED_JWT]"));

    let jsonl = logger.format_jsonl(logged).unwrap();
    assert!(!jsonl.contains(raw_jwt_1));
    assert!(!jsonl.contains(raw_jwt_2));
}

#[test]
fn test_audit_redaction_64_char_hex_strings() {
    let (logger, buffer) = AuditLogger::capture();

    // 64-character hex strings (HMAC secrets, SHA256 hashes, private keys)
    let hex_lower = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
    let hex_upper = "E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855";
    let hex_mixed = "4a5f6b7c8d9e0f1A2B3C4D5E6F7A8B9C0D1E2F3A4B5C6D7E8F9A0B1C2D3E4F5A";

    let entry = AuditEntry {
        timestamp: Utc::now(),
        route: "POST /api/execute".to_string(),
        session_id: Some("sess-003".to_string()),
        caller_sub: Some(format!("sub-{}", hex_lower)),
        caller_role: Some("agent".to_string()),
        target_endpoint: Some("/api/v1/system/status".to_string()),
        method: Some("POST".to_string()),
        decision: "deny".to_string(),
        matched_rule: Some(DenyRule::BackendError),
        reason: Some(format!(
            "Mismatch between expected {} and computed {}",
            hex_upper, hex_mixed
        )),
        injection_confidence: None,
        malicious_intent: None,
        latency_ms: 10,
    };

    logger.log(entry).unwrap();

    let captured = buffer.lock().unwrap();
    let logged = &captured[0];

    assert!(!logged.caller_sub.as_ref().unwrap().contains(hex_lower));
    assert!(!logged.reason.as_ref().unwrap().contains(hex_upper));
    assert!(!logged.reason.as_ref().unwrap().contains(hex_mixed));

    assert!(logged
        .caller_sub
        .as_ref()
        .unwrap()
        .contains("[REDACTED_HEX]"));
    assert!(logged.reason.as_ref().unwrap().contains("[REDACTED_HEX]"));

    let jsonl = logger.format_jsonl(logged).unwrap();
    assert!(!jsonl.contains(hex_lower));
    assert!(!jsonl.contains(hex_upper));
    assert!(!jsonl.contains(hex_mixed));

    // Test with various delimiters: slash, colon, equals, brackets, quotes
    let delimiters = vec![
        format!("hash={}", hex_lower),
        format!("hash:{}", hex_lower),
        format!("hash/{}", hex_lower),
        format!("\"{}\"", hex_lower),
        format!("[{}]", hex_lower),
        format!("({})", hex_lower),
    ];
    for delimited in delimiters {
        let scrubbed = redact_sensitive_text(&delimited);
        assert!(
            !scrubbed.contains(hex_lower),
            "Hex secret delimited as '{}' was not redacted: got '{}'",
            delimited,
            scrubbed
        );
    }
}

#[test]
fn test_audit_redaction_registered_config_and_custom_secrets() {
    let custom_secret = "super-secret-production-signing-token-999";
    let sidecar_secret = "my-private-sidecar-shared-key-12345";
    let jwt_secret = "confidential-jwt-symmetric-secret-key-67890";

    let logger = AuditLogger::buffered()
        .with_secret(custom_secret)
        .with_secret(sidecar_secret)
        .with_secret(jwt_secret);

    let entry = AuditEntry {
        timestamp: Utc::now(),
        route: "POST /api/execute".to_string(),
        session_id: Some(format!("sess-{}", custom_secret)),
        caller_sub: Some(format!("user-{}", sidecar_secret)),
        caller_role: Some("admin".to_string()),
        target_endpoint: Some(format!("/api/v1/status?token={}", jwt_secret)),
        method: Some("POST".to_string()),
        decision: "allow".to_string(),
        matched_rule: None,
        reason: Some(format!("Processed with sidecar={}", sidecar_secret)),
        injection_confidence: Some(0.01),
        malicious_intent: Some(0.01),
        latency_ms: 5,
    };

    logger.log(entry).unwrap();
    let entries = logger.entries();
    let logged = &entries[0];

    assert!(!logged.session_id.as_ref().unwrap().contains(custom_secret));
    assert!(!logged.caller_sub.as_ref().unwrap().contains(sidecar_secret));
    assert!(!logged
        .target_endpoint
        .as_ref()
        .unwrap()
        .contains(jwt_secret));
    assert!(!logged.reason.as_ref().unwrap().contains(sidecar_secret));

    let jsonl = logger.format_jsonl(logged).unwrap();
    assert!(!jsonl.contains(custom_secret));
    assert!(!jsonl.contains(sidecar_secret));
    assert!(!jsonl.contains(jwt_secret));
}

#[test]
fn test_audit_json_value_deep_redaction_comprehensive() {
    let mut payload = json!({
        "username": "alice",
        "password": "plaintext-super-secret-password-123",
        "api_key": "sk-ant-api03-secret-key-value",
        "token": "sensitive-bearer-token",
        "credentials": {
            "private_key": "-----BEGIN RSA PRIVATE KEY-----...",
            "secret_token": "deeply-nested-secret",
            "safe_counter": 42
        },
        "nested_safe": {
            "token_count": 100, // contains "token"!
            "safe_field": "visible_data"
        },
        "auth": "basic dXNlcjpwYXNz",
        "authorization": "Bearer xyz123",
        "items": [
            {"apikey": "nested-in-array-key"},
            {"public_info": "unclassified"}
        ]
    });

    redact_json_value(&mut payload);

    assert_eq!(payload["username"], "alice");
    assert_eq!(payload["password"], "[REDACTED]");
    assert_eq!(payload["api_key"], "[REDACTED]");
    assert_eq!(payload["token"], "[REDACTED]");
    // "credentials" key contains "credential" so the entire object is redacted
    assert_eq!(payload["credentials"], "[REDACTED]");
    // "nested_safe" was traversed: "token_count" contained "token" so it's redacted, but "safe_field" remains
    assert_eq!(payload["nested_safe"]["token_count"], "[REDACTED]");
    assert_eq!(payload["nested_safe"]["safe_field"], "visible_data");
    assert_eq!(payload["auth"], "[REDACTED]");
    assert_eq!(payload["authorization"], "[REDACTED]");
    assert_eq!(payload["items"][0]["apikey"], "[REDACTED]");
    assert_eq!(payload["items"][1]["public_info"], "unclassified");
}

// ==============================================================================
// 4. CONCURRENT MULTI-THREADED STRESS TESTING
// ==============================================================================

#[test]
fn test_concurrent_policy_evaluation_and_audit_logging_stress() {
    let engine = Arc::new(PolicyEngine::from_str(ADVERSARIAL_POLICY_TOML).unwrap());
    let logger = Arc::new(AuditLogger::buffered().with_secret("concurrent-secret-key"));

    let num_threads = 16;
    let iterations_per_thread = 50;
    let mut handles = Vec::with_capacity(num_threads);

    for thread_id in 0..num_threads {
        let eng = Arc::clone(&engine);
        let log = Arc::clone(&logger);

        let handle = std::thread::spawn(move || {
            for i in 0..iterations_per_thread {
                // Alternately issue valid and invalid requests
                let (role, expected_ok) = if i % 2 == 0 {
                    ("agent", true)
                } else {
                    ("unauthorized", false)
                };

                let claims = make_claims(&format!("thread-{}-user-{}", thread_id, i), role);
                let req = make_request(
                    "/api/v1/system/status",
                    "GET",
                    json!({"service": "auth-service"}),
                );

                let eval_res = eng.evaluate(&claims, &req);
                if expected_ok {
                    assert_eq!(eval_res, Ok(()));
                } else {
                    assert_eq!(eval_res, Err(DenyRule::UnauthorizedRole));
                }

                // Log audit entry with embedded secret
                let entry = AuditEntry {
                    timestamp: Utc::now(),
                    route: "POST /api/execute".to_string(),
                    session_id: Some(format!("sess-{}-{}", thread_id, i)),
                    caller_sub: Some(format!("sub-{}-concurrent-secret-key", thread_id)),
                    caller_role: Some(role.to_string()),
                    target_endpoint: Some("/api/v1/system/status".to_string()),
                    method: Some("GET".to_string()),
                    decision: if expected_ok { "allow" } else { "deny" }.to_string(),
                    matched_rule: if expected_ok {
                        None
                    } else {
                        Some(DenyRule::UnauthorizedRole)
                    },
                    reason: if expected_ok {
                        None
                    } else {
                        Some("Role mismatch".to_string())
                    },
                    injection_confidence: None,
                    malicious_intent: None,
                    latency_ms: (thread_id + i) as u64,
                };

                log.log(entry).unwrap();
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().unwrap();
    }

    let captured = logger.entries();
    assert_eq!(
        captured.len(),
        num_threads * iterations_per_thread,
        "All concurrent audit logs must be reliably captured without loss or deadlock"
    );

    // Verify all captured entries were redacted properly
    for entry in &captured {
        let sub = entry.caller_sub.as_ref().unwrap();
        assert!(
            !sub.contains("concurrent-secret-key"),
            "Secret must be redacted even under heavy concurrency"
        );
        assert!(sub.contains("[REDACTED]"));
    }
}
