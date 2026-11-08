//! Empirical Adversarial Stress & Verification Harness for Milestone 2 (Challenger 2)
//!
//! Systematically challenges:
//! 1. Concurrency & Thread-safety: 50+ (64) concurrent threads executing `policy.evaluate`
//!    and `audit.log` simultaneously with zero panics, zero deadlocks, and recovery from lock poisoning.
//! 2. Buffer bounds & memory: high volume audit log buffer (50,000 entries), buffer clear & re-use,
//!    large payload diagnostics (250KB strings), hundreds of registered secrets.
//! 3. String Length Limits: exact boundary conditions (`max_len`, `max_len + 1`, `max_len - 1`),
//!    `max_len = 0`, and Unicode scalar value (char) counts vs UTF-8 byte counts.
//! 4. Null Handling Matrix: JSON `Value::Null` vs missing keys vs empty strings across
//!    required, optional-constrained, and optional-unconstrained arguments.
//! 5. Decision Priority Hierarchy: exhaustive permutation testing of steps 1 through 5.
//! 6. ReDoS Immunity: pathological regular expression patterns tested under adversarial inputs.
//! 7. Multi-layer Secret Redaction: edge cases with regex metacharacters, overlapping secrets,
//!    varied casing, and deep JSON hierarchy redaction.

use serde_json::json;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{Duration, Instant};

use zero_trust_agentic_sidecar::audit::{redact_json_value, redact_sensitive_text, AuditLogger};
use zero_trust_agentic_sidecar::policy::PolicyEngine;
use zero_trust_agentic_sidecar::types::{AgentClaims, AuditEntry, DenyRule, ToolExecuteRequest};

// ==============================================================================
// Test Fixtures and Helpers
// ==============================================================================

fn sample_policy_toml() -> &'static str {
    r#"
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
    allowed_roles = ["agent", "admin", "readonly"]
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
    allowed_roles = []
    required_args = ["table"]

    [endpoints."/api/v1/optional/test"]
    backend_url = "http://127.0.0.1:5000/api/v1/optional/test"
    enabled = true
    allowed_methods = ["POST"]
    allowed_roles = ["agent"]
    required_args = ["mandatory"]
    constraints.mandatory = { max_len = 20 }
    constraints.opt_tag = { regex = "^[a-z]+$", max_len = 10 }
    constraints.opt_zero = { max_len = 0 }
    "#
}

fn create_engine() -> PolicyEngine {
    PolicyEngine::from_str(sample_policy_toml()).expect("Valid policy TOML")
}

fn make_claims(role: &str, sub: &str) -> AgentClaims {
    AgentClaims {
        sub: sub.to_string(),
        role: role.to_string(),
        exp: 2000000000,
        payload_sha256: None,
    }
}

fn make_req(endpoint: &str, method: &str, payload: serde_json::Value) -> ToolExecuteRequest {
    ToolExecuteRequest {
        session_id: "test-sess-001".to_string(),
        endpoint: endpoint.to_string(),
        method: method.to_string(),
        payload: payload.as_object().cloned().unwrap_or_default(),
        context: Default::default(),
    }
}

fn make_audit_entry(
    route: &str,
    decision: &str,
    rule: Option<DenyRule>,
    reason: Option<&str>,
) -> AuditEntry {
    AuditEntry {
        timestamp: chrono::Utc::now(),
        route: route.to_string(),
        session_id: Some("sess-adversarial-test".to_string()),
        caller_sub: Some("agent-challenger-2".to_string()),
        caller_role: Some("agent".to_string()),
        target_endpoint: Some("/api/v1/system/status".to_string()),
        method: Some("POST".to_string()),
        decision: decision.to_string(),
        matched_rule: rule,
        reason: reason.map(|s| s.to_string()),
        injection_confidence: Some(0.05),
        malicious_intent: Some(0.02),
        latency_ms: 12,
    }
}

// ==============================================================================
// 1. Concurrency & Thread-Safety (50+ Concurrent Threads)
// ==============================================================================

#[test]
fn test_stress_64_concurrent_threads_evaluating_and_logging() {
    let engine = Arc::new(create_engine());
    let (logger, buffer) = AuditLogger::capture();
    let logger = Arc::new(logger.with_secret("secret-jwt-signing-key-12345"));

    const THREAD_COUNT: usize = 64;
    const OPS_PER_THREAD: usize = 100;

    let barrier = Arc::new(Barrier::new(THREAD_COUNT));
    let mut handles = Vec::with_capacity(THREAD_COUNT);
    let error_counter = Arc::new(AtomicUsize::new(0));

    for thread_id in 0..THREAD_COUNT {
        let eng = Arc::clone(&engine);
        let log = Arc::clone(&logger);
        let bar = Arc::clone(&barrier);
        let errs = Arc::clone(&error_counter);

        let handle = std::thread::spawn(move || {
            // Synchronize starting line across all 64 threads
            bar.wait();

            for op in 0..OPS_PER_THREAD {
                // Interleave policy evaluation and audit logging
                let role = if (thread_id + op) % 3 == 0 {
                    "admin"
                } else if (thread_id + op) % 3 == 1 {
                    "agent"
                } else {
                    "unauthorized"
                };
                let claims = make_claims(role, &format!("worker-{}-{}", thread_id, op));

                let req = if op % 4 == 0 {
                    // Valid status request
                    make_req(
                        "/api/v1/system/status",
                        "GET",
                        json!({"service": "database"}),
                    )
                } else if op % 4 == 1 {
                    // Valid email request (if role allows)
                    make_req(
                        "/api/v1/email/send",
                        "POST",
                        json!({
                            "to": "dev@ourcompany.com",
                            "body": "Daily check"
                        }),
                    )
                } else if op % 4 == 2 {
                    // Constraint failure: invalid email domain
                    make_req(
                        "/api/v1/email/send",
                        "POST",
                        json!({
                            "to": "evil@attacker.com",
                            "body": "Hello"
                        }),
                    )
                } else {
                    // Disabled endpoint
                    make_req("/api/v1/db/drop", "POST", json!({"table": "users"}))
                };

                let eval_result = eng.evaluate(&claims, &req);

                // Audit log based on result
                let entry = match eval_result {
                    Ok(()) => make_audit_entry("POST /api/execute", "allow", None, None),
                    Err(rule) => make_audit_entry(
                        "POST /api/execute",
                        "deny",
                        Some(rule),
                        Some("Evaluated denial by rule"),
                    ),
                };

                if let Err(e) = log.log(entry) {
                    eprintln!("Thread {} logging error: {:?}", thread_id, e);
                    errs.fetch_add(1, Ordering::SeqCst);
                }
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("Thread should not panic or deadlock");
    }

    assert_eq!(
        error_counter.load(Ordering::SeqCst),
        0,
        "Zero errors expected under 64-thread concurrent load"
    );

    let logged_count = buffer.lock().unwrap().len();
    assert_eq!(
        logged_count,
        THREAD_COUNT * OPS_PER_THREAD,
        "All 6,400 audit entries must be safely captured in memory"
    );
}

#[test]
fn test_concurrency_race_buffer_clear_read_write() {
    let (logger, buffer) = AuditLogger::capture();
    let logger = Arc::new(logger);
    let stop_signal = Arc::new(AtomicBool::new(false));

    let mut handles = Vec::new();

    // 16 Writer threads
    for t in 0..16 {
        let log = Arc::clone(&logger);
        let stop = Arc::clone(&stop_signal);
        handles.push(std::thread::spawn(move || {
            let mut i = 0;
            while !stop.load(Ordering::Relaxed) {
                let entry = make_audit_entry(
                    "POST /api/execute",
                    "allow",
                    None,
                    Some(&format!("Writer {} op {}", t, i)),
                );
                log.log(entry).unwrap();
                i += 1;
            }
        }));
    }

    // 8 Reader threads
    for _ in 0..8 {
        let log = Arc::clone(&logger);
        let stop = Arc::clone(&stop_signal);
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                let _entries = log.entries();
                let _captured = log.captured_entries();
            }
        }));
    }

    // 4 Clearer threads
    for _ in 0..4 {
        let log = Arc::clone(&logger);
        let stop = Arc::clone(&stop_signal);
        handles.push(std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                log.clear();
                std::thread::yield_now();
            }
        }));
    }

    // Let the race run for 100ms
    std::thread::sleep(Duration::from_millis(100));
    stop_signal.store(true, Ordering::SeqCst);

    for h in handles {
        h.join().expect("Racy threads must not panic or deadlock");
    }

    // Clean up
    logger.clear();
    assert_eq!(buffer.lock().unwrap().len(), 0);
}

#[test]
fn test_concurrency_with_poisoned_mutex_resilience() {
    let (logger, buffer) = AuditLogger::capture();
    let logger = Arc::new(logger);

    // Deliberately poison the mutex in a sacrificial thread
    let buf_poisoner = Arc::clone(&buffer);
    let sacrificial = std::thread::spawn(move || {
        let _guard = buf_poisoner.lock().unwrap();
        panic!("Sacrificial thread panics to poison the buffer Mutex!");
    });
    let _ = sacrificial.join();

    assert!(buffer.is_poisoned(), "Mutex must be poisoned");

    // Spawn 20 concurrent threads that must survive and log despite poisoned mutex
    let mut handles = Vec::new();
    for t in 0..20 {
        let log = Arc::clone(&logger);
        handles.push(std::thread::spawn(move || {
            for i in 0..10 {
                let entry = make_audit_entry(
                    "POST /api/execute",
                    "deny",
                    Some(DenyRule::UnauthorizedRole),
                    Some(&format!("Poisoned thread {} step {}", t, i)),
                );
                // Must not panic on poisoned lock
                let res = log.log(entry);
                assert!(res.is_ok(), "AuditLogger must handle poisoned lock cleanly");
            }
        }));
    }

    for h in handles {
        h.join().expect("Thread should not panic");
    }

    let entries = logger.entries();
    assert_eq!(
        entries.len(),
        200,
        "All 200 entries logged across poisoned mutex threads"
    );
}

// ==============================================================================
// 2. Buffer Bounds & Memory Volume
// ==============================================================================

#[test]
fn test_high_volume_audit_buffer_capacity() {
    let (logger, buffer) = AuditLogger::capture();

    const VOLUME: usize = 50_000;
    let start = Instant::now();

    for i in 0..VOLUME {
        let entry = make_audit_entry(
            "POST /api/execute",
            "allow",
            None,
            Some("High volume audit memory test entry"),
        );
        logger.log(entry).unwrap();

        // Periodically verify buffer len
        if (i + 1) % 10_000 == 0 {
            let count = buffer.lock().unwrap().len();
            assert_eq!(count, i + 1);
        }
    }

    let elapsed = start.elapsed();
    assert_eq!(logger.entries().len(), VOLUME);

    // Verify clear() properly empties and resets buffer
    logger.clear();
    assert_eq!(buffer.lock().unwrap().len(), 0);

    // Re-verify buffer can continue to log after clear
    let new_entry = make_audit_entry("POST /api/execute", "allow", None, Some("Post-clear entry"));
    logger.log(new_entry).unwrap();
    assert_eq!(logger.entries().len(), 1);

    println!("High volume test: 50,000 entries logged in {:.2?}", elapsed);
}

#[test]
fn test_extreme_diagnostic_string_lengths_in_audit_entries() {
    let logger = AuditLogger::buffered()
        .with_secret("confidential-secret-alpha")
        .with_secret("confidential-secret-beta");

    // Construct a massive 250,000 character reason string embedding secrets
    let mut huge_reason = "normal diagnostic prefix; ".repeat(5_000); // ~130KB
    huge_reason.push_str(" embedded: confidential-secret-alpha; ");
    huge_reason.push_str(&"sk-ant-api03-1234567890abcdef1234567890abcdef ".repeat(100)); // repeated API keys
    huge_reason.push_str(" embedded: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature; ");
    huge_reason.push_str(&"suffix diagnostic information; ".repeat(3_000)); // ~93KB

    let mut entry = make_audit_entry(
        "POST /api/execute",
        "deny",
        Some(DenyRule::ConstraintFailed),
        Some(&huge_reason),
    );
    entry.session_id = Some(format!("sess-huge-{}", "a".repeat(10_000)));
    entry.target_endpoint = Some(format!("/api/v1/test/{}", "endpoint-segment-".repeat(500)));

    let start = Instant::now();
    let log_res = logger.log(entry);
    let elapsed = start.elapsed();

    assert!(log_res.is_ok(), "Huge audit entry must not cause error");
    let entries = logger.entries();
    assert_eq!(entries.len(), 1);

    let sanitized_reason = entries[0].reason.as_deref().unwrap();
    assert!(
        !sanitized_reason.contains("confidential-secret-alpha"),
        "Secret must be scrubbed"
    );
    assert!(
        !sanitized_reason.contains("sk-ant-api03-1234567890abcdef1234567890abcdef"),
        "API keys must be scrubbed"
    );
    assert!(
        !sanitized_reason.contains("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature"),
        "Bearer tokens must be scrubbed"
    );

    // Format to JSONL to verify serde doesn't choke on huge strings
    let jsonl = logger.format_jsonl(&entries[0]).unwrap();
    assert!(jsonl.ends_with('\n'));
    assert!(jsonl.len() > 200_000);

    println!("Sanitized and serialized 250KB entry in {:.2?}", elapsed);
}

#[test]
fn test_many_registered_secrets_redaction() {
    let mut logger = AuditLogger::buffered();

    // Register 200 distinct secrets
    for i in 0..200 {
        logger.add_secret(format!("secret-token-key-{:04}", i));
    }

    // Build diagnostic string containing every 10th secret
    let mut diagnostic = String::from("Errors occurred: ");
    for i in (0..200).step_by(10) {
        diagnostic.push_str(&format!("Found secret-token-key-{:04}; ", i));
    }

    let entry = make_audit_entry(
        "POST /api/execute",
        "deny",
        Some(DenyRule::BackendError),
        Some(&diagnostic),
    );
    logger.log(entry).unwrap();

    let entries = logger.entries();
    let sanitized = entries[0].reason.as_deref().unwrap();

    for i in (0..200).step_by(10) {
        assert!(
            !sanitized.contains(&format!("secret-token-key-{:04}", i)),
            "Secret {} must be redacted",
            i
        );
    }
    assert!(sanitized.contains("[REDACTED]"));
}

// ==============================================================================
// 3. String Length Limits (Exact Boundary Conditions: max_len, max_len + 1)
// ==============================================================================

#[test]
fn test_string_length_exact_boundary_conditions() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint: /api/v1/system/status
    // Constraint: service regex = "^[a-z0-9-]+$", max_len = 64

    // 1. Exact boundary: max_len - 1 = 63 chars -> PASS
    let service_63 = "a".repeat(63);
    let req_63 = make_req(
        "/api/v1/system/status",
        "GET",
        json!({"service": service_63}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_63),
        Ok(()),
        "63 chars must PASS (max_len = 64)"
    );

    // 2. Exact boundary: max_len = 64 chars -> PASS
    let service_64 = "a".repeat(64);
    let req_64 = make_req(
        "/api/v1/system/status",
        "GET",
        json!({"service": service_64}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_64),
        Ok(()),
        "Exact 64 chars must PASS (max_len = 64)"
    );

    // 3. Exact boundary: max_len + 1 = 65 chars -> FAIL (DenyRule::ConstraintFailed)
    let service_65 = "a".repeat(65);
    let req_65 = make_req(
        "/api/v1/system/status",
        "GET",
        json!({"service": service_65}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_65),
        Err(DenyRule::ConstraintFailed),
        "65 chars must FAIL (max_len = 64)"
    );

    // Endpoint: /api/v1/email/send
    // Constraint: body max_len = 2000 (no regex)

    // 4. Exact boundary: body max_len - 1 = 1999 chars -> PASS
    let body_1999 = "x".repeat(1999);
    let req_body_1999 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({"to": "alice@ourcompany.com", "body": body_1999}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_body_1999),
        Ok(()),
        "1999 chars must PASS (max_len = 2000)"
    );

    // 5. Exact boundary: body max_len = 2000 chars -> PASS
    let body_2000 = "x".repeat(2000);
    let req_body_2000 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({"to": "alice@ourcompany.com", "body": body_2000}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_body_2000),
        Ok(()),
        "2000 chars must PASS (max_len = 2000)"
    );

    // 6. Exact boundary: body max_len + 1 = 2001 chars -> FAIL (DenyRule::ConstraintFailed)
    let body_2001 = "x".repeat(2001);
    let req_body_2001 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({"to": "alice@ourcompany.com", "body": body_2001}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_body_2001),
        Err(DenyRule::ConstraintFailed),
        "2001 chars must FAIL (max_len = 2000)"
    );
}

#[test]
fn test_string_length_exact_boundary_zero() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint: /api/v1/optional/test
    // Constraint: opt_zero max_len = 0

    // 1. Length 0 ("") -> PASS (0 <= 0)
    let req_zero_empty = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "ok", "opt_zero": ""}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_zero_empty),
        Ok(()),
        "Length 0 string must pass max_len = 0"
    );

    // 2. Length 1 ("a") -> FAIL (1 > 0)
    let req_zero_one = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "ok", "opt_zero": "a"}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_zero_one),
        Err(DenyRule::ConstraintFailed),
        "Length 1 string must fail max_len = 0"
    );
}

#[test]
fn test_string_length_unicode_chars_vs_bytes() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint: /api/v1/optional/test
    // Constraint: mandatory max_len = 20 (no regex)

    // 1. Exactly 20 Unicode multibyte emojis (each emoji is 4 UTF-8 bytes -> total 80 bytes, 20 chars)
    let emoji_20 = "🦀".repeat(20);
    assert_eq!(emoji_20.len(), 80); // 80 bytes
    assert_eq!(emoji_20.chars().count(), 20); // 20 chars

    let req_emoji_20 = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": emoji_20}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_emoji_20),
        Ok(()),
        "20 emojis (80 bytes) must PASS when max_len = 20 chars"
    );

    // 2. Exactly 21 Unicode multibyte emojis (84 bytes, 21 chars) -> FAIL
    let emoji_21 = "🦀".repeat(21);
    assert_eq!(emoji_21.chars().count(), 21);

    let req_emoji_21 = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": emoji_21}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_emoji_21),
        Err(DenyRule::ConstraintFailed),
        "21 emojis (21 chars > 20 max_len) must FAIL"
    );

    // 3. Mixed multibyte characters: Japanese kanji + Cyrillic + ASCII
    // "東京Москва12" = 2 + 6 + 2 = 10 chars, 20 bytes
    let mixed_10 = "東京Москва12";
    assert_eq!(mixed_10.chars().count(), 10);
    let req_mixed_10 = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": mixed_10}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_mixed_10),
        Ok(()),
        "10 multibyte chars must pass max_len = 20"
    );
}

// ==============================================================================
// 4. Null Handling Matrix: JSON Value::Null vs Missing Keys vs Empty Strings
// ==============================================================================

#[test]
fn test_null_handling_required_arguments() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint /api/v1/system/status requires "service"

    // 1. Missing key -> DenyRule::MissingArg
    let req_missing = make_req("/api/v1/system/status", "GET", json!({}));
    assert_eq!(
        engine.evaluate(&claims, &req_missing),
        Err(DenyRule::MissingArg),
        "Missing required key must return MissingArg"
    );

    // 2. Explicit Value::Null -> DenyRule::MissingArg
    let req_null = make_req("/api/v1/system/status", "GET", json!({"service": null}));
    assert_eq!(
        engine.evaluate(&claims, &req_null),
        Err(DenyRule::MissingArg),
        "Value::Null for required key must return MissingArg"
    );

    // 3. Empty string "" -> Regex fails -> DenyRule::ConstraintFailed
    let req_empty_str = make_req("/api/v1/system/status", "GET", json!({"service": ""}));
    assert_eq!(
        engine.evaluate(&claims, &req_empty_str),
        Err(DenyRule::ConstraintFailed),
        "Empty string for service fails regex ^[a-z0-9-]+$, returning ConstraintFailed"
    );

    // 4. Whitespace string "   " -> Regex fails -> DenyRule::ConstraintFailed
    let req_whitespace = make_req("/api/v1/system/status", "GET", json!({"service": "   "}));
    assert_eq!(
        engine.evaluate(&claims, &req_whitespace),
        Err(DenyRule::ConstraintFailed),
        "Whitespace string fails regex, returning ConstraintFailed"
    );
}

#[test]
fn test_null_handling_optional_constrained_arguments() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint: /api/v1/optional/test
    // Required: ["mandatory"]
    // Constrained: "opt_tag" (regex = "^[a-z]+$", max_len = 10)

    // 1. Missing optional argument -> PASS
    let req_omitted = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "hello"}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_omitted),
        Ok(()),
        "Omitted optional constrained argument must PASS"
    );

    // 2. Explicit Value::Null for optional constrained argument -> DenyRule::ConstraintFailed
    let req_opt_null = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "hello", "opt_tag": null}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_opt_null),
        Err(DenyRule::ConstraintFailed),
        "Explicit Value::Null for constrained argument must fail with ConstraintFailed"
    );

    // 3. Empty string "" for optional constrained argument (regex requires 1+ chars) -> ConstraintFailed
    let req_opt_empty = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "hello", "opt_tag": ""}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_opt_empty),
        Err(DenyRule::ConstraintFailed),
        "Empty string violating regex must fail with ConstraintFailed"
    );

    // 4. Valid non-empty string -> PASS
    let req_opt_valid = make_req(
        "/api/v1/optional/test",
        "POST",
        json!({"mandatory": "hello", "opt_tag": "validtag"}),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_opt_valid),
        Ok(()),
        "Valid optional tag must PASS"
    );
}

#[test]
fn test_null_handling_unconstrained_extra_arguments() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Endpoint /api/v1/system/status has no constraints on "extra_meta"

    // 1. Extra unconstrained key with Value::Null -> PASS
    let req_extra_null = make_req(
        "/api/v1/system/status",
        "GET",
        json!({
            "service": "auth-service",
            "extra_meta": null
        }),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_extra_null),
        Ok(()),
        "Unconstrained key with Value::Null is permitted"
    );

    // 2. Extra unconstrained key with nested object / array -> PASS
    let req_extra_complex = make_req(
        "/api/v1/system/status",
        "GET",
        json!({
            "service": "auth-service",
            "nested": {"key": [1, 2, 3]},
            "empty_str": ""
        }),
    );
    assert_eq!(
        engine.evaluate(&claims, &req_extra_complex),
        Ok(()),
        "Unconstrained complex JSON values are permitted"
    );
}

#[test]
fn test_null_handling_non_string_types_for_constrained_args() {
    let engine = create_engine();
    let claims = make_claims("agent", "agent-007");

    // Numbers, booleans, arrays, objects passed to constrained string arguments
    let bad_values = vec![
        json!(12345),
        json!(42.5),
        json!(true),
        json!(false),
        json!(["database"]),
        json!({"name": "database"}),
    ];

    for val in bad_values {
        let req = make_req("/api/v1/system/status", "GET", json!({"service": val}));
        assert_eq!(
            engine.evaluate(&claims, &req),
            Err(DenyRule::ConstraintFailed),
            "Non-string value for constrained argument must fail with ConstraintFailed"
        );
    }
}

// ==============================================================================
// 5. Decision Priority Hierarchy (Steps 1 to 5)
// ==============================================================================

#[test]
fn test_decision_priority_hierarchy_exhaustive_permutations() {
    let engine = create_engine();

    // Permutation 1: Unlisted endpoint + Bad method + Bad role + Missing args + Bad constraints
    // Expected: EndpointDenied (Step 1)
    let claims_bad = make_claims("attacker", "hacker");
    let req1 = make_req("/api/v1/unlisted/endpoint", "PATCH", json!({}));
    assert_eq!(
        engine.evaluate(&claims_bad, &req1),
        Err(DenyRule::EndpointDenied),
        "Priority 1: Unlisted endpoint must trigger EndpointDenied"
    );

    // Permutation 2: Disabled endpoint + Bad method + Bad role + Missing args
    // Expected: EndpointDenied (Step 1)
    let req2 = make_req("/api/v1/db/drop", "DELETE", json!({}));
    assert_eq!(
        engine.evaluate(&claims_bad, &req2),
        Err(DenyRule::EndpointDenied),
        "Priority 1: Disabled endpoint must trigger EndpointDenied before checking method or role"
    );

    // Permutation 3: Enabled endpoint + Method mismatch + Bad role + Missing args
    // Expected: EndpointDenied (Step 2)
    let req3 = make_req(
        "/api/v1/email/send",
        "GET", // only POST is allowed
        json!({}),
    );
    assert_eq!(
        engine.evaluate(&claims_bad, &req3),
        Err(DenyRule::EndpointDenied),
        "Priority 2: Method mismatch must trigger EndpointDenied before role check"
    );

    // Permutation 4: Enabled endpoint + Method OK + Role mismatch + Missing args + Constraint fails
    // Expected: UnauthorizedRole (Step 3)
    let claims_readonly = make_claims("readonly", "guest"); // email/send allows ["agent", "admin"]
    let req4 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({"to": "invalid_email_regex", "body": null}),
    );
    assert_eq!(
        engine.evaluate(&claims_readonly, &req4),
        Err(DenyRule::UnauthorizedRole),
        "Priority 3: Role mismatch must trigger UnauthorizedRole before required_args check"
    );

    // Permutation 5: Enabled endpoint + Method OK + Role OK + Missing args + Other constraint fails
    // Expected: MissingArg (Step 4)
    let claims_agent = make_claims("agent", "agent-007");
    let req5 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({"to": "invalid_email"}), // missing "body"
    );
    assert_eq!(
        engine.evaluate(&claims_agent, &req5),
        Err(DenyRule::MissingArg),
        "Priority 4: Missing required arg must trigger MissingArg before constraint check"
    );

    // Permutation 6: Enabled endpoint + Method OK + Role OK + Required args present + Constraint fails
    // Expected: ConstraintFailed (Step 5)
    let req6 = make_req(
        "/api/v1/email/send",
        "POST",
        json!({
            "to": "evil@attacker.com", // violates regex
            "body": "Valid body message"
        }),
    );
    assert_eq!(
        engine.evaluate(&claims_agent, &req6),
        Err(DenyRule::ConstraintFailed),
        "Priority 5: Constraint violation must trigger ConstraintFailed"
    );
}

// ==============================================================================
// 6. ReDoS Immunity Under Pathological Regex Input Patterns
// ==============================================================================

#[test]
fn test_policy_linear_time_regex_pathological_inputs() {
    let policy_toml = r#"
    [defaults]
    max_injection_confidence = 0.70
    max_malicious_intent = 0.50

    [endpoints."/api/v1/redos/test"]
    backend_url = "http://localhost:5000/redos"
    enabled = true
    allowed_methods = ["POST"]
    allowed_roles = ["agent"]
    required_args = ["payload"]
    constraints.payload = { regex = "^(a+)+$", max_len = 100000 }
    "#;

    let engine = PolicyEngine::from_str(policy_toml).expect("Regex engine compiles");
    let claims = make_claims("agent", "agent-007");

    // In a backtracking engine (e.g. PCRE / Python re), (a+)+ with 30 'a's + 'b' takes seconds/minutes.
    // Rust's regex crate uses DFA/PikeVM with guaranteed O(M * N) linear time.
    let pathological_input = format!("{}b", "a".repeat(50_000));

    let req = make_req(
        "/api/v1/redos/test",
        "POST",
        json!({"payload": pathological_input}),
    );

    let start = Instant::now();
    let result = engine.evaluate(&claims, &req);
    let elapsed = start.elapsed();

    assert_eq!(
        result,
        Err(DenyRule::ConstraintFailed),
        "Pathological input must be rejected by regex"
    );

    // Must execute in well under 100 milliseconds for 50,000 characters
    assert!(
        elapsed < Duration::from_millis(100),
        "Regex matching must be linear-time: took {:?}",
        elapsed
    );
    println!("50,000-char pathological regex evaluated in: {:?}", elapsed);
}

// ==============================================================================
// 7. Multi-Layer Secret Redaction Edge Cases
// ==============================================================================

#[test]
fn test_audit_redaction_regex_metacharacters_in_known_secret() {
    // Secret containing characters that would break or misbehave if used in a regex
    let metachar_secret = r#"s3cr3t$[with]*(regex)+.^?\d{4}"#;
    let logger = AuditLogger::buffered().with_secret(metachar_secret);

    let mut entry = make_audit_entry(
        "POST /api/execute",
        "deny",
        Some(DenyRule::BackendError),
        Some(&format!("Failed with secret: {}", metachar_secret)),
    );
    entry.session_id = Some(format!("session-{}", metachar_secret));

    logger.log(entry).unwrap();
    let entries = logger.entries();

    let sanitized_reason = entries[0].reason.as_deref().unwrap();
    let sanitized_sess = entries[0].session_id.as_deref().unwrap();

    assert!(
        !sanitized_reason.contains(metachar_secret),
        "Metachar secret must be scrubbed from reason"
    );
    assert!(
        !sanitized_sess.contains(metachar_secret),
        "Metachar secret must be scrubbed from session_id"
    );
    assert!(sanitized_reason.contains("[REDACTED]"));
    assert!(sanitized_sess.contains("[REDACTED]"));
}

#[test]
fn test_audit_redaction_overlapping_and_substring_secrets() {
    let shorter_secret = "secret123";
    let longer_secret = "secret123456";

    // Register both shorter and longer secret
    let logger = AuditLogger::buffered()
        .with_secret(shorter_secret)
        .with_secret(longer_secret);

    let text = format!("Diagnostics: {} and {}", longer_secret, shorter_secret);
    let entry = make_audit_entry("POST /api/execute", "allow", None, Some(&text));

    logger.log(entry).unwrap();
    let entries = logger.entries();
    let sanitized = entries[0].reason.as_deref().unwrap();

    assert!(!sanitized.contains(longer_secret));
    assert!(!sanitized.contains(shorter_secret));
    assert!(sanitized.contains("[REDACTED]"));
}

#[test]
fn test_audit_redaction_bearer_case_insensitivity() {
    let variations = [
        "Bearer eyJhbGciOiJIUzI1NiJ9.e30.signature",
        "bearer eyJhbGciOiJIUzI1NiJ9.e30.signature",
        "BEARER eyJhbGciOiJIUzI1NiJ9.e30.signature",
        "BeArEr eyJhbGciOiJIUzI1NiJ9.e30.signature",
    ];

    for variant in variations {
        let scrubbed = redact_sensitive_text(variant);
        assert!(
            !scrubbed.contains("eyJhbGciOiJIUzI1NiJ9.e30.signature"),
            "Token in '{}' must be redacted",
            variant
        );
        assert!(
            scrubbed.contains("Bearer [REDACTED]") || scrubbed.contains("[REDACTED]"),
            "Must contain redaction token"
        );
    }
}

#[test]
fn test_audit_redact_json_value_deep_hierarchy() {
    // Construct 20-level deeply nested JSON value
    let mut root = json!({
        "safe_outer": "outer_val",
        "api_key": "sk-ant-outer-key-12345"
    });

    let mut current = &mut root;
    for depth in 0..15 {
        let next_key = format!("level_{}", depth);
        if let serde_json::Value::Object(map) = current {
            map.insert(
                next_key.clone(),
                json!({
                    "safe_field": depth,
                    "password": format!("secret_pass_{}", depth),
                    "token": format!("secret_token_{}", depth)
                }),
            );
            current = map.get_mut(&next_key).unwrap();
        }
    }

    redact_json_value(&mut root);

    // Verify outer key
    assert_eq!(root["safe_outer"], "outer_val");
    assert_eq!(root["api_key"], "[REDACTED]");

    // Verify inner keys at depths
    for depth in 0..15 {
        let mut node = &root;
        for d in 0..=depth {
            node = &node[format!("level_{}", d)];
        }
        assert_eq!(node["safe_field"], depth);
        assert_eq!(node["password"], "[REDACTED]");
        assert_eq!(node["token"], "[REDACTED]");
    }
}
