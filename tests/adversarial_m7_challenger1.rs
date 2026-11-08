//! Comprehensive Adversarial Challenge Suite for Milestone 7 (Challenger 1)
//!
//! Empirical Challenger 1 verification:
//! 1. Serde `deny_unknown_fields`: Misspelled/unknown fields at every level of `policy.toml`:
//!    - PolicyConfig root level
//!    - DefaultPolicy (`[defaults]`)
//!    - EndpointRule (`[endpoints."..."]`)
//!    - ToolRule (`[tools."..."]`)
//!    - AgentCliMapping (`[agent_cli_mappings."..."]`)
//!    - ArgConstraint (`constraints.<name>`)
//! 2. Threshold & Timeout Bounds:
//!    - Negative thresholds (< 0.0)
//!    - Thresholds exceeding 1.0 (> 1.0)
//!    - Floating point NaN, +inf, -inf
//!    - Timeouts = 0, timeouts > 60,000, negative timeouts
//!    - Edge boundary cases: 0.0, 1.0, 1ms, 60,000ms
//!    - Verification across defaults, endpoints, and tools
//! 3. Secret Length Validation & Credential Leakage:
//!    - 0, 1, 15, and 16-byte secrets for SIDECAR_SECRET and JWT_SECRET
//!    - Unicode multi-byte characters (byte length vs char count)
//!    - Strict verification that raw secrets NEVER leak in error messages or debug representations
//! 4. URL Scheme & Host Validation:
//!    - Non-HTTP schemes: file://, ftp://, ws://, gopher://, javascript:, data:
//!    - Empty hosts, missing schemes, empty/whitespace strings
//!    - Verification on Config (UPSTREAM_LLM_URL, FLASK_BACKEND_URL, KEV_URL) and Policy (backend_url)
//! 5. Policy Engine M7 Invariants:
//!    - Threshold and timeout override inheritance vs fallback
//!    - Tool evaluation priority order (EndpointDenied -> UnauthorizedRole -> MissingArg -> ConstraintFailed)
//!    - Agent CLI mappings environment variable synthesis & audit key registration

use std::collections::HashMap;
use zero_trust_agentic_sidecar::*;

/// Base minimal valid policy TOML string to graft test variations onto.
const BASE_VALID_POLICY: &str = r#"
[defaults]
max_injection_confidence = 0.70
max_malicious_intent = 0.50
model_timeout_ms = 500
fail_closed = true
require_jwt = true
scope_current_turn_only = true
exclude_system_prompt = true
max_kev_tokens = 8192

[endpoints."/api/v1/system/status"]
backend_url = "http://127.0.0.1:5000/api/v1/system/status"
enabled = true
allowed_methods = ["GET", "POST"]
allowed_roles = ["agent", "admin"]
required_args = ["service"]
constraints.service = { regex = "^[a-z0-9-]+$", max_len = 64 }

[tools."bash"]
enabled = true
allowed_roles = ["admin"]
required_args = ["command"]
constraints.command = { regex = "^[a-zA-Z0-9_./ -]+$", max_len = 512 }
max_malicious_intent = 0.40

[agent_cli_mappings.claude]
env_var = "ANTHROPIC_BASE_URL"
api_key_env_var = "ANTHROPIC_API_KEY"
sidecar_url = "http://127.0.0.1:8080"
upstream_url = "https://api.anthropic.com/v1/messages"
protocol = "anthropic"
extra_env = { CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1" }
"#;

// ==============================================================================
// 1. Serde `deny_unknown_fields` Adversarial Tests
// ==============================================================================

#[test]
fn test_serde_rejects_unknown_field_at_root_level() {
    let toml_str = format!(
        "{}\nunknown_root_option = \"malicious_override\"\n",
        BASE_VALID_POLICY
    );
    let result = PolicyEngine::from_str(&toml_str);
    assert!(result.is_err(), "Expected failure on unknown root field");
    let err_str = result.err().unwrap().to_string();
    assert!(
        err_str.contains("unknown field `unknown_root_option`"),
        "Unexpected error: {err_str}"
    );
}

#[test]
fn test_serde_rejects_unknown_table_at_root_level() {
    let toml_str = format!(
        "{}\n[unrecognized_section]\nfoo = \"bar\"\n",
        BASE_VALID_POLICY
    );
    let result = PolicyEngine::from_str(&toml_str);
    assert!(result.is_err(), "Expected failure on unknown table");
    let err_str = result.err().unwrap().to_string();
    assert!(
        err_str.contains("unknown field `unrecognized_section`"),
        "Unexpected error: {err_str}"
    );
}

#[test]
fn test_serde_rejects_misspelled_section_names() {
    // misspelled 'default' instead of 'defaults'
    let toml_default = BASE_VALID_POLICY.replace("[defaults]", "[default]");
    assert!(PolicyEngine::from_str(&toml_default).is_err());

    // misspelled 'endpoint' instead of 'endpoints'
    let toml_endpoint = BASE_VALID_POLICY.replace("[endpoints.", "[endpoint.");
    assert!(PolicyEngine::from_str(&toml_endpoint).is_err());

    // misspelled 'tool' instead of 'tools'
    let toml_tool = BASE_VALID_POLICY.replace("[tools.", "[tool.");
    assert!(PolicyEngine::from_str(&toml_tool).is_err());

    // misspelled 'cli_mappings' instead of 'agent_cli_mappings'
    let toml_cli = BASE_VALID_POLICY.replace("[agent_cli_mappings.", "[cli_mappings.");
    assert!(PolicyEngine::from_str(&toml_cli).is_err());
}

#[test]
fn test_serde_rejects_unknown_or_misspelled_fields_in_defaults() {
    let test_cases = [
        "max_injection = 0.50",          // misspelled max_injection_confidence
        "max_intent = 0.40",             // misspelled max_malicious_intent
        "timeout = 500",                 // misspelled model_timeout_ms
        "failclosed = true",             // misspelled fail_closed
        "jwt_required = true",           // misspelled require_jwt
        "scope_current_turn = true",     // misspelled scope_current_turn_only
        "exclude_system_prompts = true", // misspelled exclude_system_prompt
        "max_tokens = 8192",             // misspelled max_kev_tokens
        "arbitrary_default = \"hello\"", // arbitrary unknown field
    ];

    for case in test_cases {
        let toml_str = format!("[defaults]\n{case}\n");
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for defaults unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }
}

#[test]
fn test_serde_rejects_unknown_or_misspelled_fields_in_endpoints() {
    let test_cases = [
        "backend = \"http://127.0.0.1:5000\"", // misspelled backend_url
        "enable = true",                       // misspelled enabled
        "methods = [\"POST\"]",                // misspelled allowed_methods
        "roles = [\"admin\"]",                 // misspelled allowed_roles
        "required = [\"service\"]",            // misspelled required_args
        "constraint.service = {}",             // misspelled constraints
        "max_injection = 0.65",                // misspelled max_injection_confidence
        "max_intent = 0.35",                   // misspelled max_malicious_intent
        "timeout = 1000",                      // misspelled model_timeout_ms
        "endpoint_secret = \"secret\"",        // unknown field
    ];

    for case in test_cases {
        let toml_str = format!(
            r#"
[defaults]

[endpoints."/api/v1/test"]
backend_url = "http://127.0.0.1:5000/api/v1/test"
{case}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for endpoint unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }
}

#[test]
fn test_serde_rejects_unknown_or_misspelled_fields_in_tools() {
    let test_cases = [
        "enable = true",            // misspelled enabled
        "roles = [\"admin\"]",      // misspelled allowed_roles
        "required = [\"command\"]", // misspelled required_args
        "constraint.cmd = {}",      // misspelled constraints
        "max_injection = 0.60",     // misspelled max_injection_confidence
        "max_intent = 0.40",        // misspelled max_malicious_intent
        "timeout = 2000",           // misspelled model_timeout_ms
        "bypass_policy = true",     // unknown field
    ];

    for case in test_cases {
        let toml_str = format!(
            r#"
[defaults]

[tools."bash"]
{case}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for tool unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }
}

#[test]
fn test_serde_rejects_unknown_or_misspelled_fields_in_agent_cli_mappings() {
    let test_cases = [
        "environment_variable = \"VAR\"",  // misspelled env_var
        "api_key_env = \"KEY\"",           // misspelled api_key_env_var
        "url = \"http://127.0.0.1:8080\"", // misspelled sidecar_url
        "upstream = \"https://api.com\"",  // misspelled upstream_url
        "proto = \"anthropic\"",           // misspelled protocol
        "extras = { FOO = \"BAR\" }",      // misspelled extra_env
        "unknown_cli_key = 123",           // unknown field
    ];

    for case in test_cases {
        let toml_str = format!(
            r#"
[defaults]

[agent_cli_mappings.claude]
env_var = "ANTHROPIC_BASE_URL"
sidecar_url = "http://127.0.0.1:8080"
{case}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for agent_cli_mapping unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }
}

#[test]
fn test_serde_rejects_unknown_or_misspelled_fields_in_constraints() {
    // In endpoints:
    let endpoint_cases = [
        "constraints.service = { pattern = \"^[a-z]+$\" }", // misspelled regex
        "constraints.service = { max_length = 64 }",        // misspelled max_len
        "constraints.service = { allow_null = false }",     // unknown constraint
        "constraints.service = { regex = \".*\", max_len = 10, evil_flag = true }",
    ];

    for case in endpoint_cases {
        let toml_str = format!(
            r#"
[defaults]

[endpoints."/api/v1/test"]
backend_url = "http://127.0.0.1:5000/api/v1/test"
{case}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for endpoint constraint unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }

    // In tools:
    let tool_cases = [
        "constraints.cmd = { pattern = \".*\" }",
        "constraints.cmd = { max_length = 100 }",
        "constraints.cmd = { sanitized = true }",
    ];

    for case in tool_cases {
        let toml_str = format!(
            r#"
[defaults]

[tools."bash"]
{case}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for tool constraint unknown field '{case}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("unknown field"),
            "Error for '{case}' must indicate unknown field, got: {err_str}"
        );
    }
}

// ==============================================================================
// 2. Threshold & Timeout Bounds Adversarial Tests
// ==============================================================================

#[test]
fn test_defaults_threshold_bounds_negative_and_excess() {
    let invalid_thresholds = [
        ("max_injection_confidence", "-0.0001"),
        ("max_injection_confidence", "-1.0"),
        ("max_injection_confidence", "1.0001"),
        ("max_injection_confidence", "2.0"),
        ("max_injection_confidence", "999.0"),
        ("max_malicious_intent", "-0.0001"),
        ("max_malicious_intent", "-0.5"),
        ("max_malicious_intent", "1.0001"),
        ("max_malicious_intent", "50.0"),
    ];

    for (field, val) in invalid_thresholds {
        let toml_str = format!("[defaults]\n{field} = {val}\n");
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected threshold validation error for {field} = {val}"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("must be between 0.0 and 1.0"),
            "Expected out of bounds error message, got: {err_str}"
        );
    }
}

#[test]
fn test_defaults_threshold_bounds_nan_and_infinity() {
    let special_floats = [
        ("max_injection_confidence", "nan"),
        ("max_injection_confidence", "+nan"),
        ("max_injection_confidence", "-nan"),
        ("max_injection_confidence", "inf"),
        ("max_injection_confidence", "+inf"),
        ("max_injection_confidence", "-inf"),
        ("max_malicious_intent", "nan"),
        ("max_malicious_intent", "inf"),
        ("max_malicious_intent", "-inf"),
    ];

    for (field, val) in special_floats {
        let toml_str = format!("[defaults]\n{field} = {val}\n");
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected rejection of special float for {field} = {val}"
        );
    }
}

#[test]
fn test_defaults_model_timeout_bounds() {
    let invalid_timeouts = [
        "model_timeout_ms = 0",
        "model_timeout_ms = 60001",
        "model_timeout_ms = 100000",
        "model_timeout_ms = 9999999",
    ];

    for case in invalid_timeouts {
        let toml_str = format!("[defaults]\n{case}\n");
        let result = PolicyEngine::from_str(&toml_str);
        assert!(result.is_err(), "Expected failure for timeout '{case}'");
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("model_timeout_ms must be between 1 and 60000"),
            "Unexpected error message for '{case}': {err_str}"
        );
    }

    // Negative timeout in TOML causes deserialization error since target is u64
    let neg_timeout = "[defaults]\nmodel_timeout_ms = -1\n";
    assert!(PolicyEngine::from_str(neg_timeout).is_err());
}

#[test]
fn test_threshold_and_timeout_exact_boundaries_accepted() {
    // 0.0, 1.0 for thresholds; 1 and 60000 for timeout
    let toml_str = r#"
[defaults]
max_injection_confidence = 0.0
max_malicious_intent = 1.0
model_timeout_ms = 1

[endpoints."/api/v1/boundary"]
backend_url = "http://127.0.0.1:5000/api/v1/boundary"
max_injection_confidence = 1.0
max_malicious_intent = 0.0
model_timeout_ms = 60000

[tools."boundary_tool"]
max_injection_confidence = 0.0
max_malicious_intent = 1.0
model_timeout_ms = 60000
"#;
    let engine = PolicyEngine::from_str(toml_str).expect("Exact boundaries must be valid");
    assert_eq!(engine.defaults().max_injection_confidence, 0.0);
    assert_eq!(engine.defaults().max_malicious_intent, 1.0);
    assert_eq!(engine.defaults().model_timeout_ms, 1);

    let (inj, intent) = engine.effective_thresholds_for_endpoint("/api/v1/boundary");
    assert_eq!(inj, 1.0);
    assert_eq!(intent, 0.0);
    assert_eq!(
        engine.effective_timeout_for_endpoint("/api/v1/boundary"),
        60000
    );

    let (t_inj, t_intent) = engine.effective_thresholds_for_tool("boundary_tool");
    assert_eq!(t_inj, 0.0);
    assert_eq!(t_intent, 1.0);
    assert_eq!(engine.effective_timeout_for_tool("boundary_tool"), 60000);
}

#[test]
fn test_endpoint_override_threshold_and_timeout_bounds() {
    let invalid_overrides = [
        "max_injection_confidence = -0.1",
        "max_injection_confidence = 1.05",
        "max_injection_confidence = nan",
        "max_injection_confidence = inf",
        "max_malicious_intent = -0.5",
        "max_malicious_intent = 2.0",
        "max_malicious_intent = nan",
        "max_malicious_intent = -inf",
        "model_timeout_ms = 0",
        "model_timeout_ms = 60001",
        "model_timeout_ms = 120000",
    ];

    for override_line in invalid_overrides {
        let toml_str = format!(
            r#"
[defaults]

[endpoints."/api/v1/test"]
backend_url = "http://127.0.0.1:5000/api/v1/test"
{override_line}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for endpoint override '{override_line}'"
        );
    }
}

#[test]
fn test_tool_override_threshold_and_timeout_bounds() {
    let invalid_overrides = [
        "max_injection_confidence = -0.1",
        "max_injection_confidence = 1.05",
        "max_injection_confidence = nan",
        "max_injection_confidence = inf",
        "max_malicious_intent = -0.5",
        "max_malicious_intent = 2.0",
        "max_malicious_intent = nan",
        "max_malicious_intent = -inf",
        "model_timeout_ms = 0",
        "model_timeout_ms = 60001",
        "model_timeout_ms = 120000",
    ];

    for override_line in invalid_overrides {
        let toml_str = format!(
            r#"
[defaults]

[tools."test_tool"]
{override_line}
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected failure for tool override '{override_line}'"
        );
    }
}

// ==============================================================================
// 3. Secret Length Validation & Credential Leakage Adversarial Tests
// ==============================================================================

#[test]
fn test_secret_length_validation_matrix_sidecar_secret() {
    let dummy_jwt = "valid_jwt_secret_16b";

    // 1. Missing secret
    let env_missing: HashMap<String, String> = [("JWT_SECRET".to_string(), dummy_jwt.to_string())]
        .into_iter()
        .collect();
    assert_eq!(
        Config::from_vars(env_missing),
        Err(ConfigError::MissingVar("SIDECAR_SECRET"))
    );

    // 2. Empty secret (0 bytes)
    let env_empty: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "".to_string()),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_empty),
        Err(ConfigError::EmptyVar("SIDECAR_SECRET"))
    );

    // 3. Whitespace-only secret
    let env_whitespace: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "   \t\n  ".to_string()),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_whitespace),
        Err(ConfigError::EmptyVar("SIDECAR_SECRET"))
    );

    // 3b. 16 spaces (satisfies len >= 16, but is whitespace-only: must fail closed as empty)
    let env_16spaces: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), " ".repeat(16)),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_16spaces),
        Err(ConfigError::EmptyVar("SIDECAR_SECRET")),
        "16 spaces must be rejected as empty secret"
    );

    // 4. 1-byte secret
    let env_1byte: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "a".to_string()),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_1byte),
        Err(ConfigError::InvalidValue {
            var: "SIDECAR_SECRET",
            reason: "secret length must be at least 16 bytes (got 1 bytes)".to_string()
        })
    );

    // 5. 15-byte secret (just below MIN_SECRET_LENGTH boundary)
    let env_15bytes: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "123456789012345".to_string()),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_15bytes),
        Err(ConfigError::InvalidValue {
            var: "SIDECAR_SECRET",
            reason: "secret length must be at least 16 bytes (got 15 bytes)".to_string()
        })
    );

    // 6. 16-byte secret (exact boundary, valid)
    let env_16bytes: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "1234567890123456".to_string()),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    let config = Config::from_vars(env_16bytes).expect("16-byte secret must be accepted");
    assert_eq!(config.sidecar_secret, "1234567890123456");

    // 7. 64-byte secret (high-entropy, valid)
    let env_64bytes: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), "a".repeat(64)),
        ("JWT_SECRET".to_string(), dummy_jwt.to_string()),
    ]
    .into_iter()
    .collect();
    assert!(Config::from_vars(env_64bytes).is_ok());
}

#[test]
fn test_secret_length_validation_matrix_jwt_secret() {
    let dummy_sidecar = "valid_sidecar_sec16b";

    // 1. Missing JWT secret
    let env_missing: HashMap<String, String> =
        [("SIDECAR_SECRET".to_string(), dummy_sidecar.to_string())]
            .into_iter()
            .collect();
    assert_eq!(
        Config::from_vars(env_missing),
        Err(ConfigError::MissingVar("JWT_SECRET"))
    );

    // 2. 15-byte JWT secret
    let env_15bytes: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_sidecar.to_string()),
        ("JWT_SECRET".to_string(), "123456789012345".to_string()),
    ]
    .into_iter()
    .collect();
    assert_eq!(
        Config::from_vars(env_15bytes),
        Err(ConfigError::InvalidValue {
            var: "JWT_SECRET",
            reason: "secret length must be at least 16 bytes (got 15 bytes)".to_string()
        })
    );

    // 3. 16-byte JWT secret
    let env_16bytes: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_sidecar.to_string()),
        ("JWT_SECRET".to_string(), "1234567890123456".to_string()),
    ]
    .into_iter()
    .collect();
    let config = Config::from_vars(env_16bytes).expect("16-byte JWT secret must be accepted");
    assert_eq!(config.jwt_secret, "1234567890123456");
}

#[test]
fn test_secret_length_unicode_multi_byte_handling() {
    let dummy_sidecar = "valid_sidecar_sec16b";

    // Case A: 5 CJK characters, each 3 bytes in UTF-8 -> 15 bytes total, but 5 chars.
    // Must be REJECTED because byte length 15 < 16.
    let cjk_15b = "密碼設定測";
    assert_eq!(cjk_15b.len(), 15, "Sanity check: UTF-8 byte length is 15");
    assert_eq!(cjk_15b.chars().count(), 5, "Sanity check: char count is 5");

    let env_cjk_15b: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_sidecar.to_string()),
        ("JWT_SECRET".to_string(), cjk_15b.to_string()),
    ]
    .into_iter()
    .collect();
    let res = Config::from_vars(env_cjk_15b);
    assert_eq!(
        res,
        Err(ConfigError::InvalidValue {
            var: "JWT_SECRET",
            reason: "secret length must be at least 16 bytes (got 15 bytes)".to_string()
        })
    );

    // Case B: 6 CJK characters, each 3 bytes in UTF-8 -> 18 bytes total, 6 chars.
    // Must be ACCEPTED because byte length 18 >= 16.
    let cjk_18b = "密碼設定測試";
    assert_eq!(cjk_18b.len(), 18, "Sanity check: UTF-8 byte length is 18");
    assert_eq!(cjk_18b.chars().count(), 6, "Sanity check: char count is 6");
    let env_cjk_18b: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_sidecar.to_string()),
        ("JWT_SECRET".to_string(), cjk_18b.to_string()),
    ]
    .into_iter()
    .collect();
    assert!(
        Config::from_vars(env_cjk_18b).is_ok(),
        "18-byte multi-byte secret must be accepted"
    );

    // Case C: 4 4-byte emoji -> 16 bytes total, 4 chars.
    // Must be ACCEPTED because byte length 16 >= 16.
    let emoji_16b = "🔐🔑🛡️⚡";
    assert!(
        emoji_16b.len() >= 16,
        "Emoji string byte length is {}",
        emoji_16b.len()
    );
    let env_emoji: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), emoji_16b.to_string()),
        ("JWT_SECRET".to_string(), dummy_sidecar.to_string()),
    ]
    .into_iter()
    .collect();
    assert!(
        Config::from_vars(env_emoji).is_ok(),
        "Emoji secret with >= 16 bytes must be accepted"
    );
}

#[test]
fn test_secret_value_never_leaks_in_error_messages() {
    let super_sensitive_secret = "super_sec_12345"; // 15 bytes
    let env: HashMap<String, String> = [
        (
            "SIDECAR_SECRET".to_string(),
            super_sensitive_secret.to_string(),
        ),
        ("JWT_SECRET".to_string(), "valid_jwt_secret_16b".to_string()),
    ]
    .into_iter()
    .collect();

    let err = Config::from_vars(env).expect_err("Must fail due to 15-byte secret");
    let display_err = format!("{err}");
    let debug_err = format!("{err:?}");

    // The raw secret value MUST NOT appear in Display representation
    assert!(
        !display_err.contains(super_sensitive_secret),
        "CRITICAL LEAK: Secret text leaked in Display output: {display_err}"
    );

    // The raw secret value MUST NOT appear in Debug representation
    assert!(
        !debug_err.contains(super_sensitive_secret),
        "CRITICAL LEAK: Secret text leaked in Debug output: {debug_err}"
    );

    // Verify it only reports byte length
    assert!(display_err.contains("got 15 bytes"));
}

// ==============================================================================
// 4. URL Scheme & Host Validation Adversarial Tests
// ==============================================================================

#[test]
fn test_config_url_validation_adversarial_schemes_and_hosts() {
    let dummy_secret = "valid_secret_16_bytes";

    let invalid_urls = [
        // Non-HTTP/HTTPS schemes
        ("file:///etc/passwd", "scheme must be 'http' or 'https'"),
        (
            "ftp://ftp.server.com/file",
            "scheme must be 'http' or 'https'",
        ),
        ("ws://127.0.0.1:8080", "scheme must be 'http' or 'https'"),
        ("wss://127.0.0.1:8080", "scheme must be 'http' or 'https'"),
        ("gopher://floodgap.com", "scheme must be 'http' or 'https'"),
        ("javascript:alert(1)", "scheme must be 'http' or 'https'"),
        (
            "data:text/plain;base64,SGVs",
            "scheme must be 'http' or 'https'",
        ),
        // Missing scheme
        ("localhost:5000", "scheme must be 'http' or 'https'"),
        ("127.0.0.1:5000", "Failed to parse URL"),
        ("api.anthropic.com/v1", "Failed to parse URL"),
        // Empty or whitespace
        ("", "URL cannot be empty or whitespace"),
        ("   \t \n  ", "URL cannot be empty or whitespace"),
        // Missing or empty host
        ("http://", "must include a valid host"),
        ("https://", "must include a valid host"),
        ("http:///", "must include a valid host"),
        ("http://:8080", "must include a valid host"),
    ];

    let url_vars = ["UPSTREAM_LLM_URL", "FLASK_BACKEND_URL", "KEV_URL"];

    for var in url_vars {
        for (raw_url, expected_substr) in &invalid_urls {
            let mut env = HashMap::new();
            env.insert("SIDECAR_SECRET".to_string(), dummy_secret.to_string());
            env.insert("JWT_SECRET".to_string(), dummy_secret.to_string());
            env.insert(var.to_string(), raw_url.to_string());

            let result = Config::from_vars(env);
            assert!(
                result.is_err(),
                "Expected failure for {var}='{raw_url}', but got Ok"
            );
            let err_str = result.err().unwrap().to_string();
            assert!(
                err_str.contains(expected_substr)
                    || (expected_substr.contains("valid host") && err_str.contains("empty host")),
                "Error for {var}='{raw_url}' did not contain '{expected_substr}': {err_str}"
            );
        }
    }
}

#[test]
fn test_policy_backend_url_validation_adversarial_schemes_and_hosts() {
    let invalid_backend_urls = [
        "file:///etc/shadow",
        "ftp://backup.server/db",
        "ws://localhost:5000",
        "gopher://example.com",
        "javascript:void(0)",
        "localhost:5000",
        "api.anthropic.com",
        "http://",
        "http:///",
        "http://:5000",
        "http://?query=1",
        "",
        "   ",
    ];

    for backend_url in invalid_backend_urls {
        let toml_str = format!(
            r#"
[defaults]

[endpoints."/api/v1/test"]
backend_url = "{backend_url}"
"#
        );
        let result = PolicyEngine::from_str(&toml_str);
        assert!(
            result.is_err(),
            "Expected PolicyEngine to reject backend_url '{backend_url}'"
        );
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("invalid backend_url")
                || err_str.contains("must use http or https")
                || err_str.contains("valid non-empty host"),
            "Unexpected error message for '{backend_url}': {err_str}"
        );
    }
}

// ==============================================================================
// 5. Policy Engine & Tool Evaluation Invariants
// ==============================================================================

#[test]
fn test_effective_thresholds_inheritance_and_overrides() {
    let toml_str = r#"
[defaults]
max_injection_confidence = 0.70
max_malicious_intent = 0.50
model_timeout_ms = 500

[endpoints."/api/v1/default_inherited"]
backend_url = "http://127.0.0.1:5000/api/v1/default_inherited"

[endpoints."/api/v1/custom_override"]
backend_url = "http://127.0.0.1:5000/api/v1/custom_override"
max_injection_confidence = 0.65
max_malicious_intent = 0.35
model_timeout_ms = 1200

[tools."inherited_tool"]
enabled = true

[tools."custom_tool"]
max_injection_confidence = 0.45
max_malicious_intent = 0.25
model_timeout_ms = 2500
"#;

    let engine = PolicyEngine::from_str(toml_str).unwrap();

    // 1. Endpoint inheritance
    assert_eq!(
        engine.effective_thresholds_for_endpoint("/api/v1/default_inherited"),
        (0.70, 0.50)
    );
    assert_eq!(
        engine.effective_timeout_for_endpoint("/api/v1/default_inherited"),
        500
    );

    // 2. Endpoint override
    assert_eq!(
        engine.effective_thresholds_for_endpoint("/api/v1/custom_override"),
        (0.65, 0.35)
    );
    assert_eq!(
        engine.effective_timeout_for_endpoint("/api/v1/custom_override"),
        1200
    );

    // 3. Unlisted endpoint falls back to defaults
    assert_eq!(
        engine.effective_thresholds_for_endpoint("/api/v1/non_existent"),
        (0.70, 0.50)
    );
    assert_eq!(
        engine.effective_timeout_for_endpoint("/api/v1/non_existent"),
        500
    );

    // 4. Tool inheritance
    assert_eq!(
        engine.effective_thresholds_for_tool("inherited_tool"),
        (0.70, 0.50)
    );
    assert_eq!(engine.effective_timeout_for_tool("inherited_tool"), 500);

    // 5. Tool override
    assert_eq!(
        engine.effective_thresholds_for_tool("custom_tool"),
        (0.45, 0.25)
    );
    assert_eq!(engine.effective_timeout_for_tool("custom_tool"), 2500);

    // 6. Unlisted tool falls back to defaults
    assert_eq!(
        engine.effective_thresholds_for_tool("unlisted_tool"),
        (0.70, 0.50)
    );
    assert_eq!(engine.effective_timeout_for_tool("unlisted_tool"), 500);
}

#[test]
fn test_tool_lookup_case_insensitivity() {
    let toml_str = r#"
[defaults]

[tools."Bash"]
enabled = true
allowed_roles = ["admin"]

[tools."FILE_EDIT"]
enabled = true
allowed_roles = ["agent"]
"#;
    let engine = PolicyEngine::from_str(toml_str).unwrap();

    // Exact matches
    assert!(engine.get_tool("Bash").is_some());
    assert!(engine.get_tool("FILE_EDIT").is_some());

    // Case-insensitive fallbacks
    assert!(engine.get_tool("bash").is_some());
    assert!(engine.get_tool("BASH").is_some());
    assert!(engine.get_tool("bAsH").is_some());
    assert!(engine.get_tool("file_edit").is_some());
    assert!(engine.get_tool("File_Edit").is_some());

    // Non-existent tool returns None
    assert!(engine.get_tool("git_push").is_none());
}

#[test]
fn test_evaluate_tool_precedence_and_validation() {
    let toml_str = r#"
[defaults]

[tools."bash"]
enabled = true
allowed_roles = ["admin"]
required_args = ["command"]
constraints.command = { regex = "^echo [a-z]+$", max_len = 16 }

[tools."disabled_tool"]
enabled = false
allowed_roles = ["admin"]
"#;
    let engine = PolicyEngine::from_str(toml_str).unwrap();

    let admin_claims = AgentClaims {
        sub: "user-admin".to_string(),
        role: "admin".to_string(),
        exp: 2_000_000_000,
        payload_sha256: None,
    };
    let agent_claims = AgentClaims {
        sub: "user-agent".to_string(),
        role: "agent".to_string(),
        exp: 2_000_000_000,
        payload_sha256: None,
    };

    // 1. Unlisted tool -> EndpointDenied
    let mut payload = serde_json::Map::new();
    payload.insert("command".to_string(), serde_json::json!("echo hello"));
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "non_existent_tool", &payload),
        Err(DenyRule::EndpointDenied)
    );

    // 2. Disabled tool -> EndpointDenied
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "disabled_tool", &payload),
        Err(DenyRule::EndpointDenied)
    );

    // 3. Unauthorized role -> UnauthorizedRole
    assert_eq!(
        engine.evaluate_tool(&agent_claims, "bash", &payload),
        Err(DenyRule::UnauthorizedRole)
    );

    // 4. Missing required argument -> MissingArg
    let empty_payload = serde_json::Map::new();
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &empty_payload),
        Err(DenyRule::MissingArg)
    );

    // 5. Null required argument -> MissingArg
    let mut null_payload = serde_json::Map::new();
    null_payload.insert("command".to_string(), serde_json::Value::Null);
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &null_payload),
        Err(DenyRule::MissingArg)
    );

    // 6. Constraint failure: Regex mismatch -> ConstraintFailed
    let mut bad_regex_payload = serde_json::Map::new();
    bad_regex_payload.insert("command".to_string(), serde_json::json!("rm -rf /"));
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &bad_regex_payload),
        Err(DenyRule::ConstraintFailed)
    );

    // 7. Constraint failure: max_len exceeded -> ConstraintFailed
    let mut toolong_payload = serde_json::Map::new();
    toolong_payload.insert(
        "command".to_string(),
        serde_json::json!("echo abcdefghijklmnop"), // 21 chars > 16 max_len
    );
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &toolong_payload),
        Err(DenyRule::ConstraintFailed)
    );

    // 8. Valid call -> Ok(())
    let mut valid_payload = serde_json::Map::new();
    valid_payload.insert("command".to_string(), serde_json::json!("echo hello"));
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &valid_payload),
        Ok(())
    );

    // Precedence test: Disabled beats UnauthorizedRole
    assert_eq!(
        engine.evaluate_tool(&agent_claims, "disabled_tool", &empty_payload),
        Err(DenyRule::EndpointDenied)
    );

    // Precedence test: UnauthorizedRole beats MissingArg
    assert_eq!(
        engine.evaluate_tool(&agent_claims, "bash", &empty_payload),
        Err(DenyRule::UnauthorizedRole)
    );

    // Precedence test: MissingArg beats ConstraintFailed
    let mut missing_and_invalid = serde_json::Map::new();
    missing_and_invalid.insert("other".to_string(), serde_json::json!("invalid"));
    assert_eq!(
        engine.evaluate_tool(&admin_claims, "bash", &missing_and_invalid),
        Err(DenyRule::MissingArg)
    );
}

#[test]
fn test_agent_cli_mappings_environment_synthesis() {
    let engine = PolicyEngine::from_str(BASE_VALID_POLICY).unwrap();

    // Verify configured_clis
    let mut clis = engine.configured_clis();
    clis.sort();
    assert_eq!(clis, vec!["claude"]);

    // Verify get_cli_mapping
    let mapping = engine
        .get_cli_mapping("claude")
        .expect("claude mapping exists");
    assert_eq!(mapping.env_var, "ANTHROPIC_BASE_URL");
    assert_eq!(
        mapping.api_key_env_var.as_deref(),
        Some("ANTHROPIC_API_KEY")
    );
    assert_eq!(mapping.sidecar_url, "http://127.0.0.1:8080");
    assert_eq!(mapping.protocol, "anthropic");

    // Verify get_cli_env synthesis
    let env_vars = engine.get_cli_env("claude").expect("cli env generated");
    assert_eq!(
        env_vars.get("ANTHROPIC_BASE_URL").map(|s| s.as_str()),
        Some("http://127.0.0.1:8080")
    );
    assert_eq!(
        env_vars
            .get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")
            .map(|s| s.as_str()),
        Some("1")
    );

    // Non-existent CLI returns None
    assert!(engine.get_cli_mapping("non_existent").is_none());
    assert!(engine.get_cli_env("non_existent").is_none());
}

#[test]
fn test_audit_logger_registers_cli_mapping_api_keys() {
    let engine = PolicyEngine::from_str(BASE_VALID_POLICY).unwrap();
    let mut logger = AuditLogger::buffered();

    // Mock environment lookup returning a secret for ANTHROPIC_API_KEY
    let mock_env: HashMap<String, String> = [(
        "ANTHROPIC_API_KEY".to_string(),
        "sk-ant-api03-very-secret-token-123456789".to_string(),
    )]
    .into_iter()
    .collect();

    logger.register_cli_mapping_secrets(&engine, |k| mock_env.get(k).cloned());

    // Create an audit entry containing the CLI secret in its reason
    let entry = AuditEntry {
        timestamp: chrono::Utc::now(),
        route: "RouteA".to_string(),
        session_id: Some("sess-123".to_string()),
        caller_sub: Some("user-test".to_string()),
        caller_role: Some("agent".to_string()),
        target_endpoint: Some("/v1/messages".to_string()),
        method: Some("POST".to_string()),
        decision: "deny".to_string(),
        matched_rule: Some(DenyRule::ModelRiskHigh),
        reason: Some(
            "Failed with token sk-ant-api03-very-secret-token-123456789 in upstream call"
                .to_string(),
        ),
        injection_confidence: Some(0.9),
        malicious_intent: Some(0.8),
        latency_ms: 12,
    };

    logger.log(entry).expect("Logging entry succeeds");

    let logged = logger.entries();
    assert_eq!(logged.len(), 1);
    let reason = logged[0].reason.as_deref().expect("Reason is present");

    assert!(
        !reason.contains("sk-ant-api03-very-secret-token-123456789"),
        "CLI API key was not redacted from diagnostic text: {reason}"
    );
    assert!(reason.contains("[REDACTED]"));
}

#[test]
fn test_url_validation_ipv6_and_invalid_ports() {
    let dummy_secret = "valid_secret_16_bytes";

    // Valid IPv6 URL must be accepted
    let env_ipv6: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_secret.to_string()),
        ("JWT_SECRET".to_string(), dummy_secret.to_string()),
        (
            "FLASK_BACKEND_URL".to_string(),
            "http://[::1]:5000".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    let config = Config::from_vars(env_ipv6).expect("Valid IPv6 URL must be accepted");
    assert_eq!(config.flask_backend_url, "http://[::1]:5000");

    // Invalid port (999999 > 65535) must be rejected
    let env_bad_port: HashMap<String, String> = [
        ("SIDECAR_SECRET".to_string(), dummy_secret.to_string()),
        ("JWT_SECRET".to_string(), dummy_secret.to_string()),
        (
            "FLASK_BACKEND_URL".to_string(),
            "http://127.0.0.1:999999".to_string(),
        ),
    ]
    .into_iter()
    .collect();
    assert!(
        Config::from_vars(env_bad_port).is_err(),
        "Invalid port must be rejected"
    );
}

#[test]
fn test_threshold_epsilon_boundaries() {
    // Exactly at boundaries: 0.0 and 1.0 are valid
    let toml_valid = r#"
[defaults]
max_injection_confidence = 0.0
max_malicious_intent = 1.0
model_timeout_ms = 500
"#;
    assert!(PolicyEngine::from_str(toml_valid).is_ok());

    // Slightly above 1.0: 1.00001 must be rejected
    let toml_above = r#"
[defaults]
max_injection_confidence = 1.00001
model_timeout_ms = 500
"#;
    assert!(PolicyEngine::from_str(toml_above).is_err());

    // Slightly below 0.0: -0.00001 must be rejected
    let toml_below = r#"
[defaults]
max_malicious_intent = -0.00001
model_timeout_ms = 500
"#;
    assert!(PolicyEngine::from_str(toml_below).is_err());
}
