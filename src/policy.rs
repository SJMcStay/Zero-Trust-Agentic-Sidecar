//! Deterministic Zero-Trust Policy Engine (`src/policy.rs`)
//!
//! Enforces fail-closed deterministic access control and argument validation
//! for Route B (`POST /api/execute`) tool execution requests before or in
//! parallel with ML risk classification.
//!
//! ReDoS Safety Invariant:
//! All parameter constraints are pre-compiled into Rust `regex::Regex` instances
//! at policy load time. The `regex` crate guarantees $O(M \cdot N)$ linear-time
//! matching via finite automata (lazy DFA / Pike VM) without backtracking,
//! preventing algorithmic complexity (ReDoS) denial-of-service attacks.

use regex::Regex;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::turn::TurnRules;
use crate::types::{
    AgentClaims, AlertPolicy, DefaultPolicy, DenyRule, PolicyConfig, RouteAPolicy,
    ToolExecuteRequest,
};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime};

/// Errors encountered while loading, parsing, or validating policy definitions.
#[derive(Debug, Error)]
pub enum PolicyError {
    #[error("failed to read policy configuration from '{path}': {source}")]
    IoError {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("failed to deserialize policy TOML: {0}")]
    ParseError(#[from] toml::de::Error),

    #[error("failed to compile linear-time regex '{pattern}' for argument '{arg}' in endpoint '{endpoint}': {source}")]
    InvalidRegex {
        endpoint: String,
        arg: String,
        pattern: String,
        #[source]
        source: regex::Error,
    },

    #[error("policy configuration validation failed: {0}")]
    ValidationError(String),
}

/// Pre-compiled argument constraint with cached linear-time Regex.
#[derive(Debug, Clone)]
pub struct CompiledArgConstraint {
    /// Pre-compiled linear-time regular expression pattern.
    pub regex: Option<Regex>,
    /// Maximum permitted character length (evaluated by Unicode character count).
    pub max_len: Option<usize>,
}

/// Pre-compiled endpoint access rule for sub-microsecond zero-lock evaluation.
#[derive(Debug, Clone)]
pub struct CompiledEndpointRule {
    /// Target forwarding URL on downstream microservice.
    pub backend_url: String,
    /// Whether the endpoint is active and callable.
    pub enabled: bool,
    /// Permitted HTTP methods (e.g. `["GET", "POST"]`).
    pub allowed_methods: Vec<String>,
    /// Permitted caller roles from JWT claims (e.g. `["agent", "admin"]`).
    pub allowed_roles: Vec<String>,
    /// Required argument keys in request payload.
    pub required_args: Vec<String>,
    /// Constraints applied per argument key.
    pub constraints: HashMap<String, CompiledArgConstraint>,
}

/// Thread-safe deterministic policy evaluation engine.
///
/// Implements `Send + Sync + Clone` with zero internal locks or mutexes,
/// allowing concurrent, lock-free evaluation across Tokio worker threads.
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    /// Raw deserialized policy configuration (defaults and rules).
    config: PolicyConfig,
    /// Pre-compiled endpoints indexed by target endpoint path for O(1) lookup.
    compiled_endpoints: HashMap<String, CompiledEndpointRule>,
    /// `[route_a]` patterns compiled, for reading new turns.
    turn_rules: TurnRules,
}

impl PolicyEngine {
    /// Loads, parses, compiles, and validates policy configuration from a file on disk.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PolicyError> {
        let p = path.as_ref();
        let content = std::fs::read_to_string(p).map_err(|e| PolicyError::IoError {
            path: p.to_path_buf(),
            source: e,
        })?;
        Self::from_str(&content)
    }

    /// Loads, parses, compiles, and validates policy configuration from a raw TOML string.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(toml_str: &str) -> Result<Self, PolicyError> {
        toml_str.parse()
    }
}

impl std::str::FromStr for PolicyEngine {
    type Err = PolicyError;

    fn from_str(toml_str: &str) -> Result<Self, Self::Err> {
        let config: PolicyConfig = toml::from_str(toml_str)?;

        // Validate operational thresholds (must be bounded in [0.0, 1.0])
        if !(0.0..=1.0).contains(&config.defaults.max_injection_confidence) {
            return Err(PolicyError::ValidationError(format!(
                "max_injection_confidence must be between 0.0 and 1.0, got {}",
                config.defaults.max_injection_confidence
            )));
        }
        if !(0.0..=1.0).contains(&config.defaults.max_malicious_intent) {
            return Err(PolicyError::ValidationError(format!(
                "max_malicious_intent must be between 0.0 and 1.0, got {}",
                config.defaults.max_malicious_intent
            )));
        }

        if config.route_a.context_chars > 20_000 {
            return Err(PolicyError::ValidationError(format!(
                "route_a.context_chars must be at most 20000, got {}",
                config.route_a.context_chars
            )));
        }
        let compile = |field: &str, patterns: &[String]| {
            patterns
                .iter()
                .map(|pat| {
                    Regex::new(pat).map_err(|e| {
                        PolicyError::ValidationError(format!(
                            "route_a.{field}: invalid regex '{pat}': {e}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()
        };
        let route_a = &config.route_a;
        let turn_rules = TurnRules {
            skip: compile("skip_patterns", &route_a.skip_patterns)?,
            tool_output_skip: compile(
                "tool_output_skip_patterns",
                &route_a.tool_output_skip_patterns,
            )?,
            inline_tool_output: compile(
                "inline_tool_output_patterns",
                &route_a.inline_tool_output_patterns,
            )?,
            include_user_text: route_a.include_user_text,
            context_chars: route_a.context_chars,
        };

        let mut compiled_endpoints = HashMap::with_capacity(config.endpoints.len());

        for (endpoint_path, endpoint_rule) in &config.endpoints {
            let mut compiled_constraints = HashMap::with_capacity(endpoint_rule.constraints.len());

            for (arg_name, constraint) in &endpoint_rule.constraints {
                let compiled_re = match &constraint.regex {
                    Some(pat) => {
                        let re = Regex::new(pat).map_err(|e| PolicyError::InvalidRegex {
                            endpoint: endpoint_path.clone(),
                            arg: arg_name.clone(),
                            pattern: pat.clone(),
                            source: e,
                        })?;
                        Some(re)
                    }
                    None => None,
                };

                compiled_constraints.insert(
                    arg_name.clone(),
                    CompiledArgConstraint {
                        regex: compiled_re,
                        max_len: constraint.max_len,
                    },
                );
            }

            compiled_endpoints.insert(
                endpoint_path.clone(),
                CompiledEndpointRule {
                    backend_url: endpoint_rule.backend_url.clone(),
                    enabled: endpoint_rule.enabled,
                    allowed_methods: endpoint_rule.allowed_methods.clone(),
                    allowed_roles: endpoint_rule.allowed_roles.clone(),
                    required_args: endpoint_rule.required_args.clone(),
                    constraints: compiled_constraints,
                },
            );
        }

        Ok(Self {
            config,
            compiled_endpoints,
            turn_rules,
        })
    }
}

/// A policy file that is re-read when it changes on disk, so users can adjust settings
/// without restarting. Checks the file's modification time at most once per second.
/// A broken edit keeps the last good policy (and logs why): settings never fail open.
pub struct LivePolicy {
    path: PathBuf,
    current: RwLock<Arc<PolicyEngine>>,
    seen_modified: Mutex<Option<SystemTime>>,
    last_check: Mutex<Instant>,
}

impl LivePolicy {
    pub const CHECK_EVERY: Duration = Duration::from_secs(1);

    pub fn new(path: impl Into<PathBuf>, initial: Arc<PolicyEngine>) -> Self {
        let path = path.into();
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        Self {
            path,
            current: RwLock::new(initial),
            seen_modified: Mutex::new(modified),
            last_check: Mutex::new(Instant::now()),
        }
    }

    /// The policy to use for this request; reloads first if the file changed.
    pub fn current(&self) -> Arc<PolicyEngine> {
        let due = {
            let mut last = self.last_check.lock().unwrap_or_else(|e| e.into_inner());
            let due = last.elapsed() >= Self::CHECK_EVERY;
            if due {
                *last = Instant::now();
            }
            due
        };
        if due {
            self.reload_if_changed();
        }
        Arc::clone(&self.current.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// Reloads when the modification time differs from the last one seen. Returns true on reload.
    pub fn reload_if_changed(&self) -> bool {
        let modified = std::fs::metadata(&self.path)
            .and_then(|m| m.modified())
            .ok();
        let mut seen = self.seen_modified.lock().unwrap_or_else(|e| e.into_inner());
        if modified.is_none() || modified == *seen {
            return false;
        }
        *seen = modified;
        match PolicyEngine::from_file(&self.path) {
            Ok(engine) => {
                *self.current.write().unwrap_or_else(|e| e.into_inner()) = Arc::new(engine);
                tracing::info!("Policy reloaded from {:?}", self.path);
                true
            }
            Err(err) => {
                tracing::error!(
                    "Policy file {:?} changed but is invalid; keeping the previous policy: {err}",
                    self.path
                );
                false
            }
        }
    }
}

impl PolicyEngine {
    /// Returns a reference to the parsed configuration.
    pub fn config(&self) -> &PolicyConfig {
        &self.config
    }

    /// Returns a reference to global default thresholds and settings.
    pub fn defaults(&self) -> &DefaultPolicy {
        &self.config.defaults
    }

    /// `[route_a]` settings: scoring scope, mode, block response.
    pub fn route_a(&self) -> &RouteAPolicy {
        &self.config.route_a
    }

    /// `[route_a]` patterns and options for reading a new turn.
    pub fn turn_rules(&self) -> &TurnRules {
        &self.turn_rules
    }

    /// `[alerts]` settings.
    pub fn alerts(&self) -> &AlertPolicy {
        &self.config.alerts
    }

    /// Retrieves pre-compiled endpoint rule for an exact endpoint path.
    pub fn get_endpoint(&self, endpoint: &str) -> Option<&CompiledEndpointRule> {
        self.compiled_endpoints.get(endpoint)
    }

    /// Resolves target backend URL for an approved endpoint.
    pub fn backend_url_for(&self, endpoint: &str) -> Option<&str> {
        self.compiled_endpoints
            .get(endpoint)
            .map(|e| e.backend_url.as_str())
    }

    /// Evaluates inbound tool execution request against deterministic zero-trust policy rules.
    ///
    /// Evaluation Priority Hierarchy (Strict 5-Step Order):
    /// 1. Endpoint allowlist & `enabled == true` -> `DenyRule::EndpointDenied`
    /// 2. Method in `allowed_methods` -> `DenyRule::EndpointDenied`
    /// 3. Caller role in `allowed_roles` -> `DenyRule::UnauthorizedRole`
    /// 4. Required arguments non-null & present in payload -> `DenyRule::MissingArg`
    /// 5. Argument constraints (max_len & regex) satisfied -> `DenyRule::ConstraintFailed`
    pub fn evaluate(&self, claims: &AgentClaims, req: &ToolExecuteRequest) -> Result<(), DenyRule> {
        // Step 1: Endpoint allowlist & enabled check
        let endpoint_rule = self
            .compiled_endpoints
            .get(&req.endpoint)
            .ok_or(DenyRule::EndpointDenied)?;

        if !endpoint_rule.enabled {
            return Err(DenyRule::EndpointDenied);
        }

        // Step 2: HTTP Method validation (case-insensitive check against allowed_methods)
        let method_allowed = endpoint_rule
            .allowed_methods
            .iter()
            .any(|m| m.eq_ignore_ascii_case(&req.method));
        if !method_allowed {
            return Err(DenyRule::EndpointDenied);
        }

        // Step 3: Caller role authorization against allowed_roles
        let role_authorized = endpoint_rule
            .allowed_roles
            .iter()
            .any(|r| r == &claims.role);
        if !role_authorized {
            return Err(DenyRule::UnauthorizedRole);
        }

        // Step 4: Required arguments presence and non-null validation
        for req_arg in &endpoint_rule.required_args {
            match req.payload.get(req_arg) {
                Some(val) if !val.is_null() => {
                    // Argument present and non-null
                }
                _ => return Err(DenyRule::MissingArg),
            }
        }

        // Step 5: Argument constraints validation (max_len and regex)
        // Sort keys to guarantee deterministic order of constraint evaluation
        let mut sorted_keys: Vec<_> = endpoint_rule.constraints.keys().collect();
        sorted_keys.sort();

        for arg_name in sorted_keys {
            let constraint = &endpoint_rule.constraints[arg_name];

            if let Some(val) = req.payload.get(arg_name) {
                // If present and null, reject
                if val.is_null() {
                    return Err(DenyRule::ConstraintFailed);
                }

                // String constraints apply exclusively to string values;
                // non-string types fail constraint validation.
                let str_val = val.as_str().ok_or(DenyRule::ConstraintFailed)?;

                // 5a. Maximum length constraint (Unicode character count)
                if let Some(max_len) = constraint.max_len {
                    if str_val.chars().count() > max_len {
                        return Err(DenyRule::ConstraintFailed);
                    }
                }

                // 5b. Linear-time Regex pattern matching
                if let Some(re) = &constraint.regex {
                    if !re.is_match(str_val) {
                        return Err(DenyRule::ConstraintFailed);
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AgentClaims, ToolContext, ToolExecuteRequest};
    use serde_json::json;
    use std::path::Path;

    const SAMPLE_POLICY_TOML: &str = r#"
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
    "#;

    fn test_engine() -> PolicyEngine {
        PolicyEngine::from_str(SAMPLE_POLICY_TOML).expect("failed to load test policy")
    }

    fn test_claims(role: &str) -> AgentClaims {
        AgentClaims {
            sub: "agent-007".to_string(),
            role: role.to_string(),
            exp: 1800000000,
            payload_sha256: None,
        }
    }

    fn tool_request(
        endpoint: &str,
        method: &str,
        payload: serde_json::Value,
    ) -> ToolExecuteRequest {
        ToolExecuteRequest {
            session_id: "sess-1".to_string(),
            endpoint: endpoint.to_string(),
            method: method.to_string(),
            payload: payload.as_object().cloned().unwrap_or_default(),
            context: ToolContext::default(),
        }
    }

    #[test]
    fn test_from_file_loads_repo_policy_toml() {
        let engine = PolicyEngine::from_file(Path::new("policy.toml")).unwrap();
        assert_eq!(engine.defaults().max_injection_confidence, 0.70);
        assert!(engine.get_endpoint("/api/v1/system/status").is_some());
        assert_eq!(
            engine.backend_url_for("/api/v1/system/status"),
            Some("http://127.0.0.1:5000/api/v1/system/status")
        );
    }

    #[test]
    fn test_from_file_non_existent_path_returns_io_error() {
        let res = PolicyEngine::from_file(Path::new("non_existent_policy_file.toml"));
        assert!(matches!(res, Err(PolicyError::IoError { .. })));
    }

    #[test]
    fn test_from_str_invalid_regex_fails_fast() {
        let bad_toml = r#"
        [defaults]
        [endpoints."/test"]
        backend_url = "http://test"
        required_args = ["name"]
        constraints.name = { regex = "[a-z" }
        "#;
        let res = PolicyEngine::from_str(bad_toml);
        assert!(matches!(res, Err(PolicyError::InvalidRegex { .. })));
    }

    #[test]
    fn test_from_str_invalid_threshold_fails_validation() {
        let bad_toml_injection = r#"
        [defaults]
        max_injection_confidence = 1.5
        "#;
        let res = PolicyEngine::from_str(bad_toml_injection);
        assert!(matches!(res, Err(PolicyError::ValidationError(_))));

        let bad_toml_intent = r#"
        [defaults]
        max_malicious_intent = -0.1
        "#;
        let res2 = PolicyEngine::from_str(bad_toml_intent);
        assert!(matches!(res2, Err(PolicyError::ValidationError(_))));
    }

    #[test]
    fn test_allowlist_unlisted_endpoint_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        let req = tool_request("/api/v1/unknown/endpoint", "POST", json!({}));
        assert_eq!(
            engine.evaluate(&claims, &req),
            Err(DenyRule::EndpointDenied)
        );
    }

    #[test]
    fn test_disabled_endpoint_rejected() {
        let engine = test_engine();
        let claims = test_claims("admin");
        let req = tool_request("/api/v1/db/drop", "POST", json!({"table": "users"}));
        assert_eq!(
            engine.evaluate(&claims, &req),
            Err(DenyRule::EndpointDenied)
        );
    }

    #[test]
    fn test_method_mismatch_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        // /api/v1/email/send only allows POST, reject GET and DELETE
        let req_get = tool_request(
            "/api/v1/email/send",
            "GET",
            json!({"to": "alice@ourcompany.com", "body": "hi"}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_get),
            Err(DenyRule::EndpointDenied)
        );

        let req_del = tool_request(
            "/api/v1/email/send",
            "DELETE",
            json!({"to": "alice@ourcompany.com", "body": "hi"}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_del),
            Err(DenyRule::EndpointDenied)
        );
    }

    #[test]
    fn test_method_case_insensitive_accepted() {
        let engine = test_engine();
        let claims = test_claims("agent");
        let req = tool_request(
            "/api/v1/system/status",
            "post",
            json!({"service": "auth-service"}),
        );
        assert_eq!(engine.evaluate(&claims, &req), Ok(()));

        let req_upper = tool_request(
            "/api/v1/system/status",
            "GET",
            json!({"service": "auth-service"}),
        );
        assert_eq!(engine.evaluate(&claims, &req_upper), Ok(()));
    }

    #[test]
    fn test_role_mismatch_rejected() {
        let engine = test_engine();
        // /api/v1/email/send allows ["agent", "admin"], "readonly" must be rejected
        let claims_readonly = test_claims("readonly");
        let req = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com",
                "body": "Daily report"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims_readonly, &req),
            Err(DenyRule::UnauthorizedRole)
        );

        let claims_unknown = test_claims("contractor");
        assert_eq!(
            engine.evaluate(&claims_unknown, &req),
            Err(DenyRule::UnauthorizedRole)
        );
    }

    #[test]
    fn test_missing_required_argument_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        // /api/v1/system/status requires "service"
        let req_empty = tool_request("/api/v1/system/status", "GET", json!({}));
        assert_eq!(
            engine.evaluate(&claims, &req_empty),
            Err(DenyRule::MissingArg)
        );

        // /api/v1/email/send requires "to" and "body"
        let req_missing_body = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_missing_body),
            Err(DenyRule::MissingArg)
        );
    }

    #[test]
    fn test_null_required_argument_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        let req_null = tool_request("/api/v1/system/status", "GET", json!({"service": null}));
        assert_eq!(
            engine.evaluate(&claims, &req_null),
            Err(DenyRule::MissingArg)
        );

        let req_null_body = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com",
                "body": null
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_null_body),
            Err(DenyRule::MissingArg)
        );
    }

    #[test]
    fn test_string_length_overflow_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        // service max_len is 64
        let long_service = "a".repeat(65);
        let req_service_overflow = tool_request(
            "/api/v1/system/status",
            "GET",
            json!({"service": long_service}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_service_overflow),
            Err(DenyRule::ConstraintFailed)
        );

        // body max_len is 2000
        let long_body = "b".repeat(2001);
        let req_body_overflow = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com",
                "body": long_body
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_body_overflow),
            Err(DenyRule::ConstraintFailed)
        );

        // Unicode multibyte character count test: 65 emojis (each 4 bytes, 65 chars > 64)
        let emoji_service = "🦀".repeat(65);
        let req_emoji = tool_request(
            "/api/v1/system/status",
            "GET",
            json!({"service": emoji_service}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_emoji),
            Err(DenyRule::ConstraintFailed)
        );
    }

    #[test]
    fn test_regex_constraint_failure_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");

        // Service regex: ^[a-z0-9-]+$
        let req_bad_service = tool_request(
            "/api/v1/system/status",
            "GET",
            json!({"service": "Core_Service!"}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_bad_service),
            Err(DenyRule::ConstraintFailed)
        );

        // Email regex: ^[^@]+@ourcompany\.com$
        let req_external_domain = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "attacker@evil.com",
                "body": "Hello"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_external_domain),
            Err(DenyRule::ConstraintFailed)
        );

        let req_subdomain_spoof = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com.attacker.com",
                "body": "Hello"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_subdomain_spoof),
            Err(DenyRule::ConstraintFailed)
        );

        let req_missing_local_part = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "@ourcompany.com",
                "body": "Hello"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_missing_local_part),
            Err(DenyRule::ConstraintFailed)
        );
    }

    #[test]
    fn test_non_string_argument_for_string_constraint_rejected() {
        let engine = test_engine();
        let claims = test_claims("agent");
        // Passing an integer where string constraint exists
        let req_num = tool_request("/api/v1/system/status", "GET", json!({"service": 12345}));
        assert_eq!(
            engine.evaluate(&claims, &req_num),
            Err(DenyRule::ConstraintFailed)
        );

        // Passing an array where string constraint exists
        let req_arr = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": ["alice@ourcompany.com"],
                "body": "Hello"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_arr),
            Err(DenyRule::ConstraintFailed)
        );

        // Passing a boolean
        let req_bool = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice@ourcompany.com",
                "body": true
            }),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_bool),
            Err(DenyRule::ConstraintFailed)
        );
    }

    #[test]
    fn test_valid_requests_succeed() {
        let engine = test_engine();
        let claims = test_claims("agent");

        // Valid status call
        let req_status = tool_request(
            "/api/v1/system/status",
            "GET",
            json!({"service": "core-v2"}),
        );
        assert_eq!(engine.evaluate(&claims, &req_status), Ok(()));

        // Valid email call
        let req_email = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "alice.smith+alerts@ourcompany.com",
                "body": "Deployment alert"
            }),
        );
        assert_eq!(engine.evaluate(&claims, &req_email), Ok(()));
    }

    #[test]
    fn test_decision_priority_hierarchy() {
        let engine = test_engine();
        let claims_readonly = test_claims("readonly");

        // Case 1: Disabled endpoint with role mismatch and missing args -> EndpointDenied wins (Step 1)
        let req1 = tool_request("/api/v1/db/drop", "POST", json!({}));
        assert_eq!(
            engine.evaluate(&claims_readonly, &req1),
            Err(DenyRule::EndpointDenied)
        );

        // Case 2: Method mismatch with role mismatch and missing args -> EndpointDenied wins (Step 2)
        let req2 = tool_request("/api/v1/email/send", "GET", json!({}));
        assert_eq!(
            engine.evaluate(&claims_readonly, &req2),
            Err(DenyRule::EndpointDenied)
        );

        // Case 3: Role mismatch with missing args and constraint violations -> UnauthorizedRole wins (Step 3)
        let req3 = tool_request("/api/v1/email/send", "POST", json!({}));
        assert_eq!(
            engine.evaluate(&claims_readonly, &req3),
            Err(DenyRule::UnauthorizedRole)
        );

        // Case 4: Missing args with constraint violations -> MissingArg wins (Step 4)
        let claims_agent = test_claims("agent");
        let req4 = tool_request(
            "/api/v1/email/send",
            "POST",
            json!({
                "to": "bad_regex_email" // missing "body"
            }),
        );
        assert_eq!(
            engine.evaluate(&claims_agent, &req4),
            Err(DenyRule::MissingArg)
        );
    }

    #[test]
    fn test_optional_constrained_argument() {
        let policy_str = r#"
        [defaults]
        [endpoints."/api/v1/test"]
        backend_url = "http://test"
        enabled = true
        allowed_methods = ["POST"]
        allowed_roles = ["agent"]
        required_args = ["req_field"]
        constraints.opt_field = { regex = "^[0-9]+$", max_len = 5 }
        "#;
        let engine = PolicyEngine::from_str(policy_str).unwrap();
        let claims = test_claims("agent");

        // 1. Omitted optional field -> OK
        let req_omitted = tool_request("/api/v1/test", "POST", json!({"req_field": "present"}));
        assert_eq!(engine.evaluate(&claims, &req_omitted), Ok(()));

        // 2. Present with valid value -> OK
        let req_valid = tool_request(
            "/api/v1/test",
            "POST",
            json!({"req_field": "present", "opt_field": "123"}),
        );
        assert_eq!(engine.evaluate(&claims, &req_valid), Ok(()));

        // 3. Present with invalid regex -> ConstraintFailed
        let req_bad_regex = tool_request(
            "/api/v1/test",
            "POST",
            json!({"req_field": "present", "opt_field": "abc"}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_bad_regex),
            Err(DenyRule::ConstraintFailed)
        );

        // 4. Present with length overflow -> ConstraintFailed
        let req_overflow = tool_request(
            "/api/v1/test",
            "POST",
            json!({"req_field": "present", "opt_field": "123456"}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_overflow),
            Err(DenyRule::ConstraintFailed)
        );

        // 5. Present with null value -> ConstraintFailed
        let req_null = tool_request(
            "/api/v1/test",
            "POST",
            json!({"req_field": "present", "opt_field": null}),
        );
        assert_eq!(
            engine.evaluate(&claims, &req_null),
            Err(DenyRule::ConstraintFailed)
        );
    }

    #[test]
    fn test_concurrent_thread_safe_evaluations() {
        use std::sync::Arc;
        let engine = Arc::new(test_engine());
        let mut handles = Vec::new();

        for i in 0..20 {
            let eng = Arc::clone(&engine);
            let handle = std::thread::spawn(move || {
                let role = if i % 2 == 0 { "agent" } else { "readonly" };
                let claims = test_claims(role);
                let req = tool_request("/api/v1/system/status", "GET", json!({"service": "auth"}));
                assert_eq!(eng.evaluate(&claims, &req), Ok(()));
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn repo_policy_documents_the_route_a_and_alert_defaults() {
        let engine = PolicyEngine::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/policy.toml"))
            .expect("policy.toml must load");
        assert_eq!(
            engine.route_a(),
            &RouteAPolicy::default(),
            "policy.toml [route_a] drifted from the defaults"
        );
        assert_eq!(engine.alerts(), &AlertPolicy::default());
        assert_eq!(engine.turn_rules().inline_tool_output.len(), 1);
    }
}
