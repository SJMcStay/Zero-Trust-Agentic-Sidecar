//! Zero-Trust Structured JSONL Audit Logger (`src/audit.rs`)
//!
//! Emits structured JSONL audit records to `stdout` in server mode and captures
//! records into a thread-safe in-memory buffer (`Arc<Mutex<Vec<AuditEntry>>>`)
//! during test execution, enforcing multi-layer secret redaction across all
//! diagnostic fields and error reasons.
//!
//! ReDoS Safety Invariant:
//! All redaction regular expressions are pre-compiled linear-time DFA patterns
//! (`regex::Regex`) cached via `OnceLock`. Matching guarantees $O(N)$ execution
//! time without catastrophic backtracking.

use regex::Regex;
use std::io::Write;
use std::sync::{Arc, Mutex, OnceLock};
use thiserror::Error;

use crate::config::Config;
use crate::types::AuditEntry;

static BEARER_REGEX: OnceLock<Regex> = OnceLock::new();
static JWT_REGEX: OnceLock<Regex> = OnceLock::new();
static ANTHROPIC_KEY_REGEX: OnceLock<Regex> = OnceLock::new();
static HEX_64_REGEX: OnceLock<Regex> = OnceLock::new();
static SENSITIVE_KV_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_bearer_regex() -> &'static Regex {
    BEARER_REGEX.get_or_init(|| {
        Regex::new(r"(?i)\bBearer\s+[A-Za-z0-9\-_\.=]+").expect("valid bearer regex")
    })
}

fn get_jwt_regex() -> &'static Regex {
    JWT_REGEX.get_or_init(|| {
        Regex::new(r"\beyJ[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\.[A-Za-z0-9-_]{10,}\b")
            .expect("valid jwt regex")
    })
}

fn get_anthropic_key_regex() -> &'static Regex {
    ANTHROPIC_KEY_REGEX.get_or_init(|| {
        Regex::new(r"\bsk-ant-[A-Za-z0-9\-_]{10,}\b").expect("valid anthropic key regex")
    })
}

fn get_hex_64_regex() -> &'static Regex {
    HEX_64_REGEX
        .get_or_init(|| Regex::new(r"\b[0-9a-fA-F]{64}\b").expect("valid 64-char hex regex"))
}

fn get_sensitive_kv_regex() -> &'static Regex {
    SENSITIVE_KV_REGEX.get_or_init(|| {
        Regex::new(
            r#"(?i)("?(?:password|token|secret|api[-_]?key|authorization|credential|private[-_]?key)"?\s*[:=]\s*)("?[^"',\s}]+"?)"#,
        )
        .expect("valid sensitive kv regex")
    })
}

/// Errors that can occur during audit logging.
#[derive(Debug, Error)]
pub enum AuditError {
    #[error("failed to serialize audit entry to JSON: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("I/O error writing audit log: {0}")]
    Io(#[from] std::io::Error),

    #[error("audit capture lock poisoned")]
    LockPoisoned,
}

/// Destination sink for audit log records.
#[derive(Clone, Debug)]
pub enum AuditSink {
    /// Writes JSONL records directly to standard output.
    Stdout,
    /// Captures audit records in a thread-safe in-memory vector for test assertions.
    Memory(Arc<Mutex<Vec<AuditEntry>>>),
    /// Writes to standard output AND records in an in-memory buffer.
    Both(Arc<Mutex<Vec<AuditEntry>>>),
}

/// Zero-trust structured JSONL audit logger with automated secret redaction.
#[derive(Clone, Debug)]
pub struct AuditLogger {
    sink: AuditSink,
    known_secrets: Vec<String>,
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::stdout()
    }
}

impl AuditLogger {
    /// Creates a logger that writes structured JSONL records directly to stdout.
    pub fn stdout() -> Self {
        Self {
            sink: AuditSink::Stdout,
            known_secrets: Vec::new(),
        }
    }

    /// Creates a logger that captures entries into an in-memory buffer for test assertions.
    pub fn buffered() -> Self {
        let (logger, _) = Self::capture();
        logger
    }

    /// Creates an in-memory capturing logger, returning both the logger and the shared buffer.
    pub fn capture() -> (Self, Arc<Mutex<Vec<AuditEntry>>>) {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let logger = Self {
            sink: AuditSink::Memory(Arc::clone(&buffer)),
            known_secrets: Vec::new(),
        };
        (logger, buffer)
    }

    /// Creates a logger that writes to stdout AND captures to buffer.
    pub fn both(buffer: Arc<Mutex<Vec<AuditEntry>>>) -> Self {
        Self {
            sink: AuditSink::Both(buffer),
            known_secrets: Vec::new(),
        }
    }

    /// Creates a stdout logger with active secrets pre-registered from `Config`.
    pub fn from_config(config: &Config) -> Self {
        let mut logger = Self::stdout();
        logger.register_config_secrets(config);
        logger
    }

    /// Registers a secret string to be scrubbed from all diagnostic output.
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.add_secret(secret);
        self
    }

    /// Registers multiple secret strings to be scrubbed.
    pub fn with_secrets<I, S>(mut self, secrets: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for s in secrets {
            self.add_secret(s);
        }
        self
    }

    /// Adds an individual secret string to the known secrets list.
    pub fn add_secret(&mut self, secret: impl Into<String>) {
        let s = secret.into().trim().to_string();
        if !s.is_empty() && s.len() >= 4 && !self.known_secrets.contains(&s) {
            self.known_secrets.push(s);
        }
    }

    /// Automatically registers all credentials from sidecar `Config`.
    pub fn register_config_secrets(&mut self, config: &Config) {
        if let Some(key) = &config.anthropic_api_key {
            self.add_secret(key);
        }
        self.add_secret(&config.sidecar_secret);
        self.add_secret(&config.jwt_secret);
    }

    /// Automatically registers all API keys configured in agent CLI mappings from process environment.
    pub fn register_cli_mapping_secrets_from_env(&mut self, policy: &crate::policy::PolicyEngine) {
        self.register_cli_mapping_secrets(policy, |var| std::env::var(var).ok());
    }

    /// Automatically registers all API keys configured in agent CLI mappings from an arbitrary resolver.
    pub fn register_cli_mapping_secrets<F>(
        &mut self,
        policy: &crate::policy::PolicyEngine,
        mut var_resolver: F,
    ) where
        F: FnMut(&str) -> Option<String>,
    {
        for mapping in policy.cli_mappings().values() {
            if let Some(key_var) = &mapping.api_key_env_var {
                if let Some(val) = var_resolver(key_var) {
                    let trimmed = val.trim();
                    if !trimmed.is_empty() && trimmed.len() >= 4 {
                        self.add_secret(trimmed);
                    }
                }
            }
        }
    }

    /// Factory method initializing AuditLogger from both Config and PolicyEngine.
    pub fn from_config_and_policy(config: &Config, policy: &crate::policy::PolicyEngine) -> Self {
        let mut logger = Self::stdout();
        logger.register_config_secrets(config);
        logger.register_cli_mapping_secrets_from_env(policy);
        logger
    }

    /// Sanitizes and logs an audit entry to the configured destination sink.
    pub fn log(&self, mut entry: AuditEntry) -> Result<(), AuditError> {
        entry.sanitize(&self.known_secrets);

        match &self.sink {
            AuditSink::Stdout => {
                let jsonl = self.format_jsonl(&entry)?;
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                handle.write_all(jsonl.as_bytes())?;
                handle.flush()?;
            }
            AuditSink::Memory(buffer) => {
                let mut guard = buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.push(entry);
            }
            AuditSink::Both(buffer) => {
                let jsonl = self.format_jsonl(&entry)?;
                let stdout = std::io::stdout();
                let mut handle = stdout.lock();
                handle.write_all(jsonl.as_bytes())?;
                handle.flush()?;

                let mut guard = buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                guard.push(entry);
            }
        }
        Ok(())
    }

    /// Returns a cloned vector of all captured entries (if memory or dual sink active).
    pub fn entries(&self) -> Vec<AuditEntry> {
        match &self.sink {
            AuditSink::Stdout => Vec::new(),
            AuditSink::Memory(buf) | AuditSink::Both(buf) => buf
                .lock()
                .map(|g| g.clone())
                .unwrap_or_else(|poisoned| poisoned.into_inner().clone()),
        }
    }

    /// Returns an optional cloned vector of captured entries.
    pub fn captured_entries(&self) -> Option<Vec<AuditEntry>> {
        match &self.sink {
            AuditSink::Stdout => None,
            AuditSink::Memory(buf) | AuditSink::Both(buf) => {
                let guard = buf.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                Some(guard.clone())
            }
        }
    }

    /// Empties the in-memory capture buffer.
    pub fn clear(&self) {
        if let AuditSink::Memory(buf) | AuditSink::Both(buf) = &self.sink {
            let mut guard = buf.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.clear();
        }
    }

    /// Alias for `clear()`.
    pub fn clear_captured(&self) {
        self.clear();
    }

    /// Returns slice of all registered known secrets.
    pub fn known_secrets(&self) -> &[String] {
        &self.known_secrets
    }

    /// Redacts all known secrets, API keys, tokens, and sensitive patterns from text.
    pub fn redact_string(&self, input: &str) -> String {
        redact_string(input, &self.known_secrets)
    }

    /// Recursively traverses a JSON Value, redacting sensitive keys and scrubbing all string leaves.
    pub fn redact_value(&self, value: &mut serde_json::Value) {
        redact_value(value, &self.known_secrets);
    }

    /// Formats an `AuditEntry` as a single-line JSONL string terminating in `\n`.
    pub fn format_jsonl(&self, entry: &AuditEntry) -> Result<String, AuditError> {
        let mut json = serde_json::to_string(entry)?;
        json.push('\n');
        Ok(json)
    }
}

/// Redacts known secrets, API keys, Bearer tokens, hex secrets, and sensitive key-values from arbitrary text.
pub fn redact_string(input: &str, known_secrets: &[String]) -> String {
    let mut result = input.to_string();

    // 1. Redact exact matches of registered known secrets (length >= 4)
    for secret in known_secrets {
        if !secret.is_empty() && secret.len() >= 4 {
            result = result.replace(secret, "[REDACTED]");
        }
    }

    // 2. Redact Bearer tokens
    if get_bearer_regex().is_match(&result) {
        result = get_bearer_regex()
            .replace_all(&result, "Bearer [REDACTED]")
            .to_string();
    }

    // 3. Redact Anthropic API keys (`sk-ant-*`)
    if get_anthropic_key_regex().is_match(&result) {
        result = get_anthropic_key_regex()
            .replace_all(&result, "[REDACTED_API_KEY]")
            .to_string();
    }

    // 4. Redact naked JWT tokens
    if get_jwt_regex().is_match(&result) {
        result = get_jwt_regex()
            .replace_all(&result, "[REDACTED_JWT]")
            .to_string();
    }

    // 5. Redact 64-character hex secrets
    if get_hex_64_regex().is_match(&result) {
        result = get_hex_64_regex()
            .replace_all(&result, "[REDACTED_HEX]")
            .to_string();
    }

    // 6. Redact sensitive key-value pairs
    if get_sensitive_kv_regex().is_match(&result) {
        result = get_sensitive_kv_regex()
            .replace_all(&result, r#"${1}"[REDACTED]""#)
            .to_string();
    }

    result
}

/// Convenience function to redact sensitive tokens and keys without custom known secrets.
pub fn redact_sensitive_text(input: &str) -> String {
    redact_string(input, &[])
}

/// Recursively traverses a JSON Value, masking values of sensitive keys
/// and scrubbing all string values against known secrets and sensitive patterns.
pub fn redact_value(value: &mut serde_json::Value, known_secrets: &[String]) {
    match value {
        serde_json::Value::Object(map) => {
            for (key, val) in map.iter_mut() {
                let lower = key.to_ascii_lowercase();
                if lower.contains("password")
                    || lower.contains("token")
                    || lower.contains("secret")
                    || lower.contains("api_key")
                    || lower.contains("apikey")
                    || lower.contains("credential")
                    || lower == "auth"
                    || lower == "authorization"
                {
                    *val = serde_json::Value::String("[REDACTED]".to_string());
                } else {
                    redact_value(val, known_secrets);
                }
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr.iter_mut() {
                redact_value(item, known_secrets);
            }
        }
        serde_json::Value::String(s) => {
            *s = redact_string(s, known_secrets);
        }
        _ => {}
    }
}

/// Backward-compatible wrapper for `redact_value` without custom known secrets.
pub fn redact_json_value(value: &mut serde_json::Value) {
    redact_value(value, &[]);
}

impl AuditEntry {
    /// Sanitizes all string fields in place to eliminate secret leaks.
    pub fn sanitize(&mut self, known_secrets: &[String]) {
        if let Some(sess) = &mut self.session_id {
            *sess = redact_string(sess, known_secrets);
        }
        if let Some(sub) = &mut self.caller_sub {
            *sub = redact_string(sub, known_secrets);
        }
        if let Some(role) = &mut self.caller_role {
            *role = redact_string(role, known_secrets);
        }
        if let Some(ep) = &mut self.target_endpoint {
            *ep = redact_string(ep, known_secrets);
        }
        if let Some(m) = &mut self.method {
            *m = redact_string(m, known_secrets);
        }
        if let Some(r) = &mut self.reason {
            *r = redact_string(r, known_secrets);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DenyRule;
    use serde_json::json;
    use std::collections::HashMap;

    fn sample_allow_entry() -> AuditEntry {
        AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some("sess-allow-01".to_string()),
            caller_sub: Some("agent-007".to_string()),
            caller_role: Some("agent".to_string()),
            target_endpoint: Some("/api/v1/system/status".to_string()),
            method: Some("POST".to_string()),
            decision: "allow".to_string(),
            matched_rule: None,
            reason: None,
            injection_confidence: Some(0.02),
            malicious_intent: Some(0.01),
            latency_ms: 15,
        }
    }

    fn sample_deny_entry(rule: DenyRule, reason: &str) -> AuditEntry {
        AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some("sess-deny-02".to_string()),
            caller_sub: Some("agent-007".to_string()),
            caller_role: Some("agent".to_string()),
            target_endpoint: Some("/api/v1/db/drop".to_string()),
            method: Some("POST".to_string()),
            decision: "deny".to_string(),
            matched_rule: Some(rule),
            reason: Some(reason.to_string()),
            injection_confidence: None,
            malicious_intent: None,
            latency_ms: 3,
        }
    }

    #[test]
    fn test_audit_logger_stdout_creation() {
        let logger = AuditLogger::stdout();
        assert!(matches!(logger.sink, AuditSink::Stdout));
        let def = AuditLogger::default();
        assert!(matches!(def.sink, AuditSink::Stdout));
    }

    #[test]
    fn test_audit_logger_capture_buffer_operations() {
        let (logger, buffer) = AuditLogger::capture();
        assert_eq!(logger.entries().len(), 0);

        logger.log(sample_allow_entry()).unwrap();
        logger
            .log(sample_deny_entry(
                DenyRule::EndpointDenied,
                "Endpoint disabled",
            ))
            .unwrap();

        assert_eq!(buffer.lock().unwrap().len(), 2);
        assert_eq!(logger.entries().len(), 2);
        assert_eq!(logger.captured_entries().unwrap().len(), 2);

        logger.clear();
        assert_eq!(logger.entries().len(), 0);
        assert_eq!(buffer.lock().unwrap().len(), 0);
    }

    #[test]
    fn test_audit_logger_buffered_helper() {
        let logger = AuditLogger::buffered();
        logger.log(sample_allow_entry()).unwrap();
        assert_eq!(logger.entries().len(), 1);
        logger.clear_captured();
        assert_eq!(logger.entries().len(), 0);
    }

    #[test]
    fn test_format_jsonl_validity_and_rfc3339() {
        let logger = AuditLogger::buffered();
        let entry = sample_allow_entry();
        let jsonl = logger.format_jsonl(&entry).unwrap();

        // Must end with newline and contain exactly one newline
        assert!(jsonl.ends_with('\n'));
        assert_eq!(jsonl.matches('\n').count(), 1);

        // Parse back to Value to verify valid single-line JSON
        let val: serde_json::Value = serde_json::from_str(jsonl.trim()).unwrap();
        assert_eq!(val["decision"], "allow");
        assert_eq!(val["method"], "POST");
        assert_eq!(val["target_endpoint"], "/api/v1/system/status");

        // Verify RFC3339 timestamp parseability
        let ts_str = val["timestamp"].as_str().expect("timestamp string");
        let parsed_ts = chrono::DateTime::parse_from_rfc3339(ts_str);
        assert!(parsed_ts.is_ok());
    }

    #[test]
    fn test_format_jsonl_omits_none_fields() {
        let logger = AuditLogger::buffered();
        let entry = AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /v1/messages".to_string(),
            session_id: None,
            caller_sub: None,
            caller_role: None,
            target_endpoint: None,
            method: None,
            decision: "allow".to_string(),
            matched_rule: None,
            reason: None,
            injection_confidence: None,
            malicious_intent: None,
            latency_ms: 45,
        };

        let jsonl = logger.format_jsonl(&entry).unwrap();
        assert!(!jsonl.contains("session_id"));
        assert!(!jsonl.contains("caller_sub"));
        assert!(!jsonl.contains("caller_role"));
        assert!(!jsonl.contains("target_endpoint"));
        assert!(!jsonl.contains("method"));
        assert!(!jsonl.contains("matched_rule"));
        assert!(!jsonl.contains("reason"));
        assert!(!jsonl.contains("injection_confidence"));
        assert!(!jsonl.contains("malicious_intent"));
    }

    #[test]
    fn test_redact_anthropic_api_key() {
        let logger = AuditLogger::buffered();
        let raw_key = "sk-ant-api03-abcdef1234567890abcdef1234567890abcdef";
        let reason = format!("Failed call with key {}", raw_key);
        let entry = sample_deny_entry(DenyRule::BackendError, &reason);

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let scrubbed_reason = entries[0].reason.as_deref().unwrap();

        assert!(!scrubbed_reason.contains(raw_key));
        assert!(scrubbed_reason.contains("[REDACTED_API_KEY]"));
    }

    #[test]
    fn test_redact_bearer_token() {
        let logger = AuditLogger::buffered();
        let raw_bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.t-IDcSemACt8x4iTMCda8Yhe3iZaWbvV5XKSTbuAn0M";
        let reason = format!("Authorization header rejected: {}", raw_bearer);
        let entry = sample_deny_entry(DenyRule::UnauthorizedToken, &reason);

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let scrubbed = entries[0].reason.as_deref().unwrap();

        assert!(!scrubbed.contains(raw_bearer));
        assert!(scrubbed.contains("Bearer [REDACTED]"));
    }

    #[test]
    fn test_redact_naked_jwt() {
        let logger = AuditLogger::buffered();
        let naked_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUPvptV_80";
        let reason = format!("Naked JWT in diagnostic: {}", naked_jwt);
        let entry = sample_deny_entry(DenyRule::UnauthorizedToken, &reason);

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let scrubbed = entries[0].reason.as_deref().unwrap();

        assert!(!scrubbed.contains(naked_jwt));
        assert!(scrubbed.contains("[REDACTED_JWT]"));
    }

    #[test]
    fn test_redact_64_char_hex_secret() {
        let logger = AuditLogger::buffered();
        let hex_secret = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        let reason = format!("HMAC mismatch: computed {}", hex_secret);
        let entry = sample_deny_entry(DenyRule::UnauthorizedToken, &reason);

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let scrubbed = entries[0].reason.as_deref().unwrap();

        assert!(!scrubbed.contains(hex_secret));
        assert!(scrubbed.contains("[REDACTED_HEX]"));
    }

    #[test]
    fn test_redact_registered_known_secrets() {
        let secret1 = "my-ultra-secret-jwt-signing-key";
        let secret2 = "another-confidential-sidecar-token";
        let logger = AuditLogger::buffered()
            .with_secret(secret1)
            .with_secret(secret2);

        let mut entry = sample_allow_entry();
        entry.session_id = Some(format!("sess-{}", secret1));
        entry.target_endpoint = Some(format!("/api/v1/{}", secret2));

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let logged_session = entries[0].session_id.as_deref().unwrap();
        let logged_endpoint = entries[0].target_endpoint.as_deref().unwrap();

        assert!(!logged_session.contains(secret1));
        assert!(logged_session.contains("[REDACTED]"));

        assert!(!logged_endpoint.contains(secret2));
        assert!(logged_endpoint.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_json_value_sensitive_keys() {
        let mut val = json!({
            "service": "auth",
            "password": "super-secret-password",
            "token": "secret-token-value",
            "nested": {
                "api_key": "sk-ant-12345",
                "normal": "safe-value"
            },
            "array": [
                {"credential": "ssh-private-key"},
                {"public_id": "user-42"}
            ]
        });

        redact_json_value(&mut val);

        assert_eq!(val["service"], "auth");
        assert_eq!(val["password"], "[REDACTED]");
        assert_eq!(val["token"], "[REDACTED]");
        assert_eq!(val["nested"]["api_key"], "[REDACTED]");
        assert_eq!(val["nested"]["normal"], "safe-value");
        assert_eq!(val["array"][0]["credential"], "[REDACTED]");
        assert_eq!(val["array"][1]["public_id"], "user-42");
    }

    #[test]
    fn test_all_deny_rules_serialization() {
        let logger = AuditLogger::buffered();

        let rules = [
            (DenyRule::Malformed, "malformed"),
            (DenyRule::UnauthorizedToken, "unauthorized_token"),
            (DenyRule::EndpointDenied, "endpoint_denied"),
            (DenyRule::UnauthorizedRole, "unauthorized_role"),
            (DenyRule::MissingArg, "missing_arg"),
            (DenyRule::ConstraintFailed, "constraint_failed"),
            (DenyRule::ModelUnavailable, "model_unavailable"),
            (DenyRule::ModelRiskHigh, "model_risk_high"),
            (DenyRule::BackendError, "backend_error"),
        ];

        for (rule, expected_str) in rules {
            let entry = sample_deny_entry(rule, "Denied by test");
            let jsonl = logger.format_jsonl(&entry).unwrap();
            let val: serde_json::Value = serde_json::from_str(jsonl.trim()).unwrap();
            assert_eq!(val["matched_rule"], expected_str);
        }
    }

    #[test]
    fn test_audit_logger_both_sink() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let logger = AuditLogger::both(Arc::clone(&buffer));

        let entry = sample_allow_entry();
        logger.log(entry).unwrap();

        assert_eq!(buffer.lock().unwrap().len(), 1);
        assert_eq!(logger.entries().len(), 1);
    }

    #[test]
    fn test_audit_logger_from_config() {
        let vars = [
            ("SIDECAR_SECRET", "test-sidecar-secret-12345"),
            ("JWT_SECRET", "test-jwt-secret-67890"),
            ("ANTHROPIC_API_KEY", "sk-ant-api03-configured-key-99999"),
        ];
        let cfg = Config::from_vars(vars).unwrap();
        let logger = AuditLogger::from_config(&cfg);

        assert!(logger
            .known_secrets
            .contains(&"test-sidecar-secret-12345".to_string()));
        assert!(logger
            .known_secrets
            .contains(&"test-jwt-secret-67890".to_string()));
        assert!(logger
            .known_secrets
            .contains(&"sk-ant-api03-configured-key-99999".to_string()));
    }

    #[test]
    fn test_redact_multiple_secrets_combined_in_single_entry() {
        let secret = "sidecar-private-key-42";
        let logger = AuditLogger::buffered().with_secret(secret);

        let composite_reason = format!(
            "Failed: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature with sk-ant-api03-1234567890abcdef1234567890abcdef and {} and 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            secret
        );
        let entry = sample_deny_entry(DenyRule::BackendError, &composite_reason);

        logger.log(entry).unwrap();
        let entries = logger.entries();
        let scrubbed = entries[0].reason.as_deref().unwrap();

        assert!(!scrubbed.contains(secret));
        assert!(!scrubbed.contains("sk-ant-api03-1234567890abcdef1234567890abcdef"));
        assert!(
            !scrubbed.contains("0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
        );
        assert!(!scrubbed.contains("Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature"));

        assert!(scrubbed.contains("Bearer [REDACTED]"));
        assert!(scrubbed.contains("[REDACTED_API_KEY]"));
        assert!(scrubbed.contains("[REDACTED]"));
        assert!(scrubbed.contains("[REDACTED_HEX]"));
    }

    #[test]
    fn test_concurrent_thread_safety() {
        let logger = Arc::new(AuditLogger::buffered());
        let mut handles = Vec::new();

        for thread_idx in 0..20 {
            let log_ref = Arc::clone(&logger);
            let handle = std::thread::spawn(move || {
                for i in 0..50 {
                    let mut entry = sample_allow_entry();
                    entry.session_id = Some(format!("thread-{}-req-{}", thread_idx, i));
                    log_ref.log(entry).unwrap();
                }
            });
            handles.push(handle);
        }

        for h in handles {
            h.join().unwrap();
        }

        let total_captured = logger.entries().len();
        assert_eq!(total_captured, 1000);
    }

    #[test]
    fn test_resilience_to_lock_poisoning() {
        let buffer = Arc::new(Mutex::new(Vec::new()));
        let buf_clone = Arc::clone(&buffer);

        // Deliberately poison the mutex by panicking while holding the lock
        let _ = std::panic::catch_unwind(move || {
            let _guard = buf_clone.lock().unwrap();
            panic!("intentional panic to poison mutex");
        });

        assert!(buffer.is_poisoned());

        // AuditLogger should recover cleanly without panicking
        let logger = AuditLogger {
            sink: AuditSink::Memory(buffer),
            known_secrets: Vec::new(),
        };

        let res = logger.log(sample_allow_entry());
        assert!(res.is_ok());
        assert_eq!(logger.entries().len(), 1);
    }

    #[test]
    fn test_audit_logger_register_cli_mapping_secrets() {
        let toml_policy = r#"
        [defaults]
        [agent_cli_mappings.claude]
        env_var = "ANTHROPIC_BASE_URL"
        api_key_env_var = "ANTHROPIC_API_KEY"
        sidecar_url = "http://127.0.0.1:8080"

        [agent_cli_mappings.codex]
        env_var = "OPENAI_BASE_URL"
        api_key_env_var = "OPENAI_API_KEY"
        sidecar_url = "http://127.0.0.1:8080/v1"
        "#;
        let policy = crate::policy::PolicyEngine::from_str(toml_policy).unwrap();

        let mut logger = AuditLogger::buffered();
        let mock_env = HashMap::from([
            (
                "ANTHROPIC_API_KEY".to_string(),
                "sk-ant-secret-123456".to_string(),
            ),
            (
                "OPENAI_API_KEY".to_string(),
                "sk-proj-openai-secret-abcdef".to_string(),
            ),
        ]);

        logger.register_cli_mapping_secrets(&policy, |var| mock_env.get(var).cloned());

        let mut entry = sample_allow_entry();
        entry.reason = Some(
            "Failure connecting with sk-ant-secret-123456 and sk-proj-openai-secret-abcdef"
                .to_string(),
        );
        logger.log(entry).unwrap();

        let logged = logger.entries();
        assert_eq!(logged.len(), 1);
        let logged_reason = logged[0].reason.as_ref().unwrap();
        assert!(!logged_reason.contains("sk-ant-secret-123456"));
        assert!(!logged_reason.contains("sk-proj-openai-secret-abcdef"));
        assert!(logged_reason.contains("[REDACTED]"));
    }

    #[test]
    fn test_redact_value_nested_string_leaves_known_secrets_and_patterns() {
        let known = vec!["registered-custom-secret-999".to_string()];
        let mut val = json!({
            "status": "error",
            "message": "Encountered registered-custom-secret-999 during processing",
            "metadata": {
                "auth_header": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.signature",
                "description": "Key is sk-ant-api03-1234567890abcdef1234567890abcdef here",
                "nested_jwt": "Token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozS6-m_WvQ",
                "nested_hex": "Checksum 0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef matched",
                "safe_info": "safe text"
            },
            "array_items": [
                "plain text",
                "leak: registered-custom-secret-999",
                {"in_array_obj": "Bearer token1234567890"}
            ],
            "password": "should-be-masked-by-key"
        });

        redact_value(&mut val, &known);

        assert_eq!(val["message"], "Encountered [REDACTED] during processing");
        assert_eq!(val["metadata"]["auth_header"], "Bearer [REDACTED]");
        assert_eq!(
            val["metadata"]["description"],
            "Key is [REDACTED_API_KEY] here"
        );
        assert_eq!(val["metadata"]["nested_jwt"], "Token [REDACTED_JWT]");
        assert_eq!(
            val["metadata"]["nested_hex"],
            "Checksum [REDACTED_HEX] matched"
        );
        assert_eq!(val["metadata"]["safe_info"], "safe text");
        assert_eq!(val["array_items"][0], "plain text");
        assert_eq!(val["array_items"][1], "leak: [REDACTED]");
        assert_eq!(val["array_items"][2]["in_array_obj"], "Bearer [REDACTED]");
        assert_eq!(val["password"], "[REDACTED]");
    }
}
