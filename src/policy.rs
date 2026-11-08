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
    AgentClaims, AgentCliMapping, AlertPolicy, ArgConstraint, DefaultPolicy, DenyRule,
    PolicyConfig, RouteAPolicy, ToolExecuteRequest,
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

    #[error("failed to compile linear-time regex '{pattern}' for argument '{arg}' in target '{endpoint}': {source}")]
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
    /// Optional endpoint-specific risk threshold overrides.
    pub max_injection_confidence: Option<f32>,
    pub max_malicious_intent: Option<f32>,
    pub model_timeout_ms: Option<u64>,
}

/// Pre-compiled tool access rule for deterministic CLI tool execution evaluation.
#[derive(Debug, Clone)]
pub struct CompiledToolRule {
    /// Whether the tool is active and callable.
    pub enabled: bool,
    /// Permitted caller roles from JWT claims (e.g. `["agent", "admin"]`).
    pub allowed_roles: Vec<String>,
    /// Required argument keys in tool payload.
    pub required_args: Vec<String>,
    /// Constraints applied per argument key.
    pub constraints: HashMap<String, CompiledArgConstraint>,
    /// Optional tool-specific risk threshold overrides.
    pub max_injection_confidence: Option<f32>,
    pub max_malicious_intent: Option<f32>,
    pub model_timeout_ms: Option<u64>,
}

/// Thread-safe deterministic policy evaluation engine.
///
/// Implements `Send + Sync + Clone` with zero internal locks or mutexes,
/// allowing concurrent, lock-free evaluation across Tokio worker threads.
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    /// Raw deserialized policy configuration (defaults, endpoints, tools, CLI mappings).
    config: PolicyConfig,
    /// Pre-compiled endpoints indexed by target endpoint path for O(1) lookup.
    compiled_endpoints: HashMap<String, CompiledEndpointRule>,
    /// `[route_a]` patterns compiled, for reading new turns.
    turn_rules: TurnRules,
    /// Pre-compiled tools indexed by tool name for O(1) lookup.
    compiled_tools: HashMap<String, CompiledToolRule>,
}

/// Pre-compiles raw argument constraints into linear-time Regex instances.
fn compile_constraints(
    target_name: &str,
    raw_constraints: &HashMap<String, ArgConstraint>,
) -> Result<HashMap<String, CompiledArgConstraint>, PolicyError> {
    let mut compiled = HashMap::with_capacity(raw_constraints.len());
    for (arg_name, constraint) in raw_constraints {
        let compiled_re = match &constraint.regex {
            Some(pat) => {
                let re = Regex::new(pat).map_err(|e| PolicyError::InvalidRegex {
                    endpoint: target_name.to_string(),
                    arg: arg_name.clone(),
                    pattern: pat.clone(),
                    source: e,
                })?;
                Some(re)
            }
            None => None,
        };

        compiled.insert(
            arg_name.clone(),
            CompiledArgConstraint {
                regex: compiled_re,
                max_len: constraint.max_len,
            },
        );
    }
    Ok(compiled)
}

/// Validates payload required arguments and constraints with deterministic key ordering.
fn validate_payload_args_and_constraints(
    payload: &serde_json::Map<String, serde_json::Value>,
    required_args: &[String],
    constraints: &HashMap<String, CompiledArgConstraint>,
) -> Result<(), DenyRule> {
    // 1. Required arguments presence and non-null validation
    for req_arg in required_args {
        match payload.get(req_arg) {
            Some(val) if !val.is_null() => {}
            _ => return Err(DenyRule::MissingArg),
        }
    }

    // 2. Argument constraints validation (sorted keys for deterministic evaluation)
    let mut sorted_keys: Vec<_> = constraints.keys().collect();
    sorted_keys.sort();

    for arg_name in sorted_keys {
        let constraint = &constraints[arg_name];
        if let Some(val) = payload.get(arg_name) {
            if val.is_null() {
                return Err(DenyRule::ConstraintFailed);
            }

            let str_val = val.as_str().ok_or(DenyRule::ConstraintFailed)?;

            // Maximum Unicode scalar character length
            if let Some(max_len) = constraint.max_len {
                if str_val.chars().count() > max_len {
                    return Err(DenyRule::ConstraintFailed);
                }
            }

            // Linear-time regular expression match
            if let Some(re) = &constraint.regex {
                if !re.is_match(str_val) {
                    return Err(DenyRule::ConstraintFailed);
                }
            }
        }
    }

    Ok(())
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

        // 1. Validate operational thresholds (must be bounded in [0.0, 1.0] and !is_nan())
        let max_inj = config.defaults.max_injection_confidence;
        if !(0.0..=1.0).contains(&max_inj) || max_inj.is_nan() {
            return Err(PolicyError::ValidationError(format!(
                "max_injection_confidence must be between 0.0 and 1.0, got {max_inj}"
            )));
        }
        let max_intent = config.defaults.max_malicious_intent;
        if !(0.0..=1.0).contains(&max_intent) || max_intent.is_nan() {
            return Err(PolicyError::ValidationError(format!(
                "max_malicious_intent must be between 0.0 and 1.0, got {max_intent}"
            )));
        }

        // 2. Validate global model timeout [1, 60_000] ms
        let timeout = config.defaults.model_timeout_ms;
        if !(1..=60_000).contains(&timeout) {
            return Err(PolicyError::ValidationError(format!(
                "model_timeout_ms must be between 1 and 60000, got {timeout}"
            )));
        }

        // 2b. Route A settings: bounded context, regexes compiled once
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
            include_system_prompt: route_a.include_system_prompt,
            context_chars: route_a.context_chars,
        };

        // 3. Compile and validate endpoints
        let mut compiled_endpoints = HashMap::with_capacity(config.endpoints.len());

        for (endpoint_path, endpoint_rule) in &config.endpoints {
            // Validate endpoint path format
            if endpoint_path.trim().is_empty() || !endpoint_path.starts_with('/') {
                return Err(PolicyError::ValidationError(format!(
                    "endpoint path '{endpoint_path}' must start with '/' and not be empty"
                )));
            }

            // Validate backend URL using reqwest::Url::parse
            let parsed_url = reqwest::Url::parse(&endpoint_rule.backend_url).map_err(|e| {
                PolicyError::ValidationError(format!(
                    "invalid backend_url '{}' for endpoint '{endpoint_path}': {e}",
                    endpoint_rule.backend_url
                ))
            })?;

            if parsed_url.scheme() != "http" && parsed_url.scheme() != "https" {
                return Err(PolicyError::ValidationError(format!(
                    "backend_url '{}' for endpoint '{endpoint_path}' must use http or https scheme, got '{}'",
                    endpoint_rule.backend_url,
                    parsed_url.scheme()
                )));
            }

            if parsed_url.host_str().is_none_or(|h| h.is_empty()) {
                return Err(PolicyError::ValidationError(format!(
                    "backend_url '{}' for endpoint '{endpoint_path}' must have a valid non-empty host",
                    endpoint_rule.backend_url
                )));
            }

            // Validate per-endpoint threshold overrides
            if let Some(inj) = endpoint_rule.max_injection_confidence {
                if !(0.0..=1.0).contains(&inj) || inj.is_nan() {
                    return Err(PolicyError::ValidationError(format!(
                        "max_injection_confidence for endpoint '{endpoint_path}' must be between 0.0 and 1.0, got {inj}"
                    )));
                }
            }

            if let Some(intent) = endpoint_rule.max_malicious_intent {
                if !(0.0..=1.0).contains(&intent) || intent.is_nan() {
                    return Err(PolicyError::ValidationError(format!(
                        "max_malicious_intent for endpoint '{endpoint_path}' must be between 0.0 and 1.0, got {intent}"
                    )));
                }
            }

            if let Some(t) = endpoint_rule.model_timeout_ms {
                if !(1..=60_000).contains(&t) {
                    return Err(PolicyError::ValidationError(format!(
                        "model_timeout_ms for endpoint '{endpoint_path}' must be between 1 and 60000, got {t}"
                    )));
                }
            }

            let compiled_constraints =
                compile_constraints(endpoint_path, &endpoint_rule.constraints)?;

            compiled_endpoints.insert(
                endpoint_path.clone(),
                CompiledEndpointRule {
                    backend_url: endpoint_rule.backend_url.clone(),
                    enabled: endpoint_rule.enabled,
                    allowed_methods: endpoint_rule.allowed_methods.clone(),
                    allowed_roles: endpoint_rule.allowed_roles.clone(),
                    required_args: endpoint_rule.required_args.clone(),
                    constraints: compiled_constraints,
                    max_injection_confidence: endpoint_rule.max_injection_confidence,
                    max_malicious_intent: endpoint_rule.max_malicious_intent,
                    model_timeout_ms: endpoint_rule.model_timeout_ms,
                },
            );
        }

        // 4. Compile and validate tool rules
        let mut compiled_tools = HashMap::with_capacity(config.tools.len());

        for (tool_name, tool_rule) in &config.tools {
            if tool_name.trim().is_empty() {
                return Err(PolicyError::ValidationError(
                    "tool name in [tools] must not be empty".to_string(),
                ));
            }

            // Validate per-tool threshold overrides
            if let Some(inj) = tool_rule.max_injection_confidence {
                if !(0.0..=1.0).contains(&inj) || inj.is_nan() {
                    return Err(PolicyError::ValidationError(format!(
                        "max_injection_confidence for tool '{tool_name}' must be between 0.0 and 1.0, got {inj}"
                    )));
                }
            }

            if let Some(intent) = tool_rule.max_malicious_intent {
                if !(0.0..=1.0).contains(&intent) || intent.is_nan() {
                    return Err(PolicyError::ValidationError(format!(
                        "max_malicious_intent for tool '{tool_name}' must be between 0.0 and 1.0, got {intent}"
                    )));
                }
            }

            if let Some(t) = tool_rule.model_timeout_ms {
                if !(1..=60_000).contains(&t) {
                    return Err(PolicyError::ValidationError(format!(
                        "model_timeout_ms for tool '{tool_name}' must be between 1 and 60000, got {t}"
                    )));
                }
            }

            let tool_target_name = format!("tool:{tool_name}");
            let compiled_constraints =
                compile_constraints(&tool_target_name, &tool_rule.constraints)?;

            compiled_tools.insert(
                tool_name.clone(),
                CompiledToolRule {
                    enabled: tool_rule.enabled,
                    allowed_roles: tool_rule.allowed_roles.clone(),
                    required_args: tool_rule.required_args.clone(),
                    constraints: compiled_constraints,
                    max_injection_confidence: tool_rule.max_injection_confidence,
                    max_malicious_intent: tool_rule.max_malicious_intent,
                    model_timeout_ms: tool_rule.model_timeout_ms,
                },
            );
        }

        Ok(Self {
            config,
            compiled_endpoints,
            turn_rules,
            compiled_tools,
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

    /// Retrieves pre-compiled tool rule by exact name, falling back to case-insensitive match.
    pub fn get_tool(&self, tool: &str) -> Option<&CompiledToolRule> {
        self.compiled_tools.get(tool).or_else(|| {
            self.compiled_tools
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(tool))
                .map(|(_, v)| v)
        })
    }

    /// Resolves effective (max_injection_confidence, max_malicious_intent) thresholds for an endpoint.
    /// Falls back to global defaults if unlisted or override not specified.
    pub fn effective_thresholds_for_endpoint(&self, endpoint: &str) -> (f32, f32) {
        if let Some(rule) = self.compiled_endpoints.get(endpoint) {
            let inj = rule
                .max_injection_confidence
                .unwrap_or(self.config.defaults.max_injection_confidence);
            let intent = rule
                .max_malicious_intent
                .unwrap_or(self.config.defaults.max_malicious_intent);
            (inj, intent)
        } else {
            (
                self.config.defaults.max_injection_confidence,
                self.config.defaults.max_malicious_intent,
            )
        }
    }

    /// Resolves effective model timeout in milliseconds for an endpoint.
    /// Falls back to global defaults if unlisted or override not specified.
    pub fn effective_timeout_for_endpoint(&self, endpoint: &str) -> u64 {
        self.compiled_endpoints
            .get(endpoint)
            .and_then(|rule| rule.model_timeout_ms)
            .unwrap_or(self.config.defaults.model_timeout_ms)
    }

    /// Resolves effective (max_injection_confidence, max_malicious_intent) thresholds for a tool.
    /// Falls back to global defaults if unlisted or override not specified.
    pub fn effective_thresholds_for_tool(&self, tool: &str) -> (f32, f32) {
        if let Some(rule) = self.get_tool(tool) {
            let inj = rule
                .max_injection_confidence
                .unwrap_or(self.config.defaults.max_injection_confidence);
            let intent = rule
                .max_malicious_intent
                .unwrap_or(self.config.defaults.max_malicious_intent);
            (inj, intent)
        } else {
            (
                self.config.defaults.max_injection_confidence,
                self.config.defaults.max_malicious_intent,
            )
        }
    }

    /// Resolves effective model timeout in milliseconds for a tool.
    /// Falls back to global defaults if unlisted or override not specified.
    pub fn effective_timeout_for_tool(&self, tool: &str) -> u64 {
        self.get_tool(tool)
            .and_then(|rule| rule.model_timeout_ms)
            .unwrap_or(self.config.defaults.model_timeout_ms)
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

        // Step 4 & 5: Required arguments and constraints validation
        validate_payload_args_and_constraints(
            &req.payload,
            &endpoint_rule.required_args,
            &endpoint_rule.constraints,
        )
    }

    /// Evaluates local CLI tool execution against deterministic tool policy rules.
    ///
    /// Evaluation Priority Hierarchy:
    /// 1. Tool allowlist & `enabled == true` -> `DenyRule::EndpointDenied`
    /// 2. Caller role in `allowed_roles` -> `DenyRule::UnauthorizedRole`
    /// 3. Required arguments non-null & present in payload -> `DenyRule::MissingArg`
    /// 4. Argument constraints (max_len & linear-time regex) satisfied -> `DenyRule::ConstraintFailed`
    pub fn evaluate_tool(
        &self,
        claims: &AgentClaims,
        tool_name: &str,
        payload: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<(), DenyRule> {
        // Step 1: Tool allowlist and enabled check
        let tool_rule = self.get_tool(tool_name).ok_or(DenyRule::EndpointDenied)?;

        if !tool_rule.enabled {
            return Err(DenyRule::EndpointDenied);
        }

        // Step 2: Caller role authorization against allowed_roles
        let role_authorized = tool_rule.allowed_roles.iter().any(|r| r == &claims.role);
        if !role_authorized {
            return Err(DenyRule::UnauthorizedRole);
        }

        // Step 3 & 4: Required arguments and constraints validation
        validate_payload_args_and_constraints(
            payload,
            &tool_rule.required_args,
            &tool_rule.constraints,
        )
    }

    /// Convenience wrapper evaluating a tool execution where payload is provided as `serde_json::Value`.
    /// Fails closed with `DenyRule::Malformed` if payload is not a JSON object.
    pub fn evaluate_tool_value(
        &self,
        claims: &AgentClaims,
        tool_name: &str,
        payload_val: &serde_json::Value,
    ) -> Result<(), DenyRule> {
        let payload_map = payload_val.as_object().ok_or(DenyRule::Malformed)?;
        self.evaluate_tool(claims, tool_name, payload_map)
    }

    /// Returns a reference to all configured agent CLI mappings.
    pub fn cli_mappings(&self) -> &HashMap<String, AgentCliMapping> {
        &self.config.agent_cli_mappings
    }

    /// Retrieves an individual CLI mapping by name (e.g. "claude", "codex", "ollama").
    pub fn get_cli_mapping(&self, cli_name: &str) -> Option<&AgentCliMapping> {
        self.config.agent_cli_mappings.get(cli_name)
    }

    /// Generates the process environment variable map needed to configure a target agent CLI.
    pub fn get_cli_env(&self, cli_name: &str) -> Option<HashMap<String, String>> {
        let mapping = self.config.agent_cli_mappings.get(cli_name)?;
        let mut env = HashMap::new();
        env.insert(mapping.env_var.clone(), mapping.sidecar_url.clone());
        for (k, v) in &mapping.extra_env {
            env.insert(k.clone(), v.clone());
        }
        Some(env)
    }

    /// Returns sorted list of all configured CLI mapping identifiers.
    pub fn configured_clis(&self) -> Vec<String> {
        let mut names: Vec<_> = self.config.agent_cli_mappings.keys().cloned().collect();
        names.sort();
        names
    }

    /// Environment that sends `cli`'s traffic through the sidecar, base URL variable first
    /// (`zero-trust-agentic-sidecar env <cli>`, used by `scripts/agent.sh`). Refuses CLIs whose
    /// protocol the sidecar does not serve yet, and names that are not valid variables.
    pub fn agent_env(&self, cli: &str) -> Result<Vec<(String, String)>, String> {
        let mapping = self.get_cli_mapping(cli).ok_or_else(|| {
            format!(
                "no [agent_cli_mappings.{cli}] in the policy (configured: {})",
                self.configured_clis().join(", ")
            )
        })?;
        if mapping.protocol != "anthropic" {
            return Err(format!(
                "{cli} speaks the '{}' protocol, which the sidecar does not serve yet \
                 (only 'anthropic'); its traffic would not be inspected",
                mapping.protocol
            ));
        }
        let mut extra: Vec<(String, String)> = mapping
            .extra_env
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        extra.sort();
        let vars: Vec<(String, String)> =
            std::iter::once((mapping.env_var.clone(), mapping.sidecar_url.clone()))
                .chain(extra)
                .collect();
        for (key, value) in &vars {
            let valid_key = !key.is_empty()
                && !key.starts_with(|c: char| c.is_ascii_digit())
                && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
            if !valid_key || value.contains(['\n', '\r', '\0']) {
                return Err(format!(
                    "[agent_cli_mappings.{cli}]: invalid variable '{key}'"
                ));
            }
        }
        Ok(vars)
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

    #[test]
    fn test_validation_model_timeout_bounds() {
        // Zero timeout rejected
        let bad_zero = r#"
        [defaults]
        model_timeout_ms = 0
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_zero),
            Err(PolicyError::ValidationError(_))
        ));

        // Timeout > 60_000 rejected
        let bad_large = r#"
        [defaults]
        model_timeout_ms = 60001
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_large),
            Err(PolicyError::ValidationError(_))
        ));

        // Valid timeout in bounds accepted
        let ok_timeout = r#"
        [defaults]
        model_timeout_ms = 2500
        "#;
        assert!(PolicyEngine::from_str(ok_timeout).is_ok());
    }

    #[test]
    fn test_validation_nan_threshold_rejected() {
        let bad_nan = r#"
        [defaults]
        max_injection_confidence = nan
        "#;
        assert!(PolicyEngine::from_str(bad_nan).is_err());
    }

    #[test]
    fn test_validation_endpoint_path_requirements() {
        // Missing leading slash rejected
        let bad_path = r#"
        [defaults]
        [endpoints."api/v1/status"]
        backend_url = "http://127.0.0.1:5000/api/v1/status"
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_path),
            Err(PolicyError::ValidationError(_))
        ));

        // Empty endpoint path rejected
        let bad_empty = r#"
        [defaults]
        [endpoints.""]
        backend_url = "http://127.0.0.1:5000"
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_empty),
            Err(PolicyError::ValidationError(_))
        ));
    }

    #[test]
    fn test_validation_backend_url_format_and_scheme() {
        // Unparseable URL rejected
        let bad_url = r#"
        [defaults]
        [endpoints."/api/status"]
        backend_url = "not a valid url"
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_url),
            Err(PolicyError::ValidationError(_))
        ));

        // Non-http/https scheme rejected (e.g. ftp)
        let bad_scheme = r#"
        [defaults]
        [endpoints."/api/status"]
        backend_url = "ftp://127.0.0.1:5000/api/status"
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_scheme),
            Err(PolicyError::ValidationError(_))
        ));

        // Missing host rejected
        let bad_host = r#"
        [defaults]
        [endpoints."/api/status"]
        backend_url = "http://:8080/path"
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_host),
            Err(PolicyError::ValidationError(_))
        ));
    }

    #[test]
    fn test_validation_endpoint_threshold_overrides_bounds() {
        let bad_override = r#"
        [defaults]
        [endpoints."/api/v1/test"]
        backend_url = "http://127.0.0.1:5000/api/v1/test"
        max_malicious_intent = 1.2
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_override),
            Err(PolicyError::ValidationError(_))
        ));

        let bad_override_timeout = r#"
        [defaults]
        [endpoints."/api/v1/test"]
        backend_url = "http://127.0.0.1:5000/api/v1/test"
        model_timeout_ms = 0
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_override_timeout),
            Err(PolicyError::ValidationError(_))
        ));
    }

    #[test]
    fn test_validation_tool_rules_bounds_and_regex() {
        // Empty tool name rejected
        let bad_tool_name = r#"
        [defaults]
        [tools." "]
        enabled = true
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_tool_name),
            Err(PolicyError::ValidationError(_))
        ));

        // Invalid tool threshold rejected
        let bad_tool_intent = r#"
        [defaults]
        [tools.bash]
        max_malicious_intent = -0.5
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_tool_intent),
            Err(PolicyError::ValidationError(_))
        ));

        // Invalid tool regex fails fast
        let bad_tool_regex = r#"
        [defaults]
        [tools.bash]
        constraints.command = { regex = "([a-z" }
        "#;
        assert!(matches!(
            PolicyEngine::from_str(bad_tool_regex),
            Err(PolicyError::InvalidRegex { .. })
        ));
    }

    #[test]
    fn test_effective_thresholds_resolution() {
        let toml_policy = r#"
        [defaults]
        max_injection_confidence = 0.70
        max_malicious_intent = 0.50
        model_timeout_ms = 500

        [endpoints."/api/v1/standard"]
        backend_url = "http://127.0.0.1:5000/api/v1/standard"

        [endpoints."/api/v1/sensitive"]
        backend_url = "http://127.0.0.1:5000/api/v1/sensitive"
        max_malicious_intent = 0.30
        model_timeout_ms = 1200

        [tools."bash"]
        max_injection_confidence = 0.60
        max_malicious_intent = 0.35
        model_timeout_ms = 800

        [tools."read_file"]
        enabled = true
        "#;

        let engine = PolicyEngine::from_str(toml_policy).unwrap();

        // Standard endpoint falls back to defaults
        assert_eq!(
            engine.effective_thresholds_for_endpoint("/api/v1/standard"),
            (0.70, 0.50)
        );
        assert_eq!(
            engine.effective_timeout_for_endpoint("/api/v1/standard"),
            500
        );

        // Sensitive endpoint overrides intent and timeout
        assert_eq!(
            engine.effective_thresholds_for_endpoint("/api/v1/sensitive"),
            (0.70, 0.30)
        );
        assert_eq!(
            engine.effective_timeout_for_endpoint("/api/v1/sensitive"),
            1200
        );

        // Unlisted endpoint falls back to defaults
        assert_eq!(
            engine.effective_thresholds_for_endpoint("/api/v1/unlisted"),
            (0.70, 0.50)
        );
        assert_eq!(
            engine.effective_timeout_for_endpoint("/api/v1/unlisted"),
            500
        );

        // Bash tool overrides injection, intent, and timeout
        assert_eq!(engine.effective_thresholds_for_tool("bash"), (0.60, 0.35));
        assert_eq!(engine.effective_timeout_for_tool("bash"), 800);

        // read_file tool falls back to defaults
        assert_eq!(
            engine.effective_thresholds_for_tool("read_file"),
            (0.70, 0.50)
        );
        assert_eq!(engine.effective_timeout_for_tool("read_file"), 500);

        // Case-insensitive tool lookup
        assert_eq!(engine.effective_thresholds_for_tool("BASH"), (0.60, 0.35));
    }

    #[test]
    fn test_evaluate_tool_priority_and_rules() {
        let toml_policy = r#"
        [defaults]

        [tools."bash"]
        enabled = true
        allowed_roles = ["admin"]
        required_args = ["command"]
        constraints.command = { regex = "^[a-zA-Z0-9_./ -]+$", max_len = 100 }

        [tools."disabled_tool"]
        enabled = false
        allowed_roles = ["admin"]
        "#;

        let engine = PolicyEngine::from_str(toml_policy).unwrap();
        let claims_admin = test_claims("admin");
        let claims_agent = test_claims("agent");

        // Priority 1: Unlisted or disabled tool -> EndpointDenied
        let res_unlisted = engine.evaluate_tool(
            &claims_admin,
            "nonexistent",
            json!({"command": "ls"}).as_object().unwrap(),
        );
        assert_eq!(res_unlisted, Err(DenyRule::EndpointDenied));

        let res_disabled = engine.evaluate_tool(
            &claims_admin,
            "disabled_tool",
            json!({}).as_object().unwrap(),
        );
        assert_eq!(res_disabled, Err(DenyRule::EndpointDenied));

        // Priority 2: Unauthorized role -> UnauthorizedRole
        let res_unauth = engine.evaluate_tool(
            &claims_agent,
            "bash",
            json!({"command": "ls"}).as_object().unwrap(),
        );
        assert_eq!(res_unauth, Err(DenyRule::UnauthorizedRole));

        // Priority 3: Missing required argument -> MissingArg
        let res_missing =
            engine.evaluate_tool(&claims_admin, "bash", json!({}).as_object().unwrap());
        assert_eq!(res_missing, Err(DenyRule::MissingArg));

        let res_null_arg = engine.evaluate_tool(
            &claims_admin,
            "bash",
            json!({"command": null}).as_object().unwrap(),
        );
        assert_eq!(res_null_arg, Err(DenyRule::MissingArg));

        // Priority 4: Constraint failed (regex) -> ConstraintFailed
        let res_bad_regex = engine.evaluate_tool(
            &claims_admin,
            "bash",
            json!({"command": "cat secret; rm -rf /"})
                .as_object()
                .unwrap(),
        );
        assert_eq!(res_bad_regex, Err(DenyRule::ConstraintFailed));

        // Priority 4: Constraint failed (length) -> ConstraintFailed
        let long_cmd = "a".repeat(101);
        let res_too_long = engine.evaluate_tool(
            &claims_admin,
            "bash",
            json!({"command": long_cmd}).as_object().unwrap(),
        );
        assert_eq!(res_too_long, Err(DenyRule::ConstraintFailed));

        // Approved tool call -> Ok(())
        let res_ok = engine.evaluate_tool(
            &claims_admin,
            "bash",
            json!({"command": "ls -la"}).as_object().unwrap(),
        );
        assert_eq!(res_ok, Ok(()));

        // Case-insensitive tool name matching -> Ok(())
        let res_ci = engine.evaluate_tool(
            &claims_admin,
            "BASH",
            json!({"command": "pwd"}).as_object().unwrap(),
        );
        assert_eq!(res_ci, Ok(()));
    }

    #[test]
    fn test_evaluate_tool_value_and_malformed() {
        let toml_policy = r#"
        [defaults]
        [tools."bash"]
        enabled = true
        allowed_roles = ["admin"]
        required_args = ["command"]
        "#;
        let engine = PolicyEngine::from_str(toml_policy).unwrap();
        let claims = test_claims("admin");

        // Non-object payload -> Malformed
        let bad_payload = json!("not an object");
        assert_eq!(
            engine.evaluate_tool_value(&claims, "bash", &bad_payload),
            Err(DenyRule::Malformed)
        );

        // Valid object payload -> Ok
        let good_payload = json!({"command": "uptime"});
        assert_eq!(
            engine.evaluate_tool_value(&claims, "bash", &good_payload),
            Ok(())
        );
    }

    #[test]
    fn test_cli_mappings_accessors() {
        let toml_policy = r#"
        [defaults]

        [agent_cli_mappings.claude]
        env_var = "ANTHROPIC_BASE_URL"
        api_key_env_var = "ANTHROPIC_API_KEY"
        sidecar_url = "http://127.0.0.1:8080"
        upstream_url = "https://api.anthropic.com/v1/messages"
        protocol = "anthropic"
        extra_env = { CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = "1" }

        [agent_cli_mappings.ollama]
        env_var = "OLLAMA_HOST"
        sidecar_url = "127.0.0.1:8080"
        "#;

        let engine = PolicyEngine::from_str(toml_policy).unwrap();
        assert_eq!(engine.cli_mappings().len(), 2);
        assert_eq!(engine.configured_clis(), vec!["claude", "ollama"]);

        let claude = engine.get_cli_mapping("claude").unwrap();
        assert_eq!(claude.env_var, "ANTHROPIC_BASE_URL");
        assert_eq!(claude.api_key_env_var.as_deref(), Some("ANTHROPIC_API_KEY"));

        let env = engine.get_cli_env("claude").unwrap();
        assert_eq!(
            env.get("ANTHROPIC_BASE_URL").unwrap(),
            "http://127.0.0.1:8080"
        );
        assert_eq!(
            env.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC").unwrap(),
            "1"
        );

        assert!(engine.get_cli_mapping("nonexistent").is_none());
        assert!(engine.get_cli_env("nonexistent").is_none());
    }

    #[test]
    fn agent_env_serves_only_anthropic_clis_and_puts_the_base_url_first() {
        let engine =
            PolicyEngine::from_file(concat!(env!("CARGO_MANIFEST_DIR"), "/policy.toml")).unwrap();
        let env = engine.agent_env("claude").unwrap();
        assert_eq!(
            env,
            vec![
                (
                    "ANTHROPIC_BASE_URL".to_string(),
                    "http://127.0.0.1:8080".to_string()
                ),
                (
                    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".to_string(),
                    "1".to_string()
                ),
            ]
        );
        let err = engine.agent_env("codex").unwrap_err();
        assert!(err.contains("'openai' protocol"), "{err}");
        let err = engine.agent_env("cursor").unwrap_err();
        assert!(err.contains("configured: claude, codex, ollama"), "{err}");

        let bad = PolicyEngine::from_str(
            r#"
            [defaults]
            [agent_cli_mappings.x]
            env_var = "BAD;rm"
            sidecar_url = "http://127.0.0.1:8080"
            "#,
        )
        .unwrap();
        assert!(bad.agent_env("x").unwrap_err().contains("invalid variable"));
    }
}
