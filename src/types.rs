//! Core Data Transfer Objects, Serde Schemas & Protocol Enums (`src/types.rs`)
//!
//! Provides all wire types for Route A (`POST /v1/messages`), Route B (`POST /api/execute`),
//! policy definitions, Kev ML verdicts, JWT claims, and structured audit logs.

use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ==============================================================================
// 1. Verdicts & Decision Rules
// ==============================================================================

/// Calibrated verdict emitted by Kev System One ML inference.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KevVerdict {
    /// Inferred probability that the input contains a prompt injection (0.0 to 1.0).
    pub injection_confidence: f32,
    /// Inferred probability that the input exhibits malicious intent (0.0 to 1.0).
    pub malicious_intent: f32,
    /// Model identifier or adapter version that generated this verdict.
    pub model_id: String,
    /// Human-readable explanation or diagnostic reason.
    pub reason: String,
}

impl KevVerdict {
    /// Constructs a new KevVerdict with explicit scores and metadata.
    pub fn new(
        injection_confidence: f32,
        malicious_intent: f32,
        model_id: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            injection_confidence,
            malicious_intent,
            model_id: model_id.into(),
            reason: reason.into(),
        }
    }

    /// Factory method for a clean, baseline-safe verdict.
    pub fn safe(model_id: impl Into<String>) -> Self {
        Self::new(0.01, 0.01, model_id, "clean evaluation")
    }

    /// Evaluates whether this verdict satisfies safe operational thresholds.
    pub fn is_safe(&self, max_injection: f32, max_intent: f32) -> bool {
        self.injection_confidence < max_injection && self.malicious_intent < max_intent
    }
}

/// Compact float scores extracted from a KevVerdict.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub struct KevScores {
    pub injection_confidence: f32,
    pub malicious_intent: f32,
}

impl From<&KevVerdict> for KevScores {
    fn from(v: &KevVerdict) -> Self {
        Self {
            injection_confidence: v.injection_confidence,
            malicious_intent: v.malicious_intent,
        }
    }
}

/// Priority decision rules enforced during request authorization.
/// Evaluated in strict priority order (Malformed -> Auth -> Policy -> ML -> Backend).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DenyRule {
    /// Unparseable JSON, invalid schema, or malformed body.
    Malformed,
    /// Missing, expired, untrusted, or tampered JWT bearer token.
    UnauthorizedToken,
    /// Target endpoint unlisted or explicitly disabled in policy.toml.
    EndpointDenied,
    /// Caller role (from JWT claims) is not permitted for the endpoint.
    UnauthorizedRole,
    /// A required argument specified in policy.toml is absent from payload.
    MissingArg,
    /// Payload argument failed regex pattern or maximum length constraint.
    ConstraintFailed,
    /// ML engine timed out or suffered an unrecoverable failure.
    ModelUnavailable,
    /// ML injection confidence or malicious intent exceeded policy threshold.
    ModelRiskHigh,
    /// Downstream Flask backend returned an error or failed to respond.
    BackendError,
}

impl DenyRule {
    /// Returns the canonical snake_case string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::UnauthorizedToken => "unauthorized_token",
            Self::EndpointDenied => "endpoint_denied",
            Self::UnauthorizedRole => "unauthorized_role",
            Self::MissingArg => "missing_arg",
            Self::ConstraintFailed => "constraint_failed",
            Self::ModelUnavailable => "model_unavailable",
            Self::ModelRiskHigh => "model_risk_high",
            Self::BackendError => "backend_error",
        }
    }

    /// Maps the deny rule to its standard HTTP response status code.
    pub fn http_status(&self) -> StatusCode {
        match self {
            Self::Malformed => StatusCode::BAD_REQUEST,
            Self::UnauthorizedToken => StatusCode::FORBIDDEN,
            Self::EndpointDenied => StatusCode::FORBIDDEN,
            Self::UnauthorizedRole => StatusCode::FORBIDDEN,
            Self::MissingArg => StatusCode::FORBIDDEN,
            Self::ConstraintFailed => StatusCode::FORBIDDEN,
            Self::ModelUnavailable => StatusCode::FORBIDDEN,
            Self::ModelRiskHigh => StatusCode::FORBIDDEN,
            Self::BackendError => StatusCode::BAD_GATEWAY,
        }
    }
}

impl fmt::Display for DenyRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ==============================================================================
// 2. Route B: Outbound Tool Interception (`POST /api/execute`)
// ==============================================================================

fn default_post_method() -> String {
    "POST".to_string()
}

/// Context accompanying tool execution requests, capturing conversation state.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolContext {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_prompt: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub untrusted_inputs: Vec<String>,
}

pub type ExecutionContext = ToolContext;

/// Incoming request payload for Route B (`POST /api/execute`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolExecuteRequest {
    /// Unique session or trace identifier.
    pub session_id: String,
    /// Target path (e.g. `/api/v1/system/status`) or local tool name (e.g. `bash`).
    #[serde(alias = "tool", alias = "tool_name", alias = "name")]
    pub endpoint: String,
    /// Target HTTP method, defaults to `"POST"`.
    #[serde(default = "default_post_method")]
    pub method: String,
    /// Dynamic JSON payload forwarded to target microservice or tool parameters.
    #[serde(default, alias = "args", alias = "arguments", alias = "parameters")]
    pub payload: serde_json::Map<String, serde_json::Value>,
    /// Security context including user prompts and untrusted input logs.
    #[serde(default)]
    pub context: ToolContext,
}

impl ToolExecuteRequest {
    /// Resolves target tool name if this request targets a tool, or None if it targets an endpoint.
    pub fn resolved_tool_name<'a>(
        &'a self,
        engine: &'a crate::policy::PolicyEngine,
    ) -> Option<&'a str> {
        if engine.get_tool(&self.endpoint).is_some() {
            return Some(&self.endpoint);
        }
        if let Some(stripped) = self.endpoint.strip_prefix("tool:") {
            if engine.get_tool(stripped).is_some() {
                return Some(stripped);
            }
        }
        if let Some(stripped) = self.endpoint.strip_prefix("/tools/") {
            if engine.get_tool(stripped).is_some() {
                return Some(stripped);
            }
        }
        None
    }
}

/// Success response returned when a tool call is approved and executed.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteBAllowResponse {
    pub decision: String, // "allow"
    pub endpoint: String,
    pub kev_scores: KevScores,
    pub backend_response: serde_json::Value,
}

/// Rejection response returned when a tool call is denied by policy or ML.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteBDenyResponse {
    pub decision: String, // "deny"
    pub rule: DenyRule,
    pub reason: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kev_scores: Option<KevScores>,
}

// ==============================================================================
// 3. Route A: Inbound Messages Interception (`POST /v1/messages`)
// ==============================================================================

/// Wire request schema for Anthropic Messages API proxying.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicMessagesRequest {
    pub model: String,
    pub messages: Vec<AnthropicMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
}

impl AnthropicMessagesRequest {
    /// Flattens system prompt and all message/tool_result text blocks into a single string for ML scoring.
    pub fn extract_all_text(&self) -> String {
        let mut out = String::new();
        if let Some(sys) = &self.system {
            out.push_str(sys);
            out.push('\n');
        }
        for msg in &self.messages {
            match &msg.content {
                MessageContent::Text(text) => {
                    out.push_str(text);
                    out.push('\n');
                }
                MessageContent::Blocks(blocks) => {
                    for b in blocks {
                        if let Some(t) = &b.text {
                            out.push_str(t);
                            out.push('\n');
                        } else if let Some(serde_json::Value::String(s)) = &b.content {
                            out.push_str(s);
                            out.push('\n');
                        }
                    }
                }
            }
        }
        out
    }
}

/// Single conversational message in an Anthropic messages request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: MessageContent,
}

/// Content payload which may be either a raw string or an array of structured blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

/// Individual block within structured message content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<serde_json::Value>,
}

/// Anthropic-compatible error envelope returned when Route A drops a prompt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteAErrorResponse {
    #[serde(rename = "type")]
    pub response_type: String, // "error"
    pub error: RouteAErrorDetail,
}

/// Detailed error payload embedded inside RouteAErrorResponse.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RouteAErrorDetail {
    #[serde(rename = "type")]
    pub error_type: String, // "zero_trust_policy_violation"
    pub rule: String, // "model_risk_high"
    pub message: String,
    pub injection_confidence: f32,
    pub malicious_intent: f32,
}

// ==============================================================================
// 4. Zero-Trust Audit Logging Struct
// ==============================================================================

/// Structured JSONL audit log entry emitted for every processed request.
/// Strictly excludes authentication secrets and raw sensitive payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AuditEntry {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub route: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller_sub: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub caller_role: Option<String>,
    #[serde(alias = "endpoint", skip_serializing_if = "Option::is_none")]
    pub target_endpoint: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub method: Option<String>,
    pub decision: String, // "allow" or "deny"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matched_rule: Option<DenyRule>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub injection_confidence: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub malicious_intent: Option<f32>,
    pub latency_ms: u64,
}

// ==============================================================================
// 5. Inbound Agent JWT Claims
// ==============================================================================

/// Strongly typed claims contained within inbound HS256 agent bearer tokens.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AgentClaims {
    /// Subject identifier representing the caller identity (e.g. "agent-007").
    pub sub: String,
    /// Authorization role (e.g. "agent", "admin", "readonly") matched against policy.
    pub role: String,
    /// Expiration timestamp in Unix epoch seconds.
    pub exp: usize,
    /// Optional SHA-256 hex digest of the bound request body payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_sha256: Option<String>,
}

// ==============================================================================
// 6. Policy TOML Configuration Schemas
// ==============================================================================

fn default_max_injection() -> f32 {
    0.70
}

fn default_max_intent() -> f32 {
    0.50
}

fn default_model_timeout() -> u64 {
    500
}

fn default_true() -> bool {
    true
}

fn default_protocol() -> String {
    "anthropic".to_string()
}

/// Root schema deserialized from `policy.toml`.
///
/// Enforces fail-closed deserialization: any unrecognized key or malformed table
/// will cause immediate deserialization failure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    pub defaults: DefaultPolicy,
    #[serde(default)]
    pub endpoints: HashMap<String, EndpointRule>,
    /// Local CLI tools checked on Route B (`[tools."bash"]`, ...).
    #[serde(default)]
    pub tools: HashMap<String, ToolRule>,
    /// Environment each agent CLI needs to send its traffic through the sidecar.
    #[serde(default)]
    pub agent_cli_mappings: HashMap<String, AgentCliMapping>,
    /// Route A (`/v1/messages`) scoring and blocking behaviour. Optional section.
    #[serde(default)]
    pub route_a: RouteAPolicy,
    /// How the human is told about blocks. Optional section.
    #[serde(default)]
    pub alerts: AlertPolicy,
}

/// Whether a decision blocks or is only reported.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum EnforcementMode {
    /// Block requests that cross a threshold.
    #[default]
    Enforce,
    /// Never block: log, alert and forward. For tuning thresholds on real traffic.
    Monitor,
}

/// Which part of a `/v1/messages` request is scored.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum ScoreScope {
    /// System prompt and every message, every turn (strict, slow on long sessions).
    All,
    /// Only what arrived since the model's last turn: new tool results and user text.
    #[default]
    New,
}

/// What the agent receives when Route A blocks.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase")]
pub enum BlockResponse {
    /// HTTP 403 with an Anthropic error body: the agent's turn fails.
    #[default]
    Error,
    /// HTTP 200 with an assistant message explaining the block: the agent shows it and stops.
    Message,
}

/// `[route_a]` section of `policy.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct RouteAPolicy {
    pub mode: EnforcementMode,
    pub score: ScoreScope,
    /// With `score = "new"`: also score text the human typed in the new turn.
    pub include_user_text: bool,
    /// With `score = "new"`: also score the system prompt, every turn (it is resent every
    /// turn and is instruction-heavy, so this is slow and prone to false positives).
    pub include_system_prompt: bool,
    /// Regexes for harness-written text removed before scoring (e.g. Claude Code's
    /// `<system-reminder>` blocks). Compiled at load; an invalid pattern rejects the policy.
    pub skip_patterns: Vec<String>,
    /// Regexes for exact harness text appended to tool outputs. Keep these exact: a generic
    /// pattern lets a poisoned file or page hide an injection from the model.
    pub tool_output_skip_patterns: Vec<String>,
    /// Regexes for user text blocks that hold tool output inlined by the harness (Claude Code
    /// `@file` mentions). Such a block is scored whole as tool output; skip patterns ignored.
    /// An optional `(?P<tool>...)` group names the tool.
    pub inline_tool_output_patterns: Vec<String>,
    /// Characters of the user's latest request sent to the model as context.
    pub context_chars: usize,
    pub on_block: BlockResponse,
    /// Keep blocking a conversation that still contains content blocked earlier.
    pub remember_blocked: bool,
}

impl Default for RouteAPolicy {
    fn default() -> Self {
        Self {
            mode: EnforcementMode::Enforce,
            score: ScoreScope::New,
            include_user_text: true,
            include_system_prompt: false,
            skip_patterns: vec![r"(?s)<system-reminder>.*?</system-reminder>".to_string()],
            tool_output_skip_patterns: Vec::new(),
            inline_tool_output_patterns: vec![
                r"Result of calling the (?P<tool>\w+) tool".to_string()
            ],
            context_chars: 1000,
            on_block: BlockResponse::Error,
            remember_blocked: true,
        }
    }
}

/// `[alerts]` section of `policy.toml`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(default, deny_unknown_fields)]
pub struct AlertPolicy {
    /// Native desktop notification (macOS / Linux) when something is blocked or flagged.
    pub desktop: bool,
}

/// Global operational thresholds and invariants.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DefaultPolicy {
    #[serde(default = "default_max_injection")]
    pub max_injection_confidence: f32,
    #[serde(default = "default_max_intent")]
    pub max_malicious_intent: f32,
    #[serde(default = "default_model_timeout")]
    pub model_timeout_ms: u64,
    #[serde(default = "default_true")]
    pub fail_closed: bool,
    #[serde(default = "default_true")]
    pub require_jwt: bool,
}

impl Default for DefaultPolicy {
    fn default() -> Self {
        Self {
            max_injection_confidence: default_max_injection(),
            max_malicious_intent: default_max_intent(),
            model_timeout_ms: default_model_timeout(),
            fail_closed: true,
            require_jwt: true,
        }
    }
}

/// Specific access control and constraint rules per endpoint.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct EndpointRule {
    pub backend_url: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub allowed_methods: Vec<String>,
    #[serde(default)]
    pub allowed_roles: Vec<String>,
    #[serde(default)]
    pub required_args: Vec<String>,
    #[serde(default)]
    pub constraints: HashMap<String, ArgConstraint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_injection_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_malicious_intent: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_timeout_ms: Option<u64>,
}

impl EndpointRule {
    /// Resolves effective operational thresholds, falling back to global defaults.
    pub fn effective_thresholds(&self, defaults: &DefaultPolicy) -> (f32, f32) {
        (
            self.max_injection_confidence
                .unwrap_or(defaults.max_injection_confidence),
            self.max_malicious_intent
                .unwrap_or(defaults.max_malicious_intent),
        )
    }

    /// Resolves effective model timeout in milliseconds, falling back to global default.
    pub fn effective_timeout_ms(&self, defaults: &DefaultPolicy) -> u64 {
        self.model_timeout_ms.unwrap_or(defaults.model_timeout_ms)
    }
}

/// Specific access control and constraint rules for local CLI tool execution (R1, R4).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ToolRule {
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default)]
    pub allowed_roles: Vec<String>,
    #[serde(default)]
    pub required_args: Vec<String>,
    #[serde(default)]
    pub constraints: HashMap<String, ArgConstraint>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_injection_confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_malicious_intent: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_timeout_ms: Option<u64>,
}

impl Default for ToolRule {
    fn default() -> Self {
        Self {
            enabled: true,
            allowed_roles: Vec::new(),
            required_args: Vec::new(),
            constraints: HashMap::new(),
            max_injection_confidence: None,
            max_malicious_intent: None,
            model_timeout_ms: None,
        }
    }
}

impl ToolRule {
    /// Resolves effective operational thresholds, falling back to global defaults.
    pub fn effective_thresholds(&self, defaults: &DefaultPolicy) -> (f32, f32) {
        (
            self.max_injection_confidence
                .unwrap_or(defaults.max_injection_confidence),
            self.max_malicious_intent
                .unwrap_or(defaults.max_malicious_intent),
        )
    }

    /// Resolves effective model timeout in milliseconds, falling back to global default.
    pub fn effective_timeout_ms(&self, defaults: &DefaultPolicy) -> u64 {
        self.model_timeout_ms.unwrap_or(defaults.model_timeout_ms)
    }
}

/// Configuration-driven mapping for target agent CLIs (R2).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct AgentCliMapping {
    /// Primary environment variable name to route CLI traffic through sidecar (e.g. "ANTHROPIC_BASE_URL", "OPENAI_BASE_URL", "OLLAMA_HOST").
    pub env_var: String,
    /// Upstream API key environment variable name to redact from logs (e.g. "ANTHROPIC_API_KEY", "OPENAI_API_KEY").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env_var: Option<String>,
    /// Sidecar URL that the agent CLI should be directed to (e.g. "http://127.0.0.1:8080").
    pub sidecar_url: String,
    /// Upstream LLM URL to which the sidecar proxies requests (e.g. "https://api.anthropic.com/v1/messages").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream_url: Option<String>,
    /// Wire protocol ("anthropic", "openai", "ollama", "gemini"). Defaults to "anthropic".
    #[serde(default = "default_protocol")]
    pub protocol: String,
    /// Additional environment variables required by the target agent CLI.
    #[serde(default)]
    pub extra_env: HashMap<String, String>,
}

impl AgentCliMapping {
    /// Constructs the environment variable pairs needed to direct the agent CLI to the sidecar.
    pub fn to_env_vars(&self) -> HashMap<String, String> {
        let mut envs = self.extra_env.clone();
        envs.insert(self.env_var.clone(), self.sidecar_url.clone());
        envs
    }
}

/// Constraint validation rules for an individual payload argument.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArgConstraint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub regex: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_len: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kev_verdict_safe_factory_and_is_safe() {
        let verdict = KevVerdict::safe("kev-0.8b");
        assert_eq!(verdict.model_id, "kev-0.8b");
        assert!(verdict.is_safe(0.70, 0.50));
        assert_eq!(verdict.reason, "clean evaluation");

        let scores: KevScores = (&verdict).into();
        assert_eq!(scores.injection_confidence, 0.01);
        assert_eq!(scores.malicious_intent, 0.01);

        let dangerous = KevVerdict::new(0.85, 0.10, "kev-0.8b", "injection detected");
        assert!(!dangerous.is_safe(0.70, 0.50));

        let malicious = KevVerdict::new(0.20, 0.60, "kev-0.8b", "malicious payload");
        assert!(!malicious.is_safe(0.70, 0.50));
    }

    #[test]
    fn test_deny_rule_as_str_and_http_status() {
        assert_eq!(DenyRule::Malformed.as_str(), "malformed");
        assert_eq!(DenyRule::Malformed.http_status(), StatusCode::BAD_REQUEST);

        assert_eq!(DenyRule::UnauthorizedToken.as_str(), "unauthorized_token");
        assert_eq!(
            DenyRule::UnauthorizedToken.http_status(),
            StatusCode::FORBIDDEN
        );

        assert_eq!(DenyRule::EndpointDenied.as_str(), "endpoint_denied");
        assert_eq!(
            DenyRule::EndpointDenied.http_status(),
            StatusCode::FORBIDDEN
        );

        assert_eq!(DenyRule::UnauthorizedRole.as_str(), "unauthorized_role");
        assert_eq!(
            DenyRule::UnauthorizedRole.http_status(),
            StatusCode::FORBIDDEN
        );

        assert_eq!(DenyRule::MissingArg.as_str(), "missing_arg");
        assert_eq!(DenyRule::MissingArg.http_status(), StatusCode::FORBIDDEN);

        assert_eq!(DenyRule::ConstraintFailed.as_str(), "constraint_failed");
        assert_eq!(
            DenyRule::ConstraintFailed.http_status(),
            StatusCode::FORBIDDEN
        );

        assert_eq!(DenyRule::ModelUnavailable.as_str(), "model_unavailable");
        assert_eq!(
            DenyRule::ModelUnavailable.http_status(),
            StatusCode::FORBIDDEN
        );

        assert_eq!(DenyRule::ModelRiskHigh.as_str(), "model_risk_high");
        assert_eq!(DenyRule::ModelRiskHigh.http_status(), StatusCode::FORBIDDEN);

        assert_eq!(DenyRule::BackendError.as_str(), "backend_error");
        assert_eq!(
            DenyRule::BackendError.http_status(),
            StatusCode::BAD_GATEWAY
        );
    }

    #[test]
    fn test_deny_rule_serde_roundtrip() {
        for rule in [
            DenyRule::Malformed,
            DenyRule::UnauthorizedToken,
            DenyRule::EndpointDenied,
            DenyRule::UnauthorizedRole,
            DenyRule::MissingArg,
            DenyRule::ConstraintFailed,
            DenyRule::ModelUnavailable,
            DenyRule::ModelRiskHigh,
            DenyRule::BackendError,
        ] {
            let json = serde_json::to_string(&rule).unwrap();
            let parsed: DenyRule = serde_json::from_str(&json).unwrap();
            assert_eq!(rule, parsed);
            assert_eq!(format!("\"{}\"", rule.as_str()), json);
        }
    }

    #[test]
    fn test_tool_execute_request_deserialization_defaults() {
        let raw = r#"{
            "session_id": "sess-123",
            "endpoint": "/api/v1/system/status"
        }"#;

        let req: ToolExecuteRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id, "sess-123");
        assert_eq!(req.endpoint, "/api/v1/system/status");
        assert_eq!(req.method, "POST"); // default applied
        assert!(req.payload.is_empty());
        assert_eq!(req.context.user_prompt, None);
        assert!(req.context.untrusted_inputs.is_empty());
    }

    #[test]
    fn test_tool_execute_request_with_full_payload_and_context() {
        let raw = r#"{
            "session_id": "sess-456",
            "endpoint": "/api/v1/email/send",
            "method": "POST",
            "payload": {
                "to": "alice@ourcompany.com",
                "body": "Hello world"
            },
            "context": {
                "user_prompt": "Send an email to Alice",
                "untrusted_inputs": ["raw clipboard text"]
            }
        }"#;

        let req: ToolExecuteRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.session_id, "sess-456");
        assert_eq!(req.endpoint, "/api/v1/email/send");
        assert_eq!(req.method, "POST");
        assert_eq!(
            req.payload.get("to").unwrap(),
            &serde_json::Value::String("alice@ourcompany.com".to_string())
        );
        assert_eq!(
            req.context.user_prompt.as_deref(),
            Some("Send an email to Alice")
        );
        assert_eq!(
            req.context.untrusted_inputs,
            vec!["raw clipboard text".to_string()]
        );
    }

    #[test]
    fn test_route_b_allow_and_deny_response_serde() {
        let allow = RouteBAllowResponse {
            decision: "allow".to_string(),
            endpoint: "/api/v1/system/status".to_string(),
            kev_scores: KevScores {
                injection_confidence: 0.05,
                malicious_intent: 0.02,
            },
            backend_response: serde_json::json!({"status": "ok"}),
        };
        let allow_json = serde_json::to_string(&allow).unwrap();
        let allow_deser: RouteBAllowResponse = serde_json::from_str(&allow_json).unwrap();
        assert_eq!(allow, allow_deser);

        let deny = RouteBDenyResponse {
            decision: "deny".to_string(),
            rule: DenyRule::ModelRiskHigh,
            reason: "Injection risk 0.85 exceeds threshold 0.70".to_string(),
            kev_scores: Some(KevScores {
                injection_confidence: 0.85,
                malicious_intent: 0.12,
            }),
        };
        let deny_json = serde_json::to_string(&deny).unwrap();
        let deny_deser: RouteBDenyResponse = serde_json::from_str(&deny_json).unwrap();
        assert_eq!(deny, deny_deser);
    }

    #[test]
    fn test_anthropic_messages_request_text_extraction() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            system: Some("You are a helpful security assistant.".to_string()),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text("Check status of service core".to_string()),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![
                        ContentBlock {
                            block_type: "text".to_string(),
                            text: Some("Executing status check...".to_string()),
                            content: None,
                        },
                        ContentBlock {
                            block_type: "tool_result".to_string(),
                            text: None,
                            content: Some(serde_json::Value::String("service healthy".to_string())),
                        },
                    ]),
                },
            ],
            max_tokens: Some(1024),
            stream: Some(false),
            temperature: Some(0.2),
        };

        let extracted = req.extract_all_text();
        assert!(extracted.contains("You are a helpful security assistant."));
        assert!(extracted.contains("Check status of service core"));
        assert!(extracted.contains("Executing status check..."));
        assert!(extracted.contains("service healthy"));
    }

    #[test]
    fn test_route_a_error_response_serde() {
        let err_resp = RouteAErrorResponse {
            response_type: "error".to_string(),
            error: RouteAErrorDetail {
                error_type: "zero_trust_policy_violation".to_string(),
                rule: "model_risk_high".to_string(),
                message: "Prompt rejected by in-process zero-trust classifier".to_string(),
                injection_confidence: 0.92,
                malicious_intent: 0.44,
            },
        };

        let json = serde_json::to_string(&err_resp).unwrap();
        assert!(json.contains(r#""type":"error""#));
        assert!(json.contains(r#""type":"zero_trust_policy_violation""#));
        assert!(json.contains(r#""rule":"model_risk_high""#));

        let parsed: RouteAErrorResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(err_resp, parsed);
    }

    #[test]
    fn test_audit_entry_serde_omits_none() {
        let entry = AuditEntry {
            timestamp: chrono::Utc::now(),
            route: "POST /api/execute".to_string(),
            session_id: Some("sess-999".to_string()),
            caller_sub: Some("agent-007".to_string()),
            caller_role: Some("agent".to_string()),
            target_endpoint: Some("/api/v1/system/status".to_string()),
            method: None,
            decision: "allow".to_string(),
            matched_rule: None,
            reason: None,
            injection_confidence: Some(0.02),
            malicious_intent: Some(0.01),
            latency_ms: 12,
        };

        let json = serde_json::to_string(&entry).unwrap();
        assert!(!json.contains("matched_rule"));
        assert!(!json.contains("reason"));
        assert!(!json.contains("method"));
        assert!(json.contains("caller_sub"));
        assert!(json.contains("agent-007"));
    }

    #[test]
    fn test_agent_claims_serde() {
        let claims = AgentClaims {
            sub: "agent-001".to_string(),
            role: "admin".to_string(),
            exp: 1731000000,
            payload_sha256: Some(
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string(),
            ),
        };
        let json = serde_json::to_string(&claims).unwrap();
        let parsed: AgentClaims = serde_json::from_str(&json).unwrap();
        assert_eq!(claims, parsed);
    }

    #[test]
    fn test_policy_config_toml_deserialization() {
        let toml_data = r#"
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

        [endpoints."/api/v1/db/drop"]
        backend_url = "http://127.0.0.1:5000/api/v1/db/drop"
        enabled = false
        allowed_methods = ["POST"]
        allowed_roles = []
        required_args = ["table"]
        "#;

        let policy: PolicyConfig = toml::from_str(toml_data).unwrap();
        assert_eq!(policy.defaults.max_injection_confidence, 0.70);
        assert_eq!(policy.defaults.max_malicious_intent, 0.50);
        assert_eq!(policy.defaults.model_timeout_ms, 500);
        assert!(policy.defaults.fail_closed);
        assert!(policy.defaults.require_jwt);

        let status_rule = policy.endpoints.get("/api/v1/system/status").unwrap();
        assert!(status_rule.enabled);
        assert_eq!(status_rule.allowed_roles, vec!["agent", "admin"]);
        assert_eq!(status_rule.required_args, vec!["service"]);
        let svc_constraint = status_rule.constraints.get("service").unwrap();
        assert_eq!(svc_constraint.regex.as_deref(), Some("^[a-z0-9-]+$"));
        assert_eq!(svc_constraint.max_len, Some(64));

        let drop_rule = policy.endpoints.get("/api/v1/db/drop").unwrap();
        assert!(!drop_rule.enabled);
        assert!(drop_rule.allowed_roles.is_empty());
    }

    #[test]
    fn test_policy_config_full_deserialization_with_tools_and_mappings() {
        let toml_data = r#"
        [defaults]
        max_injection_confidence = 0.65
        max_malicious_intent = 0.45
        model_timeout_ms = 400
        fail_closed = true
        require_jwt = true

        [route_a]
        score = "all"
        include_system_prompt = true

        [endpoints."/api/v1/system/status"]
        backend_url = "http://127.0.0.1:5000/api/v1/system/status"
        enabled = true
        allowed_methods = ["GET", "POST"]
        allowed_roles = ["agent"]
        required_args = ["service"]
        constraints.service = { regex = "^[a-z0-9-]+$", max_len = 64 }
        max_malicious_intent = 0.30
        model_timeout_ms = 600

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

        let policy: PolicyConfig = toml::from_str(toml_data).unwrap();
        assert_eq!(policy.defaults.max_injection_confidence, 0.65);
        assert_eq!(policy.defaults.max_malicious_intent, 0.45);
        assert_eq!(policy.defaults.model_timeout_ms, 400);
        assert_eq!(policy.route_a.score, ScoreScope::All);
        assert!(policy.route_a.include_system_prompt);

        // Verify Endpoint threshold override
        let ep = policy.endpoints.get("/api/v1/system/status").unwrap();
        assert_eq!(ep.max_malicious_intent, Some(0.30));
        assert_eq!(ep.max_injection_confidence, None);
        let (inj, intent) = ep.effective_thresholds(&policy.defaults);
        assert_eq!(inj, 0.65); // inherited from default
        assert_eq!(intent, 0.30); // overridden
        assert_eq!(ep.effective_timeout_ms(&policy.defaults), 600); // overridden

        // Verify Tool rule
        let tool = policy.tools.get("bash").unwrap();
        assert!(tool.enabled);
        assert_eq!(tool.allowed_roles, vec!["admin"]);
        assert_eq!(tool.required_args, vec!["command"]);
        let (t_inj, t_intent) = tool.effective_thresholds(&policy.defaults);
        assert_eq!(t_inj, 0.65);
        assert_eq!(t_intent, 0.40);
        assert_eq!(tool.effective_timeout_ms(&policy.defaults), 400);

        // Verify Agent CLI Mapping
        let cli = policy.agent_cli_mappings.get("claude").unwrap();
        assert_eq!(cli.env_var, "ANTHROPIC_BASE_URL");
        assert_eq!(cli.api_key_env_var.as_deref(), Some("ANTHROPIC_API_KEY"));
        assert_eq!(cli.protocol, "anthropic");
        let env_map = cli.to_env_vars();
        assert_eq!(
            env_map.get("ANTHROPIC_BASE_URL"),
            Some(&"http://127.0.0.1:8080".to_string())
        );
        assert_eq!(
            env_map.get("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_deny_unknown_fields_on_all_policy_structs() {
        // Unknown field at root
        let bad_root = r#"
        [defaults]
        [unknown_section]
        foo = "bar"
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_root).is_err());

        // Unknown field in defaults
        let bad_defaults = r#"
        [defaults]
        unknown_key = 123
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_defaults).is_err());

        // Misspelled [route_a] / [alerts] keys (on_blok, desktp)
        let bad_route_a = r#"
        [defaults]
        [route_a]
        on_blok = "message"
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_route_a).is_err());
        let bad_alerts = r#"
        [defaults]
        [alerts]
        desktp = true
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_alerts).is_err());

        // Unknown field in endpoint
        let bad_endpoint = r#"
        [defaults]
        [endpoints."/api/v1/status"]
        backend_url = "http://localhost:5000"
        invalid_endpoint_key = true
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_endpoint).is_err());

        // Unknown field in tools
        let bad_tool = r#"
        [defaults]
        [tools."bash"]
        backend_url = "http://localhost:5000"
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_tool).is_err());

        // Unknown field in agent_cli_mappings
        let bad_cli = r#"
        [defaults]
        [agent_cli_mappings.claude]
        env_var = "ANTHROPIC_BASE_URL"
        sidecar_url = "http://localhost:8080"
        unknown_flag = false
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_cli).is_err());

        // Unknown field in constraint (e.g. misspelling max_len as maxlen)
        let bad_constraint = r#"
        [defaults]
        [endpoints."/api/v1/status"]
        backend_url = "http://localhost:5000"
        constraints.service = { maxlen = 64 }
        "#;
        assert!(toml::from_str::<PolicyConfig>(bad_constraint).is_err());
    }

    #[test]
    fn test_tool_rule_defaults_and_effective_thresholds() {
        let defaults = DefaultPolicy::default();
        assert_eq!(defaults.max_injection_confidence, 0.70);
        assert_eq!(defaults.max_malicious_intent, 0.50);
        assert_eq!(defaults.model_timeout_ms, 500);
        let route_a = RouteAPolicy::default();
        assert_eq!(route_a.score, ScoreScope::New);
        assert!(!route_a.include_system_prompt);

        let mut tool = ToolRule::default();
        assert!(tool.enabled);
        assert!(tool.allowed_roles.is_empty());
        assert!(tool.required_args.is_empty());
        assert!(tool.constraints.is_empty());
        assert_eq!(tool.effective_thresholds(&defaults), (0.70, 0.50));
        assert_eq!(tool.effective_timeout_ms(&defaults), 500);

        tool.max_malicious_intent = Some(0.25);
        tool.model_timeout_ms = Some(1200);
        assert_eq!(tool.effective_thresholds(&defaults), (0.70, 0.25));
        assert_eq!(tool.effective_timeout_ms(&defaults), 1200);
    }
}
