//! Configuration Loader for Zero-Trust Agentic Sidecar (`src/config.rs`)
//!
//! Enforces fail-closed validation of required cryptographic secrets
//! (`SIDECAR_SECRET`, `JWT_SECRET`) and provides production-safe defaults
//! for all operational endpoints and thresholds.

use std::collections::HashMap;
use std::net::SocketAddr;
use std::path::PathBuf;
use thiserror::Error;

/// Errors arising during configuration parsing and validation.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("Missing required environment variable: {0}")]
    MissingVar(&'static str),

    #[error("Required environment variable cannot be empty: {0}")]
    EmptyVar(&'static str),

    #[error("Invalid configuration value for {var}: {reason}")]
    InvalidValue { var: &'static str, reason: String },
}

/// Runtime configuration settings for the Zero-Trust Agentic Sidecar.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Listener socket address (e.g. 127.0.0.1:8080)
    pub sidecar_addr: SocketAddr,
    /// Upstream Anthropic API key (optional in tests, injected into Route A requests)
    pub anthropic_api_key: Option<String>,
    /// Upstream Anthropic LLM endpoint
    pub upstream_llm_url: String,
    /// Target Python Flask microservice backend URL
    pub flask_backend_url: String,
    /// Cryptographic signing secret for Sidecar -> Flask backend HMAC-SHA256
    pub sidecar_secret: String,
    /// Cryptographic secret for verifying inbound agent Bearer JWTs
    pub jwt_secret: String,
    /// Path to the zero-trust TOML policy file
    pub policy_path: PathBuf,
    /// Hugging Face repo ID for Kev adapter
    pub kev_repo_id: String,
    /// Hugging Face repo ID for Qwen base model
    pub kev_base_repo_id: String,
    /// Local directory for downloaded model artifacts
    pub model_cache_dir: PathBuf,
}

impl Config {
    /// Loads configuration from process environment variables with fail-closed secret validation.
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_vars(std::env::vars())
    }

    /// Loads configuration from an arbitrary key-value iterator.
    /// Enables thread-safe and deterministic testing without mutating process environment variables.
    pub fn from_vars<I, K, V>(vars: I) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let env_map: HashMap<String, String> = vars
            .into_iter()
            .map(|(k, v)| (k.as_ref().to_string(), v.as_ref().to_string()))
            .collect();

        let sidecar_secret = Self::get_required_secret(&env_map, "SIDECAR_SECRET")?;
        let jwt_secret = Self::get_required_secret(&env_map, "JWT_SECRET")?;

        let sidecar_addr_str = env_map
            .get("SIDECAR_ADDR")
            .cloned()
            .unwrap_or_else(|| "127.0.0.1:8080".to_string());

        let sidecar_addr: SocketAddr =
            sidecar_addr_str
                .parse()
                .map_err(|e| ConfigError::InvalidValue {
                    var: "SIDECAR_ADDR",
                    reason: format!("Failed to parse socket address '{sidecar_addr_str}': {e}"),
                })?;

        let anthropic_api_key = env_map
            .get("ANTHROPIC_API_KEY")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());

        let upstream_llm_url = env_map
            .get("UPSTREAM_LLM_URL")
            .cloned()
            .unwrap_or_else(|| "https://api.anthropic.com/v1/messages".to_string());

        let flask_backend_url = env_map
            .get("FLASK_BACKEND_URL")
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:5000".to_string());

        let policy_path = env_map
            .get("POLICY_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("policy.toml"));

        let kev_repo_id = env_map
            .get("KEV_REPO_ID")
            .cloned()
            .unwrap_or_else(|| "jaredpalmer/kev-0.8b".to_string());

        let kev_base_repo_id = env_map
            .get("KEV_BASE_REPO_ID")
            .cloned()
            .unwrap_or_else(|| "Qwen/Qwen3.5-0.8B-Base".to_string());

        let model_cache_dir = env_map
            .get("MODEL_CACHE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("models"));

        Ok(Self {
            sidecar_addr,
            anthropic_api_key,
            upstream_llm_url,
            flask_backend_url,
            sidecar_secret,
            jwt_secret,
            policy_path,
            kev_repo_id,
            kev_base_repo_id,
            model_cache_dir,
        })
    }

    /// Convenience helper for unit and integration testing without process env mutation.
    pub fn for_test(sidecar_secret: &str, jwt_secret: &str) -> Self {
        Self {
            sidecar_addr: "127.0.0.1:8080".parse().expect("valid test socket addr"),
            anthropic_api_key: Some("sk-test-key".to_string()),
            upstream_llm_url: "https://api.anthropic.com/v1/messages".to_string(),
            flask_backend_url: "http://127.0.0.1:5000".to_string(),
            sidecar_secret: sidecar_secret.to_string(),
            jwt_secret: jwt_secret.to_string(),
            policy_path: PathBuf::from("policy.toml"),
            kev_repo_id: "jaredpalmer/kev-0.8b".to_string(),
            kev_base_repo_id: "Qwen/Qwen3.5-0.8B-Base".to_string(),
            model_cache_dir: PathBuf::from("models"),
        }
    }

    fn get_required_secret(
        env_map: &HashMap<String, String>,
        var_name: &'static str,
    ) -> Result<String, ConfigError> {
        match env_map.get(var_name) {
            Some(val) if val.trim().is_empty() => Err(ConfigError::EmptyVar(var_name)),
            Some(val) => Ok(val.clone()),
            None => Err(ConfigError::MissingVar(var_name)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_for_test_deterministic_values() {
        let config = Config::for_test("secret-sidecar-1", "secret-jwt-2");
        assert_eq!(config.sidecar_secret, "secret-sidecar-1");
        assert_eq!(config.jwt_secret, "secret-jwt-2");
        assert_eq!(config.sidecar_addr, "127.0.0.1:8080".parse().unwrap());
        assert_eq!(config.anthropic_api_key, Some("sk-test-key".to_string()));
        assert_eq!(
            config.upstream_llm_url,
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(config.flask_backend_url, "http://127.0.0.1:5000");
        assert_eq!(config.policy_path, PathBuf::from("policy.toml"));
        assert_eq!(config.kev_repo_id, "jaredpalmer/kev-0.8b");
        assert_eq!(config.kev_base_repo_id, "Qwen/Qwen3.5-0.8B-Base");
        assert_eq!(config.model_cache_dir, PathBuf::from("models"));
    }

    #[test]
    fn test_config_from_vars_valid_all_overrides() {
        let vars = [
            ("SIDECAR_SECRET", "custom-sidecar-secret"),
            ("JWT_SECRET", "custom-jwt-secret"),
            ("SIDECAR_ADDR", "0.0.0.0:9090"),
            ("ANTHROPIC_API_KEY", "sk-custom-key"),
            ("UPSTREAM_LLM_URL", "https://custom.llm.example/v1/messages"),
            ("FLASK_BACKEND_URL", "http://backend.internal:5000"),
            ("POLICY_PATH", "custom_policy.toml"),
            ("KEV_REPO_ID", "custom/kev-model"),
            ("KEV_BASE_REPO_ID", "custom/base-model"),
            ("MODEL_CACHE_DIR", "/custom/cache"),
        ];

        let config = Config::from_vars(vars).expect("Config should parse successfully");
        assert_eq!(config.sidecar_secret, "custom-sidecar-secret");
        assert_eq!(config.jwt_secret, "custom-jwt-secret");
        assert_eq!(config.sidecar_addr, "0.0.0.0:9090".parse().unwrap());
        assert_eq!(config.anthropic_api_key, Some("sk-custom-key".to_string()));
        assert_eq!(
            config.upstream_llm_url,
            "https://custom.llm.example/v1/messages"
        );
        assert_eq!(config.flask_backend_url, "http://backend.internal:5000");
        assert_eq!(config.policy_path, PathBuf::from("custom_policy.toml"));
        assert_eq!(config.kev_repo_id, "custom/kev-model");
        assert_eq!(config.kev_base_repo_id, "custom/base-model");
        assert_eq!(config.model_cache_dir, PathBuf::from("/custom/cache"));
    }

    #[test]
    fn test_config_from_vars_default_fallbacks() {
        let vars = [
            ("SIDECAR_SECRET", "test-sidecar-secret"),
            ("JWT_SECRET", "test-jwt-secret"),
        ];

        let config = Config::from_vars(vars).expect("Config should parse successfully");
        assert_eq!(config.sidecar_addr, "127.0.0.1:8080".parse().unwrap());
        assert_eq!(config.anthropic_api_key, None);
        assert_eq!(
            config.upstream_llm_url,
            "https://api.anthropic.com/v1/messages"
        );
        assert_eq!(config.flask_backend_url, "http://127.0.0.1:5000");
        assert_eq!(config.policy_path, PathBuf::from("policy.toml"));
        assert_eq!(config.kev_repo_id, "jaredpalmer/kev-0.8b");
        assert_eq!(config.kev_base_repo_id, "Qwen/Qwen3.5-0.8B-Base");
        assert_eq!(config.model_cache_dir, PathBuf::from("models"));
    }

    #[test]
    fn test_config_missing_sidecar_secret_fails_closed() {
        let vars = [("JWT_SECRET", "jwt-secret-only")];
        let err = Config::from_vars(vars).unwrap_err();
        assert_eq!(err, ConfigError::MissingVar("SIDECAR_SECRET"));
    }

    #[test]
    fn test_config_empty_sidecar_secret_fails_closed() {
        let vars = [
            ("SIDECAR_SECRET", "   \t\n"),
            ("JWT_SECRET", "jwt-secret-val"),
        ];
        let err = Config::from_vars(vars).unwrap_err();
        assert_eq!(err, ConfigError::EmptyVar("SIDECAR_SECRET"));
    }

    #[test]
    fn test_config_missing_jwt_secret_fails_closed() {
        let vars = [("SIDECAR_SECRET", "sidecar-secret-only")];
        let err = Config::from_vars(vars).unwrap_err();
        assert_eq!(err, ConfigError::MissingVar("JWT_SECRET"));
    }

    #[test]
    fn test_config_empty_jwt_secret_fails_closed() {
        let vars = [("SIDECAR_SECRET", "sidecar-secret-val"), ("JWT_SECRET", "")];
        let err = Config::from_vars(vars).unwrap_err();
        assert_eq!(err, ConfigError::EmptyVar("JWT_SECRET"));
    }

    #[test]
    fn test_config_invalid_sidecar_addr_fails() {
        let vars = [
            ("SIDECAR_SECRET", "secret-1"),
            ("JWT_SECRET", "secret-2"),
            ("SIDECAR_ADDR", "invalid-socket-address"),
        ];
        let err = Config::from_vars(vars).unwrap_err();
        match err {
            ConfigError::InvalidValue { var, reason } => {
                assert_eq!(var, "SIDECAR_ADDR");
                assert!(reason.contains("Failed to parse socket address"));
            }
            other => panic!("Expected InvalidValue, got {:?}", other),
        }
    }

    #[test]
    fn test_config_whitespace_anthropic_api_key_sanitized_to_none() {
        let vars = [
            ("SIDECAR_SECRET", "secret-1"),
            ("JWT_SECRET", "secret-2"),
            ("ANTHROPIC_API_KEY", "   "),
        ];
        let config = Config::from_vars(vars).unwrap();
        assert_eq!(config.anthropic_api_key, None);
    }
}
