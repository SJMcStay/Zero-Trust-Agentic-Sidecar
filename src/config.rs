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

/// Which decision engine scores prompts and tool calls (`KEV_ENGINE`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KevEngineKind {
    /// Kev's reference server on loopback (`scripts/kev_server.sh`). Default.
    Http,
    /// In-process Candle port. Experimental: not yet numerically equivalent to Kev.
    Candle,
    /// Deterministic toy graph. Tests and demos only.
    Synthetic,
}

impl std::str::FromStr for KevEngineKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "http" => Ok(Self::Http),
            "candle" => Ok(Self::Candle),
            "synthetic" => Ok(Self::Synthetic),
            other => Err(format!("'{other}' is not one of: http, candle, synthetic")),
        }
    }
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
    /// Decision engine (`KEV_ENGINE`: http | candle | synthetic)
    pub kev_engine: KevEngineKind,
    /// Kev reference server base URL for the http engine (`KEV_URL`)
    pub kev_url: String,
    /// Kev checkpoint name in `models.json` (`KEV_MODEL`), e.g. `kev-4b`
    pub kev_model: String,
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

        let kev_engine = match env_map.get("KEV_ENGINE") {
            Some(value) => {
                value
                    .parse::<KevEngineKind>()
                    .map_err(|reason| ConfigError::InvalidValue {
                        var: "KEV_ENGINE",
                        reason,
                    })?
            }
            None => KevEngineKind::Http,
        };

        let kev_url = env_map
            .get("KEV_URL")
            .cloned()
            .unwrap_or_else(|| "http://127.0.0.1:8008".to_string());

        let kev_model = env_map
            .get("KEV_MODEL")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "kev-4b".to_string());

        Ok(Self {
            sidecar_addr,
            anthropic_api_key,
            upstream_llm_url,
            flask_backend_url,
            sidecar_secret,
            jwt_secret,
            policy_path,
            kev_engine,
            kev_url,
            kev_model,
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
            kev_engine: KevEngineKind::Synthetic,
            kev_url: "http://127.0.0.1:8008".to_string(),
            kev_model: "kev-4b".to_string(),
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
        assert_eq!(config.kev_engine, KevEngineKind::Synthetic);
        assert_eq!(config.kev_url, "http://127.0.0.1:8008");
        assert_eq!(config.kev_model, "kev-4b");
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
            ("KEV_ENGINE", "Candle"),
            ("KEV_URL", "http://127.0.0.1:9008"),
            ("KEV_MODEL", "kev-0.8b"),
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
        assert_eq!(config.kev_engine, KevEngineKind::Candle);
        assert_eq!(config.kev_url, "http://127.0.0.1:9008");
        assert_eq!(config.kev_model, "kev-0.8b");
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
        assert_eq!(config.kev_engine, KevEngineKind::Http);
        assert_eq!(config.kev_url, "http://127.0.0.1:8008");
        assert_eq!(config.kev_model, "kev-4b");
    }

    #[test]
    fn test_config_invalid_kev_engine_fails() {
        let vars = [
            ("SIDECAR_SECRET", "secret-1"),
            ("JWT_SECRET", "secret-2"),
            ("KEV_ENGINE", "onnx"),
        ];
        match Config::from_vars(vars).unwrap_err() {
            ConfigError::InvalidValue { var, .. } => assert_eq!(var, "KEV_ENGINE"),
            other => panic!("Expected InvalidValue, got {:?}", other),
        }
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
