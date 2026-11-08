//! Cryptographic Authorization Module (`src/crypto.rs`)
//!
//! Enforces two-legged zero-trust cryptographic authorization:
//! 1. Inbound HS256 JWT validation for agent requests on Route B (`POST /api/execute`).
//! 2. Outbound HMAC-SHA256 signing and replay-window verification for Sidecar-to-Backend calls.

use chrono::Utc;
use hmac::{Hmac, Mac};
use jsonwebtoken::{decode, encode, Algorithm, DecodingKey, EncodingKey, Header, Validation};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use thiserror::Error;

pub use crate::types::AgentClaims;

/// HTTP header name carrying the Unix timestamp (seconds) of an outbound signed request.
pub const HEADER_TIMESTAMP: &str = "X-Sidecar-Timestamp";

/// HTTP header name carrying the hex-encoded HMAC-SHA256 signature.
pub const HEADER_SIGNATURE: &str = "X-Sidecar-Signature";

/// Maximum allowed clock skew / replay window duration in seconds.
pub const DEFAULT_MAX_SKEW_SECS: u64 = 30;

type HmacSha256 = Hmac<Sha256>;

/// Errors arising during cryptographic signing, verification, or token parsing.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CryptoError {
    #[error("Invalid JWT token: {0}")]
    InvalidToken(String),

    #[error("JWT token has expired")]
    TokenExpired,

    #[error("Invalid JWT claims: {0}")]
    InvalidClaims(String),

    #[error("Missing raw payload required for verification against payload_sha256 claim")]
    MissingPayload,

    #[error("Payload hash mismatch: expected {expected}, computed {actual}")]
    PayloadHashMismatch { expected: String, actual: String },

    #[error("Invalid cryptographic key: {0}")]
    InvalidKey(String),

    #[error("Invalid timestamp: {0}")]
    InvalidTimestamp(String),

    #[error(
        "Signature timestamp skewed: skew of {skew_secs}s exceeds max allowed {max_skew_secs}s"
    )]
    TimestampExpired { skew_secs: u64, max_skew_secs: u64 },

    #[error("Invalid signature format: {0}")]
    InvalidSignatureFormat(String),

    #[error("Cryptographic signature mismatch")]
    SignatureMismatch,
}

impl AgentClaims {
    /// Verifies that the given raw payload matches the bound `payload_sha256` claim in constant time.
    /// If no payload hash is bound, this check succeeds permissively.
    pub fn verify_payload(&self, raw_payload: &[u8]) -> Result<(), CryptoError> {
        if let Some(expected_hash) = &self.payload_sha256 {
            let actual_hash = compute_payload_sha256(raw_payload);
            let expected_lower = expected_hash.to_ascii_lowercase();
            if actual_hash.len() != expected_lower.len()
                || !bool::from(actual_hash.as_bytes().ct_eq(expected_lower.as_bytes()))
            {
                return Err(CryptoError::PayloadHashMismatch {
                    expected: expected_hash.clone(),
                    actual: actual_hash,
                });
            }
        }
        Ok(())
    }
}

/// Computes the lowercase hex-encoded SHA-256 digest of a byte slice.
pub fn compute_payload_sha256(payload: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(payload);
    hex::encode(hasher.finalize())
}

/// Mints an HS256 JWT with an explicit expiration Unix epoch timestamp.
pub fn mint_agent_jwt_with_exp(
    sub: &str,
    role: &str,
    payload_sha256: Option<&str>,
    exp: usize,
    secret: &[u8],
) -> Result<String, CryptoError> {
    if secret.is_empty() {
        return Err(CryptoError::InvalidKey(
            "Secret cannot be empty".to_string(),
        ));
    }
    let claims = AgentClaims {
        sub: sub.to_string(),
        role: role.to_string(),
        exp,
        payload_sha256: payload_sha256.map(|s| s.to_string()),
    };
    let header = Header::new(Algorithm::HS256);
    let encoding_key = EncodingKey::from_secret(secret);
    encode(&header, &claims, &encoding_key).map_err(|e| CryptoError::InvalidToken(e.to_string()))
}

/// Mints an HS256 JWT with a relative TTL (in seconds) from the current UTC time.
/// Negative `ttl_secs` values yield expired tokens for negative security testing.
pub fn mint_agent_jwt(
    sub: &str,
    role: &str,
    payload_sha256: Option<&str>,
    ttl_secs: i64,
    secret: &[u8],
) -> Result<String, CryptoError> {
    let now = Utc::now().timestamp();
    let exp = now.saturating_add(ttl_secs).max(0) as usize;
    mint_agent_jwt_with_exp(sub, role, payload_sha256, exp, secret)
}

/// Convenience helper to mint an HS256 JWT cryptographically bound to a specific payload.
pub fn mint_agent_jwt_for_payload(
    sub: &str,
    role: &str,
    payload: &[u8],
    ttl_secs: i64,
    secret: &[u8],
) -> Result<String, CryptoError> {
    let sha = compute_payload_sha256(payload);
    mint_agent_jwt(sub, role, Some(&sha), ttl_secs, secret)
}

/// Verifies an inbound agent HS256 JWT, enforcing signature, strict zero-leeway expiration,
/// and non-empty identity claims (`sub` and `role`).
///
/// If the token contains a bound `payload_sha256` claim, this method returns
/// `Err(CryptoError::MissingPayload)` because no payload was supplied.
/// To verify bound tokens, use [`verify_agent_jwt_with_payload`].
pub fn verify_agent_jwt(token: &str, secret: &[u8]) -> Result<AgentClaims, CryptoError> {
    verify_agent_jwt_with_payload(token, secret, None)
}

/// Verifies an inbound agent HS256 JWT, enforcing signature, zero-leeway expiration,
/// non-empty identity claims, and constant-time payload SHA-256 verification when bound.
pub fn verify_agent_jwt_with_payload(
    token: &str,
    secret: &[u8],
    raw_payload: Option<&[u8]>,
) -> Result<AgentClaims, CryptoError> {
    if secret.is_empty() {
        return Err(CryptoError::InvalidKey(
            "Secret key cannot be empty".to_string(),
        ));
    }
    let token = token.trim();
    if token.is_empty() {
        return Err(CryptoError::InvalidToken(
            "Token string is empty".to_string(),
        ));
    }

    let mut validation = Validation::new(Algorithm::HS256);
    validation.leeway = 0; // Strict zero-trust expiration: no grace period
    validation.validate_exp = true;

    let decoding_key = DecodingKey::from_secret(secret);
    let token_data =
        decode::<AgentClaims>(token, &decoding_key, &validation).map_err(|e| match e.kind() {
            jsonwebtoken::errors::ErrorKind::ExpiredSignature => CryptoError::TokenExpired,
            jsonwebtoken::errors::ErrorKind::InvalidSignature => {
                CryptoError::InvalidToken("Invalid token signature".to_string())
            }
            _ => CryptoError::InvalidToken(e.to_string()),
        })?;

    let claims = token_data.claims;

    if claims.sub.trim().is_empty() {
        return Err(CryptoError::InvalidClaims(
            "Claim 'sub' cannot be empty".to_string(),
        ));
    }
    if claims.role.trim().is_empty() {
        return Err(CryptoError::InvalidClaims(
            "Claim 'role' cannot be empty".to_string(),
        ));
    }

    if let Some(expected_hash) = &claims.payload_sha256 {
        let payload = raw_payload.ok_or(CryptoError::MissingPayload)?;
        let actual_hash = compute_payload_sha256(payload);

        let expected_lower = expected_hash.to_ascii_lowercase();
        if actual_hash.len() != expected_lower.len()
            || !bool::from(actual_hash.as_bytes().ct_eq(expected_lower.as_bytes()))
        {
            return Err(CryptoError::PayloadHashMismatch {
                expected: expected_hash.clone(),
                actual: actual_hash,
            });
        }
    }

    Ok(claims)
}

/// Computes the HMAC-SHA256 signature for outbound sidecar-to-backend requests.
///
/// Signature is computed over `"{timestamp}.{raw_body}"` using `secret`.
/// Returns a 64-character lowercase hexadecimal string.
pub fn sign_payload(secret: &[u8], timestamp: u64, payload: &[u8]) -> String {
    let mut mac = <HmacSha256 as Mac>::new_from_slice(secret)
        .expect("HMAC-SHA256 accepts arbitrary key lengths");
    mac.update(format!("{timestamp}.").as_bytes());
    mac.update(payload);
    hex::encode(mac.finalize().into_bytes())
}

/// Verifies an outbound payload HMAC-SHA256 signature against current system UTC time.
///
/// Enforces constant-time comparison and rejects timestamps with skew exceeding `max_skew_secs`.
pub fn verify_payload_signature(
    secret: &[u8],
    timestamp: u64,
    payload: &[u8],
    signature: &str,
    max_skew_secs: u64,
) -> Result<(), CryptoError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    verify_payload_signature_at(secret, timestamp, payload, signature, now, max_skew_secs)
}

/// Verifies an outbound payload signature at a specific point in time (`now`).
/// Enables deterministic unit testing without wall-clock dependency or sleep delays.
pub fn verify_payload_signature_at(
    secret: &[u8],
    timestamp: u64,
    payload: &[u8],
    signature: &str,
    now: u64,
    max_skew_secs: u64,
) -> Result<(), CryptoError> {
    if secret.is_empty() {
        return Err(CryptoError::InvalidKey(
            "Secret cannot be empty".to_string(),
        ));
    }

    let skew = now.abs_diff(timestamp);
    if skew > max_skew_secs {
        return Err(CryptoError::TimestampExpired {
            skew_secs: skew,
            max_skew_secs,
        });
    }

    let sig_bytes = hex::decode(signature.trim()).map_err(|e| {
        CryptoError::InvalidSignatureFormat(format!("Failed to decode signature hex: {e}"))
    })?;

    if sig_bytes.len() != 32 {
        return Err(CryptoError::InvalidSignatureFormat(format!(
            "Signature must be 32 bytes (64 hex characters), got {} bytes",
            sig_bytes.len()
        )));
    }

    let mut mac = <HmacSha256 as Mac>::new_from_slice(secret)
        .expect("HMAC-SHA256 accepts arbitrary key lengths");
    mac.update(format!("{timestamp}.").as_bytes());
    mac.update(payload);

    mac.verify_slice(&sig_bytes)
        .map_err(|_| CryptoError::SignatureMismatch)?;

    Ok(())
}

/// Verifies outbound signature where timestamp and signature are passed as raw header string slices.
pub fn verify_payload_signature_headers(
    secret: &[u8],
    ts_header: &str,
    sig_header: &str,
    payload: &[u8],
    max_skew_secs: u64,
) -> Result<(), CryptoError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    verify_payload_signature_headers_at(secret, ts_header, sig_header, payload, now, max_skew_secs)
}

/// Verifies outbound signature from header strings at a deterministic point in time (`now`).
pub fn verify_payload_signature_headers_at(
    secret: &[u8],
    ts_header: &str,
    sig_header: &str,
    payload: &[u8],
    now: u64,
    max_skew_secs: u64,
) -> Result<(), CryptoError> {
    let ts = ts_header.trim().parse::<u64>().map_err(|e| {
        CryptoError::InvalidTimestamp(format!(
            "Failed to parse timestamp header '{ts_header}': {e}"
        ))
    })?;

    verify_payload_signature_at(secret, ts, payload, sig_header, now, max_skew_secs)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_JWT_SECRET: &[u8] = b"test-jwt-secret-key-32-bytes-long!";
    const TEST_HMAC_SECRET: &[u8] = b"test-hmac-secret-key-32-bytes-long";

    #[test]
    fn test_jwt_mint_and_verify_valid() {
        let token = mint_agent_jwt("agent-007", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let claims = verify_agent_jwt(&token, TEST_JWT_SECRET).unwrap();
        assert_eq!(claims.sub, "agent-007");
        assert_eq!(claims.role, "agent");
        assert_eq!(claims.payload_sha256, None);
    }

    #[test]
    fn test_jwt_verify_with_payload_binding_success() {
        let payload = b"{\"service\":\"auth\"}";
        let token = mint_agent_jwt_for_payload("agent-007", "agent", payload, 300, TEST_JWT_SECRET)
            .unwrap();

        let claims = verify_agent_jwt_with_payload(&token, TEST_JWT_SECRET, Some(payload)).unwrap();
        assert_eq!(claims.sub, "agent-007");
        assert_eq!(claims.role, "agent");
        assert_eq!(claims.payload_sha256, Some(compute_payload_sha256(payload)));
    }

    #[test]
    fn test_jwt_verify_payload_tampered() {
        let payload_original = b"{\"service\":\"auth\"}";
        let payload_tampered = b"{\"service\":\"root\"}";
        let token = mint_agent_jwt_for_payload(
            "agent-007",
            "agent",
            payload_original,
            300,
            TEST_JWT_SECRET,
        )
        .unwrap();

        let err = verify_agent_jwt_with_payload(&token, TEST_JWT_SECRET, Some(payload_tampered))
            .unwrap_err();
        match err {
            CryptoError::PayloadHashMismatch { expected, actual } => {
                assert_eq!(expected, compute_payload_sha256(payload_original));
                assert_eq!(actual, compute_payload_sha256(payload_tampered));
            }
            other => panic!("Expected PayloadHashMismatch, got {:?}", other),
        }
    }

    #[test]
    fn test_jwt_verify_missing_payload_when_bound() {
        let payload = b"{\"service\":\"auth\"}";
        let token = mint_agent_jwt_for_payload("agent-007", "agent", payload, 300, TEST_JWT_SECRET)
            .unwrap();

        let err = verify_agent_jwt(&token, TEST_JWT_SECRET).unwrap_err();
        assert_eq!(err, CryptoError::MissingPayload);

        let err_explicit =
            verify_agent_jwt_with_payload(&token, TEST_JWT_SECRET, None).unwrap_err();
        assert_eq!(err_explicit, CryptoError::MissingPayload);
    }

    #[test]
    fn test_jwt_verify_unbound_with_payload_passed() {
        let token = mint_agent_jwt("agent-007", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let payload = b"{\"any\":\"data\"}";
        let claims = verify_agent_jwt_with_payload(&token, TEST_JWT_SECRET, Some(payload)).unwrap();
        assert_eq!(claims.sub, "agent-007");
        assert_eq!(claims.payload_sha256, None);
    }

    #[test]
    fn test_jwt_expired_token() {
        let token = mint_agent_jwt("agent-007", "agent", None, -30, TEST_JWT_SECRET).unwrap();
        let err = verify_agent_jwt(&token, TEST_JWT_SECRET).unwrap_err();
        assert_eq!(err, CryptoError::TokenExpired);
    }

    #[test]
    fn test_jwt_wrong_secret() {
        let token = mint_agent_jwt("agent-007", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let err = verify_agent_jwt(&token, b"wrong-secret-key-32-bytes-long!").unwrap_err();
        match err {
            CryptoError::InvalidToken(msg) => {
                assert!(msg.contains("Invalid token signature") || msg.contains("signature"));
            }
            other => panic!("Expected InvalidToken, got {:?}", other),
        }
    }

    #[test]
    fn test_jwt_tampered_signature() {
        let token = mint_agent_jwt("agent-007", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let mut tampered = token.into_bytes();
        let last = tampered.len() - 1;
        tampered[last] = if tampered[last] == b'a' { b'b' } else { b'a' };
        let tampered_str = String::from_utf8(tampered).unwrap();

        let err = verify_agent_jwt(&tampered_str, TEST_JWT_SECRET).unwrap_err();
        match err {
            CryptoError::InvalidToken(_) => {}
            other => panic!("Expected InvalidToken, got {:?}", other),
        }
    }

    #[test]
    fn test_jwt_tampered_claims() {
        let token = mint_agent_jwt("agent-007", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let parts: Vec<&str> = token.split('.').collect();
        assert_eq!(parts.len(), 3);

        // Tamper payload segment characters
        let mut tampered_payload = parts[1].to_string();
        let first_char = tampered_payload.chars().next().unwrap();
        let replacement = if first_char == 'e' { 'f' } else { 'e' };
        tampered_payload.replace_range(..1, &replacement.to_string());

        let tampered_token = format!("{}.{}.{}", parts[0], tampered_payload, parts[2]);
        let err = verify_agent_jwt(&tampered_token, TEST_JWT_SECRET).unwrap_err();
        match err {
            CryptoError::InvalidToken(_) => {}
            other => panic!("Expected InvalidToken, got {:?}", other),
        }
    }

    #[test]
    fn test_jwt_empty_sub_rejected() {
        let token = mint_agent_jwt("   ", "agent", None, 300, TEST_JWT_SECRET).unwrap();
        let err = verify_agent_jwt(&token, TEST_JWT_SECRET).unwrap_err();
        assert_eq!(
            err,
            CryptoError::InvalidClaims("Claim 'sub' cannot be empty".to_string())
        );
    }

    #[test]
    fn test_jwt_empty_role_rejected() {
        let token = mint_agent_jwt("agent-007", "", None, 300, TEST_JWT_SECRET).unwrap();
        let err = verify_agent_jwt(&token, TEST_JWT_SECRET).unwrap_err();
        assert_eq!(
            err,
            CryptoError::InvalidClaims("Claim 'role' cannot be empty".to_string())
        );
    }

    #[test]
    fn test_jwt_empty_secret_rejected() {
        let err_mint = mint_agent_jwt("agent-007", "agent", None, 300, b"").unwrap_err();
        assert_eq!(
            err_mint,
            CryptoError::InvalidKey("Secret cannot be empty".to_string())
        );

        let err_verify = verify_agent_jwt("token", b"").unwrap_err();
        assert_eq!(
            err_verify,
            CryptoError::InvalidKey("Secret key cannot be empty".to_string())
        );
    }

    #[test]
    fn test_jwt_whitespace_token_rejected() {
        let err = verify_agent_jwt("   \t\n  ", TEST_JWT_SECRET).unwrap_err();
        assert_eq!(
            err,
            CryptoError::InvalidToken("Token string is empty".to_string())
        );
    }

    #[test]
    fn test_jwt_malformed_token_rejected() {
        let err = verify_agent_jwt("not-a-valid-jwt-token", TEST_JWT_SECRET).unwrap_err();
        match err {
            CryptoError::InvalidToken(_) => {}
            other => panic!("Expected InvalidToken, got {:?}", other),
        }
    }

    #[test]
    fn test_hmac_sign_and_verify_valid() {
        let ts = 1731000000;
        let payload = b"{\"service\":\"status\"}";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);

        // Skew of 5 seconds (well within 30s)
        let now = ts + 5;
        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload,
            &sig,
            now,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_hmac_empty_body_valid() {
        let ts = 1731000000;
        let payload = b"";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);
        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload,
            &sig,
            ts,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert!(res.is_ok());
    }

    #[test]
    fn test_hmac_tampered_body() {
        let ts = 1731000000;
        let payload_orig = b"{\"action\":\"allow\"}";
        let payload_tampered = b"{\"action\":\"deny\"}";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload_orig);

        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload_tampered,
            &sig,
            ts,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert_eq!(res, Err(CryptoError::SignatureMismatch));
    }

    #[test]
    fn test_hmac_past_replay_expired() {
        let ts = 1000;
        let now = 1035; // 35s skew > 30s
        let payload = b"data";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);

        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload,
            &sig,
            now,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert_eq!(
            res,
            Err(CryptoError::TimestampExpired {
                skew_secs: 35,
                max_skew_secs: 30,
            })
        );
    }

    #[test]
    fn test_hmac_future_skew_expired() {
        let ts = 1040;
        let now = 1000; // Future timestamp by 40s > 30s
        let payload = b"data";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);

        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload,
            &sig,
            now,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert_eq!(
            res,
            Err(CryptoError::TimestampExpired {
                skew_secs: 40,
                max_skew_secs: 30,
            })
        );
    }

    #[test]
    fn test_hmac_boundary_skew_accepted() {
        let ts = 1000;
        let now = 1030; // Skew exactly 30s
        let payload = b"data";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);

        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            ts,
            payload,
            &sig,
            now,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert!(res.is_ok());

        // Boundary future skew exactly 30s
        let res_future = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            1030,
            payload,
            &sign_payload(TEST_HMAC_SECRET, 1030, payload),
            1000,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert!(res_future.is_ok());
    }

    #[test]
    fn test_hmac_wrong_secret() {
        let ts = 1731000000;
        let payload = b"test payload";
        let sig = sign_payload(TEST_HMAC_SECRET, ts, payload);

        let res = verify_payload_signature_at(
            b"different-secret-key-32-bytes!!",
            ts,
            payload,
            &sig,
            ts,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert_eq!(res, Err(CryptoError::SignatureMismatch));
    }

    #[test]
    fn test_hmac_invalid_timestamp_header() {
        let res = verify_payload_signature_headers_at(
            TEST_HMAC_SECRET,
            "not-a-number",
            "79c61f56884861a154d771302e545af899fb00f60568fe74ff228a8018955bda",
            b"body",
            1000,
            30,
        );
        match res {
            Err(CryptoError::InvalidTimestamp(msg)) => {
                assert!(msg.contains("Failed to parse timestamp header"));
            }
            other => panic!("Expected InvalidTimestamp, got {:?}", other),
        }
    }

    #[test]
    fn test_hmac_invalid_signature_hex() {
        let res = verify_payload_signature_at(
            TEST_HMAC_SECRET,
            1000,
            b"body",
            "not_valid_hex_characters_zzzz",
            1000,
            30,
        );
        match res {
            Err(CryptoError::InvalidSignatureFormat(msg)) => {
                assert!(msg.contains("Failed to decode signature hex"));
            }
            other => panic!("Expected InvalidSignatureFormat, got {:?}", other),
        }
    }

    #[test]
    fn test_hmac_invalid_signature_length() {
        // Hex encodes 4 bytes instead of 32 bytes
        let short_sig = "abcdef01";
        let res = verify_payload_signature_at(TEST_HMAC_SECRET, 1000, b"body", short_sig, 1000, 30);
        match res {
            Err(CryptoError::InvalidSignatureFormat(msg)) => {
                assert!(msg.contains("Signature must be 32 bytes"));
            }
            other => panic!("Expected InvalidSignatureFormat, got {:?}", other),
        }
    }

    #[test]
    fn test_hmac_empty_secret_rejected() {
        let res =
            verify_payload_signature_at(b"", 1000, b"body", "00".repeat(32).as_str(), 1000, 30);
        assert_eq!(
            res,
            Err(CryptoError::InvalidKey(
                "Secret cannot be empty".to_string()
            ))
        );
    }

    #[test]
    fn test_hmac_cross_language_test_vector() {
        let secret = b"sidecar_secret_key_12345";
        let ts: u64 = 1731000000;
        let body = b"{\"service\":\"core\"}";
        let sig = sign_payload(secret, ts, body);

        // Verified against Python:
        // hmac.new(b"sidecar_secret_key_12345", b"1731000000.{\"service\":\"core\"}", hashlib.sha256).hexdigest()
        assert_eq!(
            sig,
            "79c61f56884861a154d771302e545af899fb00f60568fe74ff228a8018955bda"
        );

        // Also verify that verify_payload_signature_headers accepts this exact vector
        let verify_res = verify_payload_signature_headers_at(
            secret,
            "1731000000",
            &sig,
            body,
            1731000005,
            DEFAULT_MAX_SKEW_SECS,
        );
        assert!(verify_res.is_ok());
    }
}
