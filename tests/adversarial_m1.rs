//! Comprehensive Adversarial & Fuzz Testing Suite for Milestone 1
//!
//! Stresses cryptographic invariants, configuration loaders, and Serde schemas
//! under adversarial, boundary, and hostile inputs.

use chrono::Utc;
use subtle::ConstantTimeEq;
use zero_trust_agentic_sidecar::*;

const TEST_SECRET_1: &[u8] = b"sidecar-adversarial-secret-32-b!";
const TEST_SECRET_2: &[u8] = b"jwt-adversarial-secret-32-bytes!";

// Simple deterministic XorShift64 PRNG for zero-dependency fuzz testing
struct FuzzRng(u64);

impl FuzzRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x5EED_CAFE_BABE_D00D
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }

    fn next_bytes(&mut self, len: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(len);
        while bytes.len() < len {
            let u = self.next_u64();
            for b in u.to_le_bytes() {
                if bytes.len() < len {
                    bytes.push(b);
                }
            }
        }
        bytes
    }

    fn choose<'a, T>(&mut self, slice: &'a [T]) -> &'a T {
        let idx = (self.next_u32() as usize) % slice.len();
        &slice[idx]
    }
}

// ==============================================================================
// 1. Cryptographic Replay Skew & Boundary Tests
// ==============================================================================
mod crypto_replay_skew_tests {
    use super::*;

    #[test]
    fn test_exact_skew_boundaries_past_and_future() {
        let secret = TEST_SECRET_1;
        let now: u64 = 1_700_000_000;
        let payload = b"{\"action\":\"test_skew\"}";

        // Max skew is 30 seconds
        let max_skew = DEFAULT_MAX_SKEW_SECS;
        assert_eq!(max_skew, 30);

        // Case 1: Past skew = 29s -> MUST SUCCEED
        let ts_past_29 = now - 29;
        let sig_past_29 = sign_payload(secret, ts_past_29, payload);
        assert!(
            verify_payload_signature_at(secret, ts_past_29, payload, &sig_past_29, now, max_skew)
                .is_ok(),
            "29s past skew should be accepted"
        );

        // Case 2: Past skew = 30s -> MUST SUCCEED (exact boundary)
        let ts_past_30 = now - 30;
        let sig_past_30 = sign_payload(secret, ts_past_30, payload);
        assert!(
            verify_payload_signature_at(secret, ts_past_30, payload, &sig_past_30, now, max_skew)
                .is_ok(),
            "30s past skew should be accepted (exact boundary)"
        );

        // Case 3: Past skew = 31s -> MUST FAIL (skew_secs = 31 > max_skew = 30)
        let ts_past_31 = now - 31;
        let sig_past_31 = sign_payload(secret, ts_past_31, payload);
        let err_past_31 =
            verify_payload_signature_at(secret, ts_past_31, payload, &sig_past_31, now, max_skew)
                .unwrap_err();
        assert_eq!(
            err_past_31,
            CryptoError::TimestampExpired {
                skew_secs: 31,
                max_skew_secs: 30,
            },
            "31s past skew must be rejected with TimestampExpired"
        );

        // Case 4: Future skew = 29s -> MUST SUCCEED
        let ts_future_29 = now + 29;
        let sig_future_29 = sign_payload(secret, ts_future_29, payload);
        assert!(
            verify_payload_signature_at(
                secret,
                ts_future_29,
                payload,
                &sig_future_29,
                now,
                max_skew
            )
            .is_ok(),
            "29s future skew should be accepted"
        );

        // Case 5: Future skew = 30s -> MUST SUCCEED (exact boundary)
        let ts_future_30 = now + 30;
        let sig_future_30 = sign_payload(secret, ts_future_30, payload);
        assert!(
            verify_payload_signature_at(
                secret,
                ts_future_30,
                payload,
                &sig_future_30,
                now,
                max_skew
            )
            .is_ok(),
            "30s future skew should be accepted (exact boundary)"
        );

        // Case 6: Future skew = 31s -> MUST FAIL (skew_secs = 31 > max_skew = 30)
        let ts_future_31 = now + 31;
        let sig_future_31 = sign_payload(secret, ts_future_31, payload);
        let err_future_31 = verify_payload_signature_at(
            secret,
            ts_future_31,
            payload,
            &sig_future_31,
            now,
            max_skew,
        )
        .unwrap_err();
        assert_eq!(
            err_future_31,
            CryptoError::TimestampExpired {
                skew_secs: 31,
                max_skew_secs: 30,
            },
            "31s future skew must be rejected with TimestampExpired"
        );
    }

    #[test]
    fn test_zero_skew_and_zero_max_skew() {
        let secret = TEST_SECRET_1;
        let now = 1_700_000_000;
        let payload = b"zero_skew_payload";
        let sig = sign_payload(secret, now, payload);

        // Exactly zero skew with max_skew = 0 must pass
        assert!(verify_payload_signature_at(secret, now, payload, &sig, now, 0).is_ok());

        // 1 second skew with max_skew = 0 must fail
        let err = verify_payload_signature_at(secret, now + 1, payload, &sig, now, 0).unwrap_err();
        assert_eq!(
            err,
            CryptoError::TimestampExpired {
                skew_secs: 1,
                max_skew_secs: 0
            }
        );
    }

    #[test]
    fn test_extreme_and_overflow_timestamps() {
        let secret = TEST_SECRET_1;
        let payload = b"overflow_check";

        // Epoch zero with now = 30
        let sig_0 = sign_payload(secret, 0, payload);
        assert!(verify_payload_signature_at(secret, 0, payload, &sig_0, 30, 30).is_ok());
        assert!(verify_payload_signature_at(secret, 0, payload, &sig_0, 31, 30).is_err());

        // Max u64 values
        let max_u64 = u64::MAX;
        let sig_max = sign_payload(secret, max_u64, payload);
        assert!(
            verify_payload_signature_at(secret, max_u64, payload, &sig_max, max_u64 - 30, 30)
                .is_ok()
        );
        assert!(
            verify_payload_signature_at(secret, max_u64, payload, &sig_max, max_u64 - 31, 30)
                .is_err()
        );

        // Extreme difference (u64::MAX vs 0) must not panic due to overflow
        let err_extreme =
            verify_payload_signature_at(secret, max_u64, payload, &sig_max, 0, 30).unwrap_err();
        match err_extreme {
            CryptoError::TimestampExpired { skew_secs, .. } => {
                assert_eq!(skew_secs, max_u64);
            }
            other => panic!("Expected TimestampExpired, got {:?}", other),
        }
    }

    #[test]
    fn test_timestamp_header_parsing_adversarial_inputs() {
        let secret = TEST_SECRET_1;
        let payload = b"header_test";
        let dummy_sig = "00".repeat(32);

        // Negative timestamp
        let err = verify_payload_signature_headers_at(secret, "-1", &dummy_sig, payload, 1000, 30)
            .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Float timestamp
        let err =
            verify_payload_signature_headers_at(secret, "1000.5", &dummy_sig, payload, 1000, 30)
                .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Hex timestamp string
        let err =
            verify_payload_signature_headers_at(secret, "0x3e8", &dummy_sig, payload, 1000, 30)
                .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Empty string
        let err = verify_payload_signature_headers_at(secret, "", &dummy_sig, payload, 1000, 30)
            .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Whitespace only
        let err =
            verify_payload_signature_headers_at(secret, "   \t\n", &dummy_sig, payload, 1000, 30)
                .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Garbage string
        let err = verify_payload_signature_headers_at(secret, "NaN", &dummy_sig, payload, 1000, 30)
            .unwrap_err();
        assert!(matches!(err, CryptoError::InvalidTimestamp(_)));

        // Leading/trailing whitespace around valid number is trimmed and accepted
        let ts_str = "  1000  ";
        let sig = sign_payload(secret, 1000, payload);
        assert!(
            verify_payload_signature_headers_at(secret, ts_str, &sig, payload, 1000, 30).is_ok()
        );
    }
}

// ==============================================================================
// 2. Algorithm Substitution & Confusion Attacks
// ==============================================================================
mod crypto_alg_substitution_tests {
    use super::*;

    #[test]
    fn test_alg_none_unsigned_token_rejected() {
        let secret = TEST_SECRET_2;

        // Base64Url for {"alg":"none","typ":"JWT"} is "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0"
        let header_none = "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0";
        // Create claims JSON
        let claims = AgentClaims {
            sub: "attacker".to_string(),
            role: "admin".to_string(),
            exp: 2_000_000_000,
            payload_sha256: None,
        };
        let claims_json = serde_json::to_string(&claims).unwrap();
        let claims_b64 = base64_url_encode(claims_json.as_bytes());

        // Token with empty signature: header.payload.
        let token_empty_sig = format!("{header_none}.{claims_b64}.");
        let err = verify_agent_jwt(&token_empty_sig, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "alg: none with empty signature must be rejected, got {:?}",
            err
        );

        // Token without trailing dot: header.payload
        let token_no_dot = format!("{header_none}.{claims_b64}");
        let err = verify_agent_jwt(&token_no_dot, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "alg: none without signature segment must be rejected, got {:?}",
            err
        );

        // Token with dummy signature: header.payload.bogus
        let token_bogus_sig = format!("{header_none}.{claims_b64}.bogus_signature");
        let err = verify_agent_jwt(&token_bogus_sig, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "alg: none with arbitrary signature must be rejected, got {:?}",
            err
        );
    }

    #[test]
    fn test_alg_none_case_variations_rejected() {
        let secret = TEST_SECRET_2;
        let claims = AgentClaims {
            sub: "attacker".to_string(),
            role: "admin".to_string(),
            exp: 2_000_000_000,
            payload_sha256: None,
        };
        let claims_b64 = base64_url_encode(serde_json::to_string(&claims).unwrap().as_bytes());

        // Test "None", "NONE", "nOnE"
        for alg in ["None", "NONE", "nOnE"] {
            let header_json = format!(r#"{{"alg":"{alg}","typ":"JWT"}}"#);
            let header_b64 = base64_url_encode(header_json.as_bytes());
            let token = format!("{header_b64}.{claims_b64}.");
            let err = verify_agent_jwt(&token, secret).unwrap_err();
            assert!(
                matches!(err, CryptoError::InvalidToken(_)),
                "alg {alg} must be rejected, got {:?}",
                err
            );
        }
    }

    #[test]
    fn test_algorithm_confusion_hs384_hs512_rs256_rejected() {
        let secret = TEST_SECRET_2;
        let claims = AgentClaims {
            sub: "attacker".to_string(),
            role: "admin".to_string(),
            exp: 2_000_000_000,
            payload_sha256: None,
        };

        // Mint HS384 token using jsonwebtoken directly
        let header_384 = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS384);
        let token_384 = jsonwebtoken::encode(
            &header_384,
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(secret),
        )
        .unwrap();

        // Must be rejected by verify_agent_jwt because it enforces Algorithm::HS256
        let err = verify_agent_jwt(&token_384, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "HS384 token must be rejected, got {:?}",
            err
        );

        // Mint HS512 token
        let header_512 = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS512);
        let token_512 = jsonwebtoken::encode(
            &header_512,
            &claims,
            &jsonwebtoken::EncodingKey::from_secret(secret),
        )
        .unwrap();

        let err = verify_agent_jwt(&token_512, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "HS512 token must be rejected, got {:?}",
            err
        );

        // Header claiming RS256
        let header_rs256 = base64_url_encode(br#"{"alg":"RS256","typ":"JWT"}"#);
        let claims_b64 = base64_url_encode(serde_json::to_string(&claims).unwrap().as_bytes());
        let token_rs256 = format!("{header_rs256}.{claims_b64}.c2lnbmF0dXJl");
        let err = verify_agent_jwt(&token_rs256, secret).unwrap_err();
        assert!(
            matches!(err, CryptoError::InvalidToken(_)),
            "RS256 token must be rejected, got {:?}",
            err
        );
    }

    fn base64_url_encode(data: &[u8]) -> String {
        const CHARSET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::new();
        let mut i = 0;
        while i < data.len() {
            let b0 = data[i] as u32;
            let b1 = if i + 1 < data.len() {
                data[i + 1] as u32
            } else {
                0
            };
            let b2 = if i + 2 < data.len() {
                data[i + 2] as u32
            } else {
                0
            };

            let triple = (b0 << 16) | (b1 << 8) | b2;

            out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
            if i + 1 < data.len() {
                out.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
            }
            if i + 2 < data.len() {
                out.push(CHARSET[(triple & 0x3F) as usize] as char);
            }
            i += 3;
        }
        out
    }
}

// ==============================================================================
// 3. Malformed JWT & Serde Schema Edge Cases
// ==============================================================================
mod crypto_jwt_malformed_tests {
    use super::*;

    #[test]
    fn test_jwt_structural_malformations() {
        let secret = TEST_SECRET_2;

        let malformed_inputs = [
            "",
            "   ",
            "\n\t",
            ".",
            "..",
            "...",
            "header",
            "header.payload",
            "header.payload.sig.extra",
            "header.payload.sig.extra.more",
            "a.b.c",
            "not-base-64!.not-base-64!.not-base-64!",
            "eyJhbGciOiJIUzI1NiJ9..sig",
            ".eyJzdWIiOiJhZ2VudCJ9.sig",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJhZ2VudCJ9.",
        ];

        for input in malformed_inputs {
            let res = verify_agent_jwt(input, secret);
            assert!(
                res.is_err(),
                "Malformed token '{input}' was unexpectedly accepted"
            );
        }
    }

    #[test]
    fn test_jwt_claims_schema_violations() {
        let secret = TEST_SECRET_2;

        // Helper to sign custom JSON claims payload with valid HS256 header
        let sign_custom_claims = |claims_json: &str| -> String {
            let header = jsonwebtoken::Header::new(jsonwebtoken::Algorithm::HS256);
            let header_str = serde_json::to_string(&header).unwrap();
            let h_b64 = hex_to_b64_custom(header_str.as_bytes());
            let c_b64 = hex_to_b64_custom(claims_json.as_bytes());
            let to_sign = format!("{h_b64}.{c_b64}");

            let mut mac = <hmac::Hmac<sha2::Sha256> as hmac::Mac>::new_from_slice(secret).unwrap();
            hmac::Mac::update(&mut mac, to_sign.as_bytes());
            let sig = hmac::Mac::finalize(mac).into_bytes();
            let sig_b64 = hex_to_b64_custom(&sig);
            format!("{to_sign}.{sig_b64}")
        };

        // Sub as integer instead of string
        let token = sign_custom_claims(r#"{"sub":12345,"role":"admin","exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Role as array instead of string
        let token = sign_custom_claims(r#"{"sub":"agent","role":["admin"],"exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Exp as string instead of int
        let token = sign_custom_claims(r#"{"sub":"agent","role":"admin","exp":"2000000000"}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Missing sub
        let token = sign_custom_claims(r#"{"role":"admin","exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Missing role
        let token = sign_custom_claims(r#"{"sub":"agent","exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Missing exp
        let token = sign_custom_claims(r#"{"sub":"agent","role":"admin"}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Null sub
        let token = sign_custom_claims(r#"{"sub":null,"role":"admin","exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());

        // Null role
        let token = sign_custom_claims(r#"{"sub":"agent","role":null,"exp":2000000000}"#);
        assert!(verify_agent_jwt(&token, secret).is_err());
    }

    #[test]
    fn test_zero_leeway_exact_expiration() {
        let secret = TEST_SECRET_2;
        let now = Utc::now().timestamp() as usize;

        // Expired 1 second ago (exp = now - 1)
        let token_expired =
            mint_agent_jwt_with_exp("agent", "admin", None, now - 1, secret).unwrap();
        assert_eq!(
            verify_agent_jwt(&token_expired, secret).unwrap_err(),
            CryptoError::TokenExpired
        );

        // Token active for next 10 seconds
        let token_active =
            mint_agent_jwt_with_exp("agent", "admin", None, now + 10, secret).unwrap();
        assert!(verify_agent_jwt(&token_active, secret).is_ok());
    }

    fn hex_to_b64_custom(data: &[u8]) -> String {
        const CHARSET: &[u8; 64] =
            b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::new();
        let mut i = 0;
        while i < data.len() {
            let b0 = data[i] as u32;
            let b1 = if i + 1 < data.len() {
                data[i + 1] as u32
            } else {
                0
            };
            let b2 = if i + 2 < data.len() {
                data[i + 2] as u32
            } else {
                0
            };

            let triple = (b0 << 16) | (b1 << 8) | b2;

            out.push(CHARSET[((triple >> 18) & 0x3F) as usize] as char);
            out.push(CHARSET[((triple >> 12) & 0x3F) as usize] as char);
            if i + 1 < data.len() {
                out.push(CHARSET[((triple >> 6) & 0x3F) as usize] as char);
            }
            if i + 2 < data.len() {
                out.push(CHARSET[(triple & 0x3F) as usize] as char);
            }
            i += 3;
        }
        out
    }
}

// ==============================================================================
// 4. Altered Signatures & Constant-Time Comparisons
// ==============================================================================
mod crypto_signature_and_constant_time_tests {
    use super::*;

    #[test]
    fn test_hmac_signature_bit_flipping_at_all_positions() {
        let secret = TEST_SECRET_1;
        let ts = 1_700_000_000;
        let payload = b"critical-request-body";
        let valid_sig = sign_payload(secret, ts, payload);
        assert_eq!(valid_sig.len(), 64);

        let mut sig_bytes = hex::decode(&valid_sig).unwrap();
        assert_eq!(sig_bytes.len(), 32);

        // Flip bit in byte 0 (first byte)
        sig_bytes[0] ^= 0x01;
        let tampered_0 = hex::encode(&sig_bytes);
        assert_eq!(
            verify_payload_signature_at(secret, ts, payload, &tampered_0, ts, 30),
            Err(CryptoError::SignatureMismatch)
        );
        sig_bytes[0] ^= 0x01; // Restore

        // Flip bit in byte 15 (middle byte)
        sig_bytes[15] ^= 0x80;
        let tampered_15 = hex::encode(&sig_bytes);
        assert_eq!(
            verify_payload_signature_at(secret, ts, payload, &tampered_15, ts, 30),
            Err(CryptoError::SignatureMismatch)
        );
        sig_bytes[15] ^= 0x80; // Restore

        // Flip bit in byte 31 (last byte)
        sig_bytes[31] ^= 0x01;
        let tampered_31 = hex::encode(&sig_bytes);
        assert_eq!(
            verify_payload_signature_at(secret, ts, payload, &tampered_31, ts, 30),
            Err(CryptoError::SignatureMismatch)
        );
        sig_bytes[31] ^= 0x01; // Restore

        // Ensure restored signature is valid again
        assert!(
            verify_payload_signature_at(secret, ts, payload, &hex::encode(&sig_bytes), ts, 30)
                .is_ok()
        );
    }

    #[test]
    fn test_hmac_signature_formatting_anomalies() {
        let secret = TEST_SECRET_1;
        let ts = 1_700_000_000;
        let payload = b"data";
        let valid_sig = sign_payload(secret, ts, payload);

        // Uppercase hex must be accepted (hex::decode is case-insensitive)
        let upper_sig = valid_sig.to_ascii_uppercase();
        assert!(verify_payload_signature_at(secret, ts, payload, &upper_sig, ts, 30).is_ok());

        // Odd number of hex characters (63 chars)
        let odd_sig = &valid_sig[..63];
        let err = verify_payload_signature_at(secret, ts, payload, odd_sig, ts, 30).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidSignatureFormat(_)));

        // Truncated signature (32 hex chars = 16 bytes instead of 32 bytes)
        let short_sig = &valid_sig[..32];
        let err = verify_payload_signature_at(secret, ts, payload, short_sig, ts, 30).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidSignatureFormat(_)));

        // Extended signature (66 hex chars = 33 bytes)
        let long_sig = format!("{valid_sig}00");
        let err = verify_payload_signature_at(secret, ts, payload, &long_sig, ts, 30).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidSignatureFormat(_)));

        // Non-hex characters
        let non_hex_sig = format!("{}zz", &valid_sig[..62]);
        let err =
            verify_payload_signature_at(secret, ts, payload, &non_hex_sig, ts, 30).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidSignatureFormat(_)));
    }

    #[test]
    fn test_payload_sha256_binding_edge_cases() {
        let secret = TEST_SECRET_2;
        let payload = b"{\"service\":\"email\",\"to\":\"sec@example.com\"}";
        let sha256_lower = compute_payload_sha256(payload);
        let sha256_upper = sha256_lower.to_ascii_uppercase();

        // Uppercase SHA-256 in token claim must match lowercase computed hash
        let token_upper =
            mint_agent_jwt_with_exp("agent", "admin", Some(&sha256_upper), 2_000_000_000, secret)
                .unwrap();
        let verified_claims =
            verify_agent_jwt_with_payload(&token_upper, secret, Some(payload)).unwrap();
        assert_eq!(verified_claims.payload_sha256, Some(sha256_upper.clone()));

        // Mismatched hash: single byte altered
        let mut tampered_hash = sha256_lower.clone();
        let repl = if tampered_hash.starts_with('a') {
            "b"
        } else {
            "a"
        };
        tampered_hash.replace_range(..1, repl);
        let token_tampered = mint_agent_jwt_with_exp(
            "agent",
            "admin",
            Some(&tampered_hash),
            2_000_000_000,
            secret,
        )
        .unwrap();
        let err =
            verify_agent_jwt_with_payload(&token_tampered, secret, Some(payload)).unwrap_err();
        assert!(matches!(err, CryptoError::PayloadHashMismatch { .. }));

        // Hash length mismatch
        let short_hash = &sha256_lower[..60];
        let token_short =
            mint_agent_jwt_with_exp("agent", "admin", Some(short_hash), 2_000_000_000, secret)
                .unwrap();
        let err = verify_agent_jwt_with_payload(&token_short, secret, Some(payload)).unwrap_err();
        assert!(matches!(err, CryptoError::PayloadHashMismatch { .. }));

        // Empty payload bound
        let empty_payload = b"";
        let empty_sha = compute_payload_sha256(empty_payload);
        assert_eq!(
            empty_sha,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        let token_empty =
            mint_agent_jwt_with_exp("agent", "admin", Some(&empty_sha), 2_000_000_000, secret)
                .unwrap();
        assert!(verify_agent_jwt_with_payload(&token_empty, secret, Some(empty_payload)).is_ok());
    }

    #[test]
    fn test_constant_time_comparison_semantics() {
        // Confirm subtle ConstantTimeEq correctly distinguishes matching and non-matching slices
        let a = b"0123456789abcdef0123456789abcdef";
        let b = b"0123456789abcdef0123456789abcdef";
        let c = b"0123456789abcdef0123456789abcdeg"; // last byte differs

        assert!(bool::from(a.ct_eq(b)));
        assert!(!bool::from(a.ct_eq(c)));
    }
}

// ==============================================================================
// 5. Config Loader Edge Cases & Malformed Environment Variables
// ==============================================================================
mod config_edge_case_tests {
    use super::*;

    #[test]
    fn test_config_empty_secrets_variations() {
        let whitespace_variants = ["", " ", "   ", "\t", "\n", "\r\n", " \t \n "];

        for ws in whitespace_variants {
            // Empty SIDECAR_SECRET
            let vars = [
                ("SIDECAR_SECRET", ws),
                ("JWT_SECRET", "valid-jwt-secret-16b"),
            ];
            let err = Config::from_vars(vars).unwrap_err();
            assert_eq!(err, ConfigError::EmptyVar("SIDECAR_SECRET"));

            // Empty JWT_SECRET
            let vars = [("SIDECAR_SECRET", "valid-sidecar-key"), ("JWT_SECRET", ws)];
            let err = Config::from_vars(vars).unwrap_err();
            assert_eq!(err, ConfigError::EmptyVar("JWT_SECRET"));
        }
    }

    #[test]
    fn test_config_sidecar_addr_edge_cases() {
        let invalid_addrs = [
            "127.0.0.1:65536",     // port overflow (> 65535)
            "127.0.0.1:-1",        // negative port
            "127.0.0.1:abc",       // non-numeric port
            "127.0.0.1",           // missing port
            ":8080",               // missing host
            "localhost:8080",      // hostname instead of IP
            "http://127.0.0.1:80", // URI scheme prefix
            "256.0.0.1:8080",      // invalid IPv4 octet
            "1.2.3.4.5:8080",      // too many octets
            "127.0.0.1:8080/path", // trailing path
        ];

        for addr in invalid_addrs {
            let vars = [
                ("SIDECAR_SECRET", "valid-sidecar-secret-16b"),
                ("JWT_SECRET", "valid-jwt-secret-16b"),
                ("SIDECAR_ADDR", addr),
            ];
            let err = Config::from_vars(vars).unwrap_err();
            assert!(
                matches!(
                    err,
                    ConfigError::InvalidValue {
                        var: "SIDECAR_ADDR",
                        ..
                    }
                ),
                "Expected InvalidValue for SIDECAR_ADDR='{addr}', got {:?}",
                err
            );
        }

        let valid_addrs = [
            "127.0.0.1:1",
            "127.0.0.1:65535",
            "0.0.0.0:8080",
            "[::1]:8080",
            "[::]:9000",
        ];

        for addr in valid_addrs {
            let vars = [
                ("SIDECAR_SECRET", "valid-sidecar-secret-16b"),
                ("JWT_SECRET", "valid-jwt-secret-16b"),
                ("SIDECAR_ADDR", addr),
            ];
            let cfg = Config::from_vars(vars);
            assert!(
                cfg.is_ok(),
                "Valid address '{addr}' failed to parse: {:?}",
                cfg.err()
            );
        }
    }

    #[test]
    fn test_config_large_and_unicode_secrets() {
        // 64 KB secret string
        let large_secret = "a".repeat(65536);
        let vars = [
            ("SIDECAR_SECRET", large_secret.as_str()),
            ("JWT_SECRET", "valid-jwt-secret-16b"),
        ];
        let cfg = Config::from_vars(vars).unwrap();
        assert_eq!(cfg.sidecar_secret.len(), 65536);

        // Multi-byte Unicode secrets (Chinese, Emoji)
        let unicode_secret = "🔑安全密钥-sidecar-🚀";
        let vars = [
            ("SIDECAR_SECRET", unicode_secret),
            ("JWT_SECRET", "valid-jwt-secret-16b"),
        ];
        let cfg = Config::from_vars(vars).unwrap();
        assert_eq!(cfg.sidecar_secret, unicode_secret);

        // Outbound HMAC signing using unicode secret
        let sig = sign_payload(cfg.sidecar_secret.as_bytes(), 1000, b"data");
        assert!(verify_payload_signature_at(
            cfg.sidecar_secret.as_bytes(),
            1000,
            b"data",
            &sig,
            1000,
            30
        )
        .is_ok());
    }

    #[test]
    fn test_config_anthropic_api_key_sanitization_matrix() {
        let test_cases = [
            (None, None),
            (Some(""), None),
            (Some("   "), None),
            (Some("\t\n"), None),
            (Some("sk-ant-test"), Some("sk-ant-test")),
            (Some("  sk-ant-padded  "), Some("sk-ant-padded")),
        ];

        for (input, expected) in test_cases {
            let mut vars = vec![
                ("SIDECAR_SECRET", "valid-sidecar-secret-16b"),
                ("JWT_SECRET", "valid-jwt-secret-16b"),
            ];
            if let Some(key) = input {
                vars.push(("ANTHROPIC_API_KEY", key));
            }
            let cfg = Config::from_vars(vars).unwrap();
            assert_eq!(cfg.anthropic_api_key.as_deref(), expected);
        }
    }
}

// ==============================================================================
// 6. Serde Schemas & Unusual Unicode DTO Edge Cases
// ==============================================================================
mod serde_edge_case_tests {
    use super::*;

    #[test]
    fn test_unusual_unicode_in_tool_execute_request() {
        let weird_json = r#"{
            "session_id": "会话-جلسة-🔑-\u0000-\u200B-end",
            "endpoint": "/api/v1/系统/状态",
            "method": "POST",
            "payload": {
                "key_with_emoji_🚀": "值-value-\t\n\r",
                "nested": {
                    "rtl": "العربية",
                    "cjk": "日本語テスト"
                }
            },
            "context": {
                "user_prompt": "Prompt with zero-width space: test\u200Bword and null byte \u0000",
                "untrusted_inputs": ["<script>alert('xss')</script>", "DROP TABLE users;--"]
            }
        }"#;

        let req: ToolExecuteRequest = serde_json::from_str(weird_json).unwrap();
        assert!(req.session_id.contains("会话"));
        assert!(req.session_id.contains("جلسة"));
        assert_eq!(req.endpoint, "/api/v1/系统/状态");
        assert!(req.payload.contains_key("key_with_emoji_🚀"));
        assert_eq!(req.context.untrusted_inputs.len(), 2);

        // Roundtrip serialization
        let serialized = serde_json::to_string(&req).unwrap();
        let deser: ToolExecuteRequest = serde_json::from_str(&serialized).unwrap();
        assert_eq!(req, deser);
    }

    #[test]
    fn test_null_fields_in_payload_map() {
        // Tool payload can contain fields with null value
        let json = r#"{
            "session_id": "sess-null-test",
            "endpoint": "/api/v1/system/status",
            "payload": {
                "service": null,
                "timeout": 30,
                "debug": false
            }
        }"#;

        let req: ToolExecuteRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.payload.get("service"), Some(&serde_json::Value::Null));
        assert_eq!(req.payload.get("timeout"), Some(&serde_json::json!(30)));
    }

    #[test]
    fn test_null_payload_object_fails_deserialization() {
        // If the entire payload property is null (not a map), deserialization should fail
        let json = r#"{
            "session_id": "sess-null-test",
            "endpoint": "/api/v1/system/status",
            "payload": null
        }"#;

        let res: Result<ToolExecuteRequest, _> = serde_json::from_str(json);
        assert!(
            res.is_err(),
            "payload: null must fail deserialization because schema expects a map"
        );
    }

    #[test]
    fn test_anthropic_extract_all_text_complex_multilingual() {
        let req = AnthropicMessagesRequest {
            model: "claude-3-5-sonnet".to_string(),
            system: Some("系统提示: システムプロンプト: System Prompt".to_string()),
            messages: vec![
                AnthropicMessage {
                    role: "user".to_string(),
                    content: MessageContent::Text("مرحبا بالعالم (Hello Arabic)".to_string()),
                },
                AnthropicMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(vec![
                        ContentBlock {
                            block_type: "text".to_string(),
                            text: Some("English answer with emoji 🤖".to_string()),
                            content: None,
                        },
                        ContentBlock {
                            block_type: "tool_use".to_string(),
                            text: None,
                            content: Some(serde_json::json!("Tool execution output")),
                        },
                        ContentBlock {
                            block_type: "tool_use_non_string".to_string(),
                            text: None,
                            content: Some(serde_json::json!({"code": 200})),
                        },
                    ]),
                },
            ],
            max_tokens: Some(4096),
            stream: Some(true),
            temperature: Some(0.7),
        };

        let text = req.extract_all_text();
        assert!(text.contains("系统提示"));
        assert!(text.contains("システムプロンプト"));
        assert!(text.contains("مرحبا بالعالم"));
        assert!(text.contains("English answer with emoji 🤖"));
        assert!(text.contains("Tool execution output"));
    }

    #[test]
    fn test_all_deny_rules_roundtrip_and_statuses() {
        let all_rules = [
            (DenyRule::Malformed, "malformed", 400),
            (DenyRule::UnauthorizedToken, "unauthorized_token", 403),
            (DenyRule::EndpointDenied, "endpoint_denied", 403),
            (DenyRule::UnauthorizedRole, "unauthorized_role", 403),
            (DenyRule::MissingArg, "missing_arg", 403),
            (DenyRule::ConstraintFailed, "constraint_failed", 403),
            (DenyRule::ModelUnavailable, "model_unavailable", 403),
            (DenyRule::ModelRiskHigh, "model_risk_high", 403),
            (DenyRule::BackendError, "backend_error", 502),
        ];

        for (rule, expected_str, expected_code) in all_rules {
            assert_eq!(rule.as_str(), expected_str);
            assert_eq!(rule.to_string(), expected_str);
            assert_eq!(rule.http_status().as_u16(), expected_code);

            let json = serde_json::to_string(&rule).unwrap();
            assert_eq!(json, format!("\"{expected_str}\""));
            let parsed: DenyRule = serde_json::from_str(&json).unwrap();
            assert_eq!(rule, parsed);
        }
    }
}

// ==============================================================================
// 7. Property-Based Random Fuzzing Harness (1000 Iterations)
// ==============================================================================
mod fuzz_stress_tests {
    use super::*;

    #[test]
    fn test_fuzz_jwt_verification_never_panics_on_arbitrary_bytes() {
        let mut rng = FuzzRng::new(0x1337_C0DE);
        let secret = TEST_SECRET_2;

        for _ in 0..1000 {
            let len = (rng.next_u32() as usize % 256) + 1;
            let raw_bytes = rng.next_bytes(len);
            let raw_str = String::from_utf8_lossy(&raw_bytes);

            // Should never panic; must safely return Err(CryptoError)
            let result = std::panic::catch_unwind(|| verify_agent_jwt(&raw_str, secret));
            assert!(
                result.is_ok(),
                "verify_agent_jwt panicked on fuzz input: {:?}",
                raw_str
            );
        }
    }

    #[test]
    fn test_fuzz_hmac_verification_invariants_1000_iterations() {
        let mut rng = FuzzRng::new(0xCAFE_BABE);
        let secret = TEST_SECRET_1;

        for _ in 0..1000 {
            let now = 1_700_000_000u64;
            // Generate random skew between 0 and 100 seconds
            let skew = (rng.next_u32() % 100) as u64;
            let is_past = rng.next_u32().is_multiple_of(2);
            let ts = if is_past {
                now.saturating_sub(skew)
            } else {
                now.saturating_add(skew)
            };

            let payload_len = (rng.next_u32() as usize % 64) + 1;
            let payload = rng.next_bytes(payload_len);

            let sig = sign_payload(secret, ts, &payload);

            let res = verify_payload_signature_at(secret, ts, &payload, &sig, now, 30);
            if skew <= 30 {
                assert!(
                    res.is_ok(),
                    "Valid skew {skew}s (is_past={is_past}) was rejected: {:?}",
                    res
                );
            } else {
                assert_eq!(
                    res,
                    Err(CryptoError::TimestampExpired {
                        skew_secs: skew,
                        max_skew_secs: 30
                    }),
                    "Skew {skew}s (is_past={is_past}) should have expired"
                );
            }
        }
    }

    #[test]
    fn test_fuzz_config_from_vars_never_panics() {
        let mut rng = FuzzRng::new(0xDEAD_BEEF);

        for _ in 0..500 {
            let var_count = (rng.next_u32() % 10) as usize;
            let mut vars = Vec::with_capacity(var_count);

            let sample_keys = [
                "SIDECAR_SECRET",
                "JWT_SECRET",
                "SIDECAR_ADDR",
                "ANTHROPIC_API_KEY",
                "UPSTREAM_LLM_URL",
                "FLASK_BACKEND_URL",
                "POLICY_PATH",
                "RANDOM_VAR_1",
                "RANDOM_VAR_2",
            ];

            for _ in 0..var_count {
                let k = *rng.choose(&sample_keys);
                let val_len = (rng.next_u32() % 32) as usize;
                let v = String::from_utf8_lossy(&rng.next_bytes(val_len)).to_string();
                vars.push((k, v));
            }

            let result = std::panic::catch_unwind(|| {
                let _ = Config::from_vars(vars);
            });
            assert!(result.is_ok(), "Config::from_vars panicked on random input");
        }
    }
}
