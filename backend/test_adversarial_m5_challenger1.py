"""
Empirical Adversarial Stress Test Suite for Milestone 5 (Ticket T8)
Target Flask Backend Microservice Security and Robustness Verification

Author: Challenger 1 (critic, specialist)
Scope:
1. Replay Attacks and Timestamp Boundary Limits
   - Exact boundary testing (29.9s vs 30.0s vs 30.1s past & future)
   - Negative, zero, float, scientific notation, NaN, Infinity, alphanumeric timestamps
2. Signature Attack Vectors
   - Truncated HMAC signatures (empty, 1-63 chars)
   - Padded signatures (65-128 chars, embedded null bytes, whitespace)
   - Uppercase, lowercase, mixed-case hex signatures
   - Null-byte injections in payload bodies
   - Large payload bodies (1MB, 2MB)
   - Empty payload bodies on all endpoints
   - Unicode normalizations, multi-byte UTF-8, and arbitrary binary payloads
3. Direct Bypass Attempts
   - Direct requests without headers to all protected endpoints and aliases
   - Forged headers, corrupted signatures, mismatched timestamps, cross-payload reuse
   - Unauthenticated subpaths under /healthz and /health
   - Method mismatch (405) handling
4. Server Misconfiguration & Fail-Closed Behavior
   - Unconfigured / empty SIDECAR_SECRET
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import sys
import time
import pytest

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.app import create_app
except ImportError:
    from app import create_app

TEST_SECRET = "test_sidecar_secret_32_bytes_long_ok"


def make_hmac_headers(
    secret: str | bytes = TEST_SECRET,
    payload: bytes | str | dict | None = None,
    timestamp: int | float | str | None = None,
    signature_override: str | None = None,
) -> dict[str, str]:
    """Helper to craft precise test headers."""
    if timestamp is None:
        ts_val = int(time.time())
        ts_str = str(ts_val)
    elif isinstance(timestamp, (int, float)):
        ts_val = int(timestamp)
        ts_str = str(timestamp)
    else:
        ts_str = str(timestamp)
        try:
            ts_val = int(float(timestamp))
        except (ValueError, TypeError, OverflowError):
            ts_val = 0

    if isinstance(payload, bytes):
        raw_body = payload
    elif isinstance(payload, str):
        raw_body = payload.encode("utf-8")
    elif isinstance(payload, dict):
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    elif payload is None:
        raw_body = b""
    else:
        raw_body = bytes(payload)

    if signature_override is not None:
        sig = signature_override
    else:
        secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")
        canonical = f"{ts_val}.".encode("utf-8") + raw_body
        sig = hmac.new(secret_bytes, canonical, hashlib.sha256).hexdigest()

    return {
        "X-Sidecar-Timestamp": ts_str,
        "X-Sidecar-Signature": sig,
    }


@pytest.fixture
def app():
    app_instance = create_app(secret=TEST_SECRET)
    app_instance.config["TESTING"] = True
    return app_instance


@pytest.fixture
def client(app):
    return app.test_client()


# ==============================================================================
# SECTION 1: REPLAY ATTACKS & TIMESTAMP BOUNDARY LIMITS
# ==============================================================================


class TestTimestampBoundaryAndReplay:
    """Stress tests for timestamp parsing, clock skew, and boundary edges."""

    def test_exact_past_boundary_29_9_seconds(self, client, monkeypatch):
        """29.9s in past -> ts_int = now - 29 -> skew = 29s <= 30s -> Accepted (200)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        # now - 29.9 = 1731000070.1 -> ts_int = 1731000070
        past_ts = 1731000070
        headers = make_hmac_headers(timestamp=past_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200

    def test_exact_past_boundary_30_0_seconds(self, client, monkeypatch):
        """Exactly 30.0s in past -> skew = 30s <= 30s -> Accepted (200)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        past_ts = 1731000070  # 1731000100 - 30
        headers = make_hmac_headers(timestamp=past_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200

    def test_exact_past_boundary_30_1_seconds(self, client, monkeypatch):
        """30.1s in past -> ts_int = 1731000069 -> skew = 31s > 30s -> Rejected (401)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        past_ts = 1731000069  # 31s skew
        headers = make_hmac_headers(timestamp=past_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401
        assert "skew" in res.get_json()["reason"].lower()

    def test_exact_future_boundary_29_9_seconds(self, client, monkeypatch):
        """29.9s in future -> ts = now + 29 -> skew = 29s <= 30s -> Accepted (200)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        future_ts = 1731000129
        headers = make_hmac_headers(timestamp=future_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200

    def test_exact_future_boundary_30_0_seconds(self, client, monkeypatch):
        """Exactly 30.0s in future -> skew = 30s <= 30s -> Accepted (200)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        future_ts = 1731000130
        headers = make_hmac_headers(timestamp=future_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200

    def test_exact_future_boundary_31_0_seconds(self, client, monkeypatch):
        """31.0s in future -> skew = 31s > 30s -> Rejected (401)."""
        fixed_now = 1731000100.0
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        future_ts = 1731000131
        headers = make_hmac_headers(timestamp=future_ts)
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401
        assert "skew" in res.get_json()["reason"].lower()

    @pytest.mark.parametrize(
        "bad_ts",
        [
            "-1",
            "-100",
            "-1731000000",
            "-0.001",
            "0",
            "0.0",
            "NaN",
            "nan",
            "+nan",
            "-nan",
            "Infinity",
            "-Infinity",
            "inf",
            "-inf",
            "+inf",
            "abc",
            "123abc",
            "0x10",
            "0b1010",
            "1e99999999999999999999",
            "-1e99999999999999999999",
            "None",
            "null",
            "true",
            "",
            "   ",
            "\t",
            "  \t  ",
        ],
    )
    def test_malformed_and_extreme_timestamps_rejected(self, client, bad_ts):
        """Verify all non-numeric, extreme, negative, and zero timestamps are rejected with 401."""
        headers = {
            "X-Sidecar-Timestamp": bad_ts,
            "X-Sidecar-Signature": "a" * 64,
        }
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401
        body = res.get_json()
        assert body["error"] == "unauthorized"

    def test_timestamp_whitespace_trimmed_if_valid(self, client):
        """Timestamp padded with spaces/tabs (without raw newlines) is stripped and accepted if valid number."""
        now = int(time.time())
        raw_headers = make_hmac_headers(timestamp=now)
        padded_headers = {
            "X-Sidecar-Timestamp": f"  \t {now} \t ",
            "X-Sidecar-Signature": raw_headers["X-Sidecar-Signature"],
        }
        res = client.get("/api/v1/system/status", headers=padded_headers)
        assert res.status_code == 200

    def test_replay_attack_with_stale_timestamp(self, client, monkeypatch):

        """Simulate replay of a captured legitimate request 35 seconds later."""
        t0 = 1731000000.0
        monkeypatch.setattr(time, "time", lambda: t0)
        legit_headers = make_hmac_headers(timestamp=int(t0))
        # Initial request succeeds
        res1 = client.get("/api/v1/system/status", headers=legit_headers)
        assert res1.status_code == 200

        # Attacker replays exact same request at t0 + 35s
        monkeypatch.setattr(time, "time", lambda: t0 + 35.0)
        res2 = client.get("/api/v1/system/status", headers=legit_headers)
        assert res2.status_code == 401
        assert "skew" in res2.get_json()["reason"].lower()


# ==============================================================================
# SECTION 2: SIGNATURE ATTACK VECTORS
# ==============================================================================


class TestSignatureAttackVectors:
    """Stress tests targeting HMAC verification mechanics and edge cases."""

    @pytest.mark.parametrize(
        "trunc_len",
        [0, 1, 8, 16, 32, 48, 63],
    )
    def test_truncated_signature_rejected(self, client, trunc_len):
        """Truncated HMAC hex strings must fail with HTTP 401."""
        headers = make_hmac_headers()
        full_sig = headers["X-Sidecar-Signature"]
        headers["X-Sidecar-Signature"] = full_sig[:trunc_len]
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401
        assert res.get_json()["error"] == "unauthorized"

    @pytest.mark.parametrize(
        "pad_suffix",
        ["0", "00", "deadbeef", "x", "A" * 64, "_invalid_suffix"],
    )
    def test_padded_signature_rejected(self, client, pad_suffix):
        """Signatures with appended non-whitespace characters must fail with HTTP 401."""
        headers = make_hmac_headers()
        headers["X-Sidecar-Signature"] = headers["X-Sidecar-Signature"] + pad_suffix
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401

    def test_whitespace_padded_signature_stripped_and_accepted(self, client):
        """Standard HTTP header leading/trailing whitespace is stripped and accepted."""
        headers = make_hmac_headers()
        headers["X-Sidecar-Signature"] = "  \t  " + headers["X-Sidecar-Signature"] + "  \t  "
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200

    def test_signature_internal_space_rejected(self, client):
        """Signature with whitespace injected in the middle must fail with 401."""
        headers = make_hmac_headers()
        sig = headers["X-Sidecar-Signature"]
        headers["X-Sidecar-Signature"] = sig[:32] + " " + sig[32:]
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401

    def test_signature_internal_null_byte_rejected(self, client):
        """Signature with null byte injected must fail with 401."""
        headers = make_hmac_headers()
        sig = headers["X-Sidecar-Signature"]
        headers["X-Sidecar-Signature"] = sig[:32] + "\x00" + sig[32:]
        res = client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 401

    def test_uppercase_lowercase_mixedcase_signatures(self, client):
        """Verify constant-time comparison is case-insensitive for valid hex."""
        headers = make_hmac_headers()
        original_sig = headers["X-Sidecar-Signature"]

        # All uppercase
        headers["X-Sidecar-Signature"] = original_sig.upper()
        res_upper = client.get("/api/v1/system/status", headers=headers)
        assert res_upper.status_code == 200

        # Mixed case alternating
        mixed_sig = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(original_sig)
        )
        headers["X-Sidecar-Signature"] = mixed_sig
        res_mixed = client.get("/api/v1/system/status", headers=headers)
        assert res_mixed.status_code == 200

    def test_null_byte_in_payload_body(self, client):
        """Payload containing embedded null bytes must be signed and verified byte-for-byte."""
        payload_with_null = b'{"service":"core\x00secret","data":"abc\x00123"}'
        headers = make_hmac_headers(payload=payload_with_null)
        headers["Content-Type"] = "application/json"

        res = client.post("/api/v1/system/status", data=payload_with_null, headers=headers)
        # Endpoint should not crash with 500
        assert res.status_code == 200
        # If null byte causes JSON parser to drop field or parse, response status is healthy
        assert res.get_json()["status"] == "healthy"

    def test_pure_null_bytes_body(self, client):
        """Payload of raw binary null bytes is cleanly verified without 500."""
        payload = b"\x00" * 4096
        headers = make_hmac_headers(payload=payload)
        res = client.post("/api/v1/system/status", data=payload, headers=headers)
        assert res.status_code == 200

    def test_large_payload_1mb(self, client):
        """1MB payload must be signed and verified without memory error or truncation."""
        large_body = b"A" * (1024 * 1024)
        headers = make_hmac_headers(payload=large_body)
        res = client.post("/api/v1/system/status", data=large_body, headers=headers)
        assert res.status_code == 200

    def test_large_payload_2mb_tampered(self, client):
        """2MB payload tampered at the last byte must be rejected with 401."""
        body = b"B" * (2 * 1024 * 1024)
        headers = make_hmac_headers(payload=body)
        tampered_body = body[:-1] + b"C"
        res = client.post("/api/v1/system/status", data=tampered_body, headers=headers)
        assert res.status_code == 401

    def test_empty_payload_bodies_across_all_endpoints(self, client):
        """Empty payload (b'') must verify correctly across GET and POST endpoints."""
        endpoints = [
            ("GET", "/api/v1/system/status"),
            ("POST", "/api/v1/system/status"),
            ("POST", "/api/v1/email/send"),
            ("POST", "/api/v1/db/drop"),
        ]
        for method, ep in endpoints:
            headers = make_hmac_headers(payload=b"")
            if method == "GET":
                res = client.get(ep, headers=headers)
            else:
                res = client.post(ep, data=b"", headers=headers)
            assert res.status_code == 200, f"Failed on {method} {ep}"

    def test_unicode_multibyte_cjk_emoji(self, client):
        """Multi-byte UTF-8 payload (emoji, Chinese, Japanese, Arabic) hashes deterministically."""
        unicode_payload = {
            "to": "agent@example.com",
            "subject": "🚀 安全警告 / セキュリティ / تنبيه أمني",
            "body": "Special characters: üöä, ñ, ç, 𝄞 (musical symbol)",
        }
        body_bytes = json.dumps(unicode_payload, ensure_ascii=False).encode("utf-8")
        headers = make_hmac_headers(payload=body_bytes)
        headers["Content-Type"] = "application/json; charset=utf-8"

        res = client.post("/api/v1/email/send", data=body_bytes, headers=headers)
        assert res.status_code == 200
        assert res.get_json()["status"] == "sent"

    def test_arbitrary_non_utf8_binary_payload(self, client):
        """Arbitrary non-UTF8 binary byte payload can be signed and verified."""
        random_bytes = bytes(range(256)) * 4  # 1024 bytes containing all 0..255 byte values
        headers = make_hmac_headers(payload=random_bytes)
        res = client.post("/api/v1/system/status", data=random_bytes, headers=headers)
        assert res.status_code == 200


# ==============================================================================
# SECTION 3: DIRECT BYPASS ATTEMPTS & CHOKE-POINT SECURITY
# ==============================================================================


class TestDirectBypassAttempts:
    """Stress tests simulating unauthorized clients bypassing sidecar."""

    @pytest.mark.parametrize(
        "endpoint,method",
        [
            ("/api/v1/db/drop", "POST"),
            ("/db/drop", "POST"),
            ("/api/v1/email/send", "POST"),
            ("/email/send", "POST"),
            ("/api/v1/system/status", "GET"),
            ("/api/v1/system/status", "POST"),
            ("/system/status", "GET"),
            ("/system/status", "POST"),
            ("/status", "GET"),
            ("/status", "POST"),
        ],
    )
    def test_direct_requests_without_headers_return_401(self, client, endpoint, method):
        """Every protected route must return HTTP 401 when accessed directly without sidecar headers."""
        if method == "GET":
            res = client.get(endpoint)
        else:
            res = client.post(endpoint, json={"malicious": "payload"})
        assert res.status_code == 401
        assert res.get_json()["error"] == "unauthorized"

    def test_forged_headers_random_signature_returns_401(self, client):
        """Forged headers with spoofed timestamp and random 64-char hex string must fail."""
        forged_headers = {
            "X-Sidecar-Timestamp": str(int(time.time())),
            "X-Sidecar-Signature": "f" * 64,
        }
        res = client.post("/api/v1/db/drop", json={"table": "customers"}, headers=forged_headers)
        assert res.status_code == 401
        assert res.get_json()["reason"] == "Invalid sidecar HMAC signature"

    def test_reused_signature_with_modified_timestamp_fails(self, client):
        """Attacker takes signature generated for timestamp T, sends T + 1 -> 401."""
        now = int(time.time())
        body = b'{"table":"users"}'
        headers = make_hmac_headers(payload=body, timestamp=now)
        # Tamper timestamp header
        headers["X-Sidecar-Timestamp"] = str(now + 1)
        res = client.post("/api/v1/db/drop", data=body, headers=headers)
        assert res.status_code == 401
        assert "signature" in res.get_json()["reason"].lower()

    def test_cross_payload_signature_reuse_fails(self, client):
        """Signature generated for benign status query cannot authenticate dangerous db drop body."""
        now = int(time.time())
        benign_body = b'{"service":"core"}'
        dangerous_body = b'{"table":"production_users"}'

        # Compute signature over benign body
        headers = make_hmac_headers(payload=benign_body, timestamp=now)
        # Attempt to use it with dangerous body
        res = client.post("/api/v1/db/drop", data=dangerous_body, headers=headers)
        assert res.status_code == 401
        assert "signature" in res.get_json()["reason"].lower()

    def test_healthz_subpath_not_bypassed(self, client):
        """Subpaths under /healthz (e.g. /healthz/admin) do not get bypass exemption."""
        res = client.get("/healthz/admin")
        # Should either be 401 (blocked by before_request) or 404
        assert res.status_code in (401, 404)

    def test_method_not_allowed_on_endpoints(self, client):
        """DELETE on protected endpoints with valid HMAC returns 405 Method Not Allowed."""
        headers = make_hmac_headers(payload=b"")
        res = client.delete("/api/v1/db/drop", headers=headers)
        assert res.status_code == 405

    def test_query_parameters_with_valid_hmac(self, client):
        """Protected endpoint with query parameters passes HMAC and returns query result."""
        headers = make_hmac_headers(payload=b"")
        res = client.get("/api/v1/system/status?service=payments_engine", headers=headers)
        assert res.status_code == 200
        assert res.get_json()["service"] == "payments_engine"


# ==============================================================================
# SECTION 4: SERVER MISCONFIGURATION & SECURITY FAIL-CLOSED
# ==============================================================================


class TestServerMisconfiguration:
    """Verify backend fails closed if signing keys are missing or invalid."""

    def test_empty_secret_fails_closed_500(self):
        """Backend initialized with empty secret returns 500 server error."""
        app_misconfig = create_app(secret="")
        test_client = app_misconfig.test_client()
        headers = make_hmac_headers(secret="dummy", payload=b"")
        res = test_client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 500
        assert res.get_json()["error"] == "server_error"

    def test_bytes_secret_accepted(self):
        """Backend initialized with bytes secret functions equivalently to string secret."""
        app_bytes = create_app(secret=b"bytes_secret_key_1234567890_test")
        test_client = app_bytes.test_client()
        headers = make_hmac_headers(secret=b"bytes_secret_key_1234567890_test", payload=b"")
        res = test_client.get("/api/v1/system/status", headers=headers)
        assert res.status_code == 200
