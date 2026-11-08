"""
Unit and security test suite for the Flask backend microservice (Ticket T8 / Milestone 5).

Verifies:
1. Health check endpoint (/healthz) accessibility without authentication.
2. Two-legged cryptographic HMAC-SHA256 signature enforcement on all protected endpoints.
3. Timestamp freshness and 30-second replay window protection (both past and future skew).
4. Direct request bypass prevention (choke-point bypass returns HTTP 401).
5. Cross-language cryptographic compatibility with Rust sidecar (src/crypto.rs).
6. Exact 30-second boundary conditions and case-insensitive hex comparison.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import time
import pytest

# Ensure backend directory is in sys.path regardless of where pytest is invoked
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

try:
    from backend.app import create_app
except ImportError:
    from app import create_app

# Default test signing secret matching .env.example
TEST_SECRET = "test_sidecar_secret_32_bytes_long_ok"


def generate_hmac_headers(
    secret: str | bytes = TEST_SECRET,
    payload: str | bytes | dict | list | None = None,
    timestamp: int | float | None = None,
) -> dict[str, str]:
    """
    Generate X-Sidecar-Timestamp and X-Sidecar-Signature HMAC headers for testing.

    Algorithm matches Rust sidecar (crypto::sign_payload):
        canonical_payload = f"{timestamp}.".encode("utf-8") + raw_body
        signature = hmac_sha256(secret, canonical_payload).hexdigest()

    Args:
        secret: Secret key as string or bytes (defaults to TEST_SECRET).
        payload: Request payload (bytes, str, dict/list for compact JSON, or None for empty).
        timestamp: Unix epoch timestamp in seconds (defaults to int(time.time())).

    Returns:
        Dictionary containing 'X-Sidecar-Timestamp' and 'X-Sidecar-Signature'.
    """
    if timestamp is None:
        timestamp = int(time.time())

    if isinstance(payload, bytes):
        raw_body = payload
    elif isinstance(payload, str):
        raw_body = payload.encode("utf-8")
    elif isinstance(payload, (dict, list)):
        raw_body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    elif payload is None:
        raw_body = b""
    else:
        raw_body = str(payload).encode("utf-8")

    secret_bytes = secret if isinstance(secret, bytes) else secret.encode("utf-8")
    ts_int = int(timestamp)
    signing_input = f"{ts_int}.".encode("utf-8") + raw_body
    sig = hmac.new(secret_bytes, signing_input, hashlib.sha256).hexdigest()

    return {
        "X-Sidecar-Timestamp": str(ts_int),
        "X-Sidecar-Signature": sig,
    }


@pytest.fixture
def app():
    """Create and configure a clean Flask app instance for testing."""
    app_instance = create_app(secret=TEST_SECRET)
    app_instance.config["TESTING"] = True
    return app_instance


@pytest.fixture
def client(app):
    """Test client for dispatching HTTP requests against the Flask app."""
    return app.test_client()


# ==============================================================================
# Mandatory Milestone 5 Test Cases (Ticket T8)
# ==============================================================================

def test_healthz_unauthenticated(client):
    """Verify /healthz and /health are unauthenticated and return HTTP 200 {"status": "ok"}."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}

    alias_response = client.get("/health")
    assert alias_response.status_code == 200
    assert alias_response.get_json() == {"status": "ok"}


def test_valid_signature_get(client):
    """Verify GET /api/v1/system/status with valid HMAC succeeds (HTTP 200)."""
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=None)
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "core"
    assert data["uptime_seconds"] == 86400
    assert "metrics" in data

    # Verify route alias /status
    alias_response = client.get("/status", headers=headers)
    assert alias_response.status_code == 200
    assert alias_response.get_json()["status"] == "healthy"

    # Verify route alias /system/status
    sys_alias_response = client.get("/system/status", headers=headers)
    assert sys_alias_response.status_code == 200
    assert sys_alias_response.get_json()["status"] == "healthy"


def test_valid_signature_get_with_payload(client):
    """Verify GET request with outbound JSON payload (as forwarded by Axum Route B)."""
    payload = {"service": "worker_pool"}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=payload_bytes)
    headers["Content-Type"] = "application/json"

    response = client.get("/api/v1/system/status", data=payload_bytes, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert data["service"] == "worker_pool"


def test_valid_signature_post(client):
    """Verify POST /api/v1/email/send with valid HMAC and JSON body succeeds (HTTP 200)."""
    payload = {
        "to": "alice@ourcompany.com",
        "subject": "System Report",
        "body": "Daily diagnostics all normal.",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=payload_bytes)
    headers["Content-Type"] = "application/json"

    response = client.post("/api/v1/email/send", data=payload_bytes, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "sent"
    assert data["to"] == "alice@ourcompany.com"

    # Also verify route alias /email/send
    alias_response = client.post("/email/send", data=payload_bytes, headers=headers)
    assert alias_response.status_code == 200
    assert alias_response.get_json()["status"] == "sent"


def test_valid_signature_db_drop(client):
    """Verify POST /api/v1/db/drop with valid HMAC succeeds on backend (HTTP 200)."""
    payload = {"table": "test_table"}
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=payload_bytes)
    headers["Content-Type"] = "application/json"

    response = client.post("/api/v1/db/drop", data=payload_bytes, headers=headers)
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "dropped"
    assert data["table"] == "test_table"

    # Also verify route alias /db/drop
    alias_response = client.post("/db/drop", data=payload_bytes, headers=headers)
    assert alias_response.status_code == 200
    assert alias_response.get_json()["status"] == "dropped"


def test_missing_signature_header(client):
    """Protected request missing X-Sidecar-Signature returns HTTP 401."""
    headers = {"X-Sidecar-Timestamp": str(int(time.time()))}
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "missing" in data["reason"].lower()


def test_missing_timestamp_header(client):
    """Protected request missing X-Sidecar-Timestamp returns HTTP 401."""
    headers = {"X-Sidecar-Signature": "abc123deadbeef"}
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "missing" in data["reason"].lower()


def test_missing_both_headers(client):
    """Protected request missing both X-Sidecar headers returns HTTP 401."""
    response = client.get("/api/v1/system/status")
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "missing" in data["reason"].lower()


def test_invalid_signature_wrong_secret(client):
    """Signature computed using wrong secret key returns HTTP 401."""
    headers = generate_hmac_headers(
        secret="wrong-secret-key-that-does-not-match!",
        payload=None,
    )
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "signature" in data["reason"].lower()


def test_invalid_signature_corrupted_hex(client):
    """Corrupted hex signature string returns HTTP 401."""
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=None)
    headers["X-Sidecar-Signature"] = "0" * 64
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "signature" in data["reason"].lower()


def test_expired_timestamp_past(client):
    """Timestamp 31 seconds in the past returns HTTP 401 (replay window exceeded)."""
    past_ts = int(time.time()) - 31
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=past_ts)
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "skew" in data["reason"].lower() or "expired" in data["reason"].lower()


def test_expired_timestamp_future(client):
    """Timestamp 31 seconds in the future returns HTTP 401 (future clock skew exceeded)."""
    future_ts = int(time.time()) + 31
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=future_ts)
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "skew" in data["reason"].lower() or "expired" in data["reason"].lower()


def test_non_integer_timestamp(client):
    """Non-integer timestamp string in X-Sidecar-Timestamp returns HTTP 401."""
    headers = {
        "X-Sidecar-Timestamp": "not_an_integer",
        "X-Sidecar-Signature": "abc123deadbeef",
    }
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "timestamp" in data["reason"].lower()


def test_tampered_body(client):
    """Signature computed over original body, but body is tampered in transit -> returns HTTP 401."""
    original_payload = {"to": "alice@ourcompany.com", "body": "Original text"}
    tampered_payload = {"to": "attacker@evil.com", "body": "Exfiltrated text"}

    orig_bytes = json.dumps(original_payload, separators=(",", ":")).encode("utf-8")
    tampered_bytes = json.dumps(tampered_payload, separators=(",", ":")).encode("utf-8")

    # Compute valid signature over original bytes, but transmit tampered bytes
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=orig_bytes)
    headers["Content-Type"] = "application/json"

    response = client.post("/api/v1/email/send", data=tampered_bytes, headers=headers)
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "signature" in data["reason"].lower()


def test_direct_request_bypass_blocked(client):
    """Simulates direct attacker call to /api/v1/db/drop without sidecar headers -> returns HTTP 401."""
    attack_payload = {"table": "users"}
    attack_bytes = json.dumps(attack_payload, separators=(",", ":")).encode("utf-8")

    # Attacker calls Flask directly without sidecar proxy (no X-Sidecar-* headers)
    response = client.post(
        "/api/v1/db/drop",
        data=attack_bytes,
        content_type="application/json",
    )
    assert response.status_code == 401
    data = response.get_json()
    assert data["error"] == "unauthorized"
    assert "missing" in data["reason"].lower()

    # Also test shorthand route /db/drop
    alias_response = client.post(
        "/db/drop",
        data=attack_bytes,
        content_type="application/json",
    )
    assert alias_response.status_code == 401
    assert alias_response.get_json()["error"] == "unauthorized"


def test_timestamp_boundary_acceptance(client, monkeypatch):
    """Timestamps at exact 30s boundary (now - 30s and now + 30s) are accepted deterministically."""
    fixed_now = 1731000030
    monkeypatch.setattr(time, "time", lambda: float(fixed_now))

    # Past boundary (exactly 30 seconds ago: 1731000000)
    headers_past = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=fixed_now - 30)
    response_past = client.get("/api/v1/system/status", headers=headers_past)
    assert response_past.status_code == 200

    # Future boundary (exactly 30 seconds ahead: 1731000060)
    headers_future = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=fixed_now + 30)
    response_future = client.get("/api/v1/system/status", headers=headers_future)
    assert response_future.status_code == 200

    # Boundary violation: 31 seconds in the past
    headers_past_exceeded = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=fixed_now - 31)
    response_past_exceeded = client.get("/api/v1/system/status", headers=headers_past_exceeded)
    assert response_past_exceeded.status_code == 401
    assert "skew" in response_past_exceeded.get_json()["reason"].lower()

    # Boundary violation: 31 seconds in the future
    headers_future_exceeded = generate_hmac_headers(secret=TEST_SECRET, payload=None, timestamp=fixed_now + 31)
    response_future_exceeded = client.get("/api/v1/system/status", headers=headers_future_exceeded)
    assert response_future_exceeded.status_code == 401
    assert "skew" in response_future_exceeded.get_json()["reason"].lower()


def test_cross_language_vector_compatibility():
    """
    Verify exact cryptographic parity with Rust sidecar test vector:
    src/crypto.rs test_hmac_cross_language_test_vector:
      secret = b"sidecar_secret_key_12345"
      ts = 1731000000
      body = b'{"service":"core"}'
      expected = "79c61f56884861a154d771302e545af899fb00f60568fe74ff228a8018955bda"
    """
    secret = "sidecar_secret_key_12345"
    ts = 1731000000
    body = b'{"service":"core"}'
    expected_sig = "79c61f56884861a154d771302e545af899fb00f60568fe74ff228a8018955bda"

    headers = generate_hmac_headers(secret=secret, payload=body, timestamp=ts)
    assert headers["X-Sidecar-Signature"] == expected_sig


def test_server_misconfigured_no_secret():
    """When backend SIDECAR_SECRET is unset or empty, protected endpoints return HTTP 500."""
    unconfigured_app = create_app(secret="")
    unconfigured_client = unconfigured_app.test_client()
    headers = generate_hmac_headers(secret="any", payload=None)
    response = unconfigured_client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 500
    data = response.get_json()
    assert (
        "server misconfigured" in data.get("reason", "").lower()
        or "server_error" in data.get("error", "").lower()
    )


def test_case_insensitive_signature_comparison(client):
    """Verify uppercase or mixed-case hex signatures are accepted via case-insensitive comparison."""
    headers = generate_hmac_headers(secret=TEST_SECRET, payload=None)
    headers["X-Sidecar-Signature"] = headers["X-Sidecar-Signature"].upper()
    response = client.get("/api/v1/system/status", headers=headers)
    assert response.status_code == 200
