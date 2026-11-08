"""
Target Python Flask Backend Microservice (:5000)
Zero-Trust Agentic Sidecar Capability Showcase (Ticket T8 / Milestone 5)

Enforces two-legged cryptographic HMAC-SHA256 signature verification and
timestamp replay protection on all protected incoming requests from the Rust
sidecar gateway. Direct requests bypassing the sidecar fail with HTTP 401.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from typing import Any

from flask import Flask, jsonify, request

DEFAULT_FALLBACK_SECRET = "test_sidecar_secret_32_bytes_long_ok"


def create_app(
    secret: str | bytes | None = None,
    max_skew_seconds: float = 30.0,
) -> Flask:
    """
    Factory creating the target Flask application.
    Enforces HMAC-SHA256 signature verification and timestamp freshness on all protected routes.
    """
    app = Flask(__name__)

    # 1. Secret Configuration
    if secret is not None:
        raw_secret = secret if isinstance(secret, bytes) else secret.encode("utf-8")
    else:
        env_secret = os.environ.get("SIDECAR_SECRET", DEFAULT_FALLBACK_SECRET)
        raw_secret = env_secret.encode("utf-8")

    app.config["SIDECAR_SECRET"] = raw_secret
    app.config["MAX_SKEW_SECONDS"] = float(
        os.environ.get("MAX_SKEW_SECONDS", str(max_skew_seconds))
    )

    # 2. Cryptographic Security Middleware
    @app.before_request
    def verify_sidecar_hmac() -> Any:
        # Exempt health check endpoints from authentication
        if request.path in ("/healthz", "/health"):
            return None

        # Fail-closed if backend signing secret is unconfigured or empty
        secret_bytes = app.config.get("SIDECAR_SECRET", b"")
        if not secret_bytes:
            return (
                jsonify({
                    "error": "server_error",
                    "reason": "Server misconfigured: missing SIDECAR_SECRET",
                }),
                500,
            )

        ts_header = request.headers.get("X-Sidecar-Timestamp")
        sig_header = request.headers.get("X-Sidecar-Signature")

        # Missing or empty authentication headers
        if not ts_header or not sig_header or not ts_header.strip() or not sig_header.strip():
            return (
                jsonify({
                    "error": "unauthorized",
                    "reason": "Missing sidecar authentication headers",
                }),
                401,
            )

        # Parse and validate timestamp
        try:
            ts_val = float(ts_header.strip())
            if math.isnan(ts_val) or math.isinf(ts_val) or ts_val < 0:
                raise ValueError("Invalid timestamp value")
            ts_int = int(ts_val)
        except (ValueError, TypeError, OverflowError):
            return (
                jsonify({
                    "error": "unauthorized",
                    "reason": "Invalid timestamp format",
                }),
                401,
            )

        # Replay window / clock skew verification (integer seconds, matching Rust crypto)
        now = int(time.time())
        if abs(now - ts_int) > app.config["MAX_SKEW_SECONDS"]:
            return (
                jsonify({
                    "error": "unauthorized",
                    "reason": "Timestamp skew exceeded (max 30s)",
                }),
                401,
            )

        # Construct canonical payload: f"{timestamp}." + raw_body
        raw_body = request.get_data()
        canonical_payload = f"{ts_int}.".encode("utf-8") + raw_body
        expected_sig = hmac.new(
            secret_bytes,
            canonical_payload,
            hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison resistant to timing attacks
        if not hmac.compare_digest(sig_header.strip().lower(), expected_sig.lower()):
            return (
                jsonify({
                    "error": "unauthorized",
                    "reason": "Invalid sidecar HMAC signature",
                }),
                401,
            )

        return None

    # 3. Route Handlers & URL Aliases

    @app.route("/healthz", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def healthz():
        """Unauthenticated liveness probe."""
        return jsonify({"status": "ok"}), 200

    @app.route("/api/v1/system/status", methods=["GET", "POST"])
    @app.route("/system/status", methods=["GET", "POST"])
    @app.route("/status", methods=["GET", "POST"])
    def system_status():
        """Protected system status endpoint."""
        data = request.get_json(silent=True) or {}
        service = data.get("service") or request.args.get("service", "core")
        return (
            jsonify({
                "status": "healthy",
                "service": service,
                "uptime_seconds": 86400,
                "metrics": {
                    "cpu_utilization": 0.12,
                    "memory_mb": 256,
                    "active_connections": 4,
                },
            }),
            200,
        )

    @app.route("/api/v1/email/send", methods=["POST"])
    @app.route("/email/send", methods=["POST"])
    def send_email():
        """Protected email dispatch endpoint."""
        data = request.get_json(silent=True) or {}
        return (
            jsonify({
                "status": "sent",
                "to": data.get("to"),
                "subject": data.get("subject", ""),
                "message_id": f"msg-{int(time.time())}",
            }),
            200,
        )

    @app.route("/api/v1/db/drop", methods=["POST"])
    @app.route("/db/drop", methods=["POST"])
    def drop_db():
        """Protected database drop endpoint (blocked by default in policy.toml)."""
        data = request.get_json(silent=True) or {}
        return (
            jsonify({
                "status": "dropped",
                "table": data.get("table", "unknown"),
            }),
            200,
        )

    return app


# Default app instance for WSGI servers / direct module imports
app = create_app()

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    print(f"Flask target backend listening on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
