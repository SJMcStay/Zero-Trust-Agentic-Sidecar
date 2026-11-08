#!/usr/bin/env python3
"""Mint an HS256 agent JWT for Route B (POST /api/execute). Stdlib only.

  python3 scripts/mint_jwt.py                 # sub=agent-007 role=agent, JWT_SECRET from env
  python3 scripts/mint_jwt.py agent-1 admin   # other caller / role (roles: see policy.toml)
"""
import base64
import hashlib
import hmac
import json
import os
import sys
import time

secret = os.environ.get("JWT_SECRET") or sys.exit("JWT_SECRET not set (source .env first)")
sub = sys.argv[1] if len(sys.argv) > 1 else "agent-007"
role = sys.argv[2] if len(sys.argv) > 2 else "agent"


def b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


head = b64(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
body = b64(json.dumps({"sub": sub, "role": role, "exp": int(time.time()) + 3600}).encode())
sig = b64(hmac.new(secret.encode(), f"{head}.{body}".encode(), hashlib.sha256).digest())
print(f"{head}.{body}.{sig}")
