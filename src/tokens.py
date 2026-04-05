import hashlib
import os
import secrets
import threading

from itsdangerous import BadSignature
from itsdangerous import SignatureExpired
from itsdangerous import URLSafeTimedSerializer

import cookies


ACCESS_TOKEN_TTL_SECONDS = 15 * 60
REFRESH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
TOKEN_SECRET_PATH = "/app/runs/tilletia.token"

_serializer = None
_serializer_lock = threading.Lock()


def _get_serializer():
    global _serializer
    with _serializer_lock:
        if _serializer is None:
            os.makedirs(os.path.dirname(TOKEN_SECRET_PATH), exist_ok=True)
            if os.path.exists(TOKEN_SECRET_PATH):
                with open(TOKEN_SECRET_PATH, "r", encoding="utf-8") as token_input:
                    secret = token_input.read().strip()
            else:
                secret = secrets.token_urlsafe(48)
                with open(TOKEN_SECRET_PATH, "w", encoding="utf-8") as token_output:
                    token_output.write(secret)
            _serializer = URLSafeTimedSerializer(secret, salt="tilletia-access")
    return _serializer


def generate_refresh_token():
    return secrets.token_urlsafe(32)


def get_request_access_token(request):
    auth_header = (request.headers.get("Authorization") or "").strip()
    if auth_header.lower().startswith("bearer "):
        return auth_header[7:].strip()
    return (request.cookies.get(cookies.ACCESS_COOKIE_NAME) or "").strip() or None


def hash_refresh_token(token):
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def issue_access_token(user_id):
    return _get_serializer().dumps({"sub": user_id, "type": "access"})


def verify_access_token(token):
    try:
        payload = _get_serializer().loads(token, max_age=ACCESS_TOKEN_TTL_SECONDS)
    except (BadSignature, SignatureExpired):
        return None
    if payload.get("type") != "access":
        return None
    return payload.get("sub")
