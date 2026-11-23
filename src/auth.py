import flask
import hashlib
import hmac
import os
import sqlite3
import cookies
import db
import tokens

from datetime import datetime, timedelta
from functools import wraps


PASSWORD_HASH_ITERATIONS = 200_000
VALID_ROLES = ("admin", "engineer", "user", "viewer")

ROLE_PERMISSIONS = {
    "admin": {
        "cameras:view",
        "dashboard:configure",
        "dashboard:view",
        "dashboard_settings:view",
        "models:manage",
        "models:view",
        "pipeline:snapshot",
        "pipeline:start",
        "pipeline:stop",
        "roboflow:manage",
        "results:annotate",
        "results:delete",
        "results:download",
        "results:inspect",
        "results:view",
        "results:view_all",
        "smtp:manage",
        "status:view",
        "swagger:view",
        "upload:video",
        "users:manage",
    },
    "engineer": {
        "cameras:view",
        "dashboard:configure",
        "dashboard:view",
        "dashboard_settings:view",
        "models:manage",
        "models:view",
        "pipeline:snapshot",
        "pipeline:start",
        "pipeline:stop",
        "roboflow:manage",
        "results:annotate",
        "results:download",
        "results:inspect",
        "results:view",
        "results:view_all",
        "status:view",
        "swagger:view",
        "upload:video",
    },
    "user": {
        "dashboard:view",
        "dashboard_settings:view",
        "pipeline:snapshot",
        "pipeline:start",
        "pipeline:stop",
        "results:annotate",
        "results:download",
        "results:inspect",
        "results:view",
        "status:view",
    },
    "viewer": {
        "dashboard:view",
        "dashboard_settings:view",
        "results:download",
        "results:inspect",
        "results:view",
        "results:view_all",
        "status:view",
    },
}


def _hash_password(password, *, salt_hex=None):
    salt = bytes.fromhex(salt_hex) if salt_hex else os.urandom(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )
    return f"pbkdf2_sha256${PASSWORD_HASH_ITERATIONS}${salt.hex()}${digest.hex()}"


def _get_roles(connection, user_id):
    rows = connection.execute("SELECT role FROM user_roles WHERE user_id = ?", (user_id,)).fetchall()
    return [row["role"] for row in rows]


def _request_wants_html():
    if flask.request.path in ("/", "/results", "/models", "/users", "/settings", "/login"):
        return True
    accept = flask.request.headers.get("Accept", "")
    return "text/html" in accept


def _user_from_row(row, roles):
    if row is None:
        return None
    role_list = sorted(set(roles))
    return {
        "id": row["id"],
        "username": row["username"],
        "active": bool(row["active"]),
        "force_password_change": bool(row["force_password_change"]),
        "roles": role_list,
        "permissions": get_permissions_for_roles(role_list),
    }


def _active_or_none(user):
    if user is None or not user["active"]:
        return None
    return user


def authenticate_user(username, password):
    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT id, username, password_hash, active, force_password_change FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        if row is None or not row["active"]:
            return None
        if not verify_password(password, row["password_hash"]):
            return None
        return _user_from_row(row, _get_roles(connection, row["id"]))
    finally:
        connection.close()


def _clean_roles(roles):
    clean_roles = sorted(set((role or "").strip().lower() for role in (roles or [])))
    if not clean_roles:
        raise ValueError("At least one role is required.")
    if any(role not in VALID_ROLES for role in clean_roles):
        raise ValueError("Invalid roles.")
    return clean_roles


def _revoke_user_refresh_tokens(connection, user_id):
    connection.execute(
        "UPDATE refresh_tokens SET revoked_at = ? WHERE user_id = ? AND revoked_at IS NULL",
        (db.utcnow_text(), user_id),
    )


def create_user(username, password, roles):
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")
    if not password:
        raise ValueError("Password is required.")

    clean_roles = _clean_roles(roles)

    connection = db.connect()
    try:
        cursor = connection.execute(
            """
            INSERT INTO users (username, password_hash, active, force_password_change, created_at)
            VALUES (?, ?, 1, 1, ?)
            """,
            (username, _hash_password(password), db.utcnow_text()),
        )
        for role in clean_roles:
            connection.execute(
                "INSERT INTO user_roles (user_id, role) VALUES (?, ?)",
                (cursor.lastrowid, role),
            )
        connection.commit()
        return get_user_by_id(cursor.lastrowid)
    except sqlite3.IntegrityError as exc:
        connection.rollback()
        raise ValueError("Username already exists.") from exc
    finally:
        connection.close()


def forbidden_response():
    if _request_wants_html():
        return flask.make_response("Forbidden", 403)
    return flask.jsonify({"error": "Forbidden"}), 403


def get_permissions_for_roles(roles):
    permissions = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, set()))
    return permissions


def get_user_by_id(user_id):
    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT id, username, password_hash, active, force_password_change FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return _user_from_row(row, _get_roles(connection, row["id"])) if row else None
    finally:
        connection.close()


def get_user_by_username(username):
    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT id, username, password_hash, active, force_password_change FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return _user_from_row(row, _get_roles(connection, row["id"])) if row else None
    finally:
        connection.close()


def list_users():
    connection = db.connect()
    try:
        rows = connection.execute(
            """
            SELECT id, username, active, force_password_change
            FROM users
            ORDER BY username
            """
        ).fetchall()
        users = []
        for row in rows:
            users.append(
                {
                    "id": row["id"],
                    "username": row["username"],
                    "active": bool(row["active"]),
                    "force_password_change": bool(row["force_password_change"]),
                    "roles": sorted(set(_get_roles(connection, row["id"]))),
                }
            )
        return users
    finally:
        connection.close()


def init_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            active INTEGER NOT NULL DEFAULT 1,
            force_password_change INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS user_roles (
            user_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            UNIQUE(user_id, role)
        );

        CREATE TABLE IF NOT EXISTS refresh_tokens (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            token_hash TEXT NOT NULL UNIQUE,
            expires_at TEXT NOT NULL,
            revoked_at TEXT,
            created_at TEXT NOT NULL
        );
        """
    )

    if not db.has_column(connection, "users", "force_password_change"):
        connection.execute(
            "ALTER TABLE users ADD COLUMN force_password_change INTEGER NOT NULL DEFAULT 0"
        )

    admin_row = connection.execute(
        "SELECT id, password_hash FROM users WHERE username = ?",
        ("admin",),
    ).fetchone()
    if admin_row is None:
        cursor = connection.execute(
            """
            INSERT INTO users (username, password_hash, active, force_password_change, created_at)
            VALUES (?, ?, 1, 1, ?)
            """,
            ("admin", _hash_password("admin"), db.utcnow_text()),
        )
        connection.execute(
            "INSERT INTO user_roles (user_id, role) VALUES (?, ?)",
            (cursor.lastrowid, "admin"),
        )
    elif verify_password("admin", admin_row["password_hash"]):
        connection.execute(
            "UPDATE users SET force_password_change = 1 WHERE id = ?",
            (admin_row["id"],),
        )


def can_view_all_results(user):
    return user_has_permission(user, "results:view_all")


def issue_refresh_token(user_id):
    raw_token = tokens.generate_refresh_token()
    connection = db.connect()
    try:
        connection.execute(
            """
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at, revoked_at, created_at)
            VALUES (?, ?, ?, NULL, ?)
            """,
            (
                user_id,
                tokens.hash_refresh_token(raw_token),
                (db.utcnow() + timedelta(seconds=tokens.REFRESH_TOKEN_TTL_SECONDS)).isoformat() + "Z",
                db.utcnow_text(),
            ),
        )
        connection.commit()
        return raw_token
    finally:
        connection.close()


def authenticate_refresh_token(raw_token):
    if not raw_token:
        return None

    token_hash = tokens.hash_refresh_token(raw_token)
    connection = db.connect()
    try:
        row = connection.execute(
            """
            SELECT user_id, expires_at, revoked_at
            FROM refresh_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None or row["revoked_at"]:
            return None

        expires_at = row["expires_at"] or ""
        if expires_at.endswith("Z"):
            expires_at = expires_at[:-1]
        if datetime.fromisoformat(expires_at) <= db.utcnow():
            return None

        return _active_or_none(get_user_by_id(row["user_id"]))
    finally:
        connection.close()


def change_password(username, current_password, new_password):
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")
    if not current_password:
        raise ValueError("Current password is required.")
    if not new_password:
        raise ValueError("New password is required.")
    if new_password == current_password:
        raise ValueError("New password must differ from the current password.")

    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT id, password_hash FROM users WHERE username = ? AND active = 1",
            (username,),
        ).fetchone()
        if row is None or not verify_password(current_password, row["password_hash"]):
            return None
        connection.execute(
            """
            UPDATE users
            SET password_hash = ?, force_password_change = 0
            WHERE id = ?
            """,
            (_hash_password(new_password), row["id"]),
        )
        connection.commit()
        return get_user_by_id(row["id"])
    finally:
        connection.close()


def load_request_user():
    cached = getattr(flask.g, "_current_user", None)
    if cached is not None:
        return cached

    auth_header_token = tokens.get_authorization_access_token(flask.request)
    if auth_header_token:
        user_id = tokens.verify_access_token(auth_header_token)
        if not user_id:
            return None
        user = _active_or_none(get_user_by_id(user_id))
        if user is not None:
            flask.g._current_user = user
        return user

    cookie_access_token = tokens.get_cookie_access_token(flask.request)
    if cookie_access_token:
        user_id = tokens.verify_access_token(cookie_access_token)
        if user_id:
            user = _active_or_none(get_user_by_id(user_id))
            if user is not None:
                flask.g._current_user = user
            return user

    refresh_token = cookies.get_refresh_token_from_request(flask.request)
    if not refresh_token:
        return None

    user = authenticate_refresh_token(refresh_token)
    if user is None:
        return None

    flask.g._current_user = user
    flask.g._auth_cookie_refresh = {
        "access_token": tokens.issue_access_token(user["id"]),
        "refresh_token": refresh_token,
    }
    return user


def require_permission(permission=None, *, html_redirect=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            user = load_request_user()
            if user is None:
                return unauthorized_response(html_redirect=html_redirect)
            flask.g.current_user = user
            if permission and not user_has_permission(user, permission):
                return forbidden_response()
            return func(*args, **kwargs)

        return wrapper

    return decorator


def reset_user_password(user_id, new_password):
    if not new_password:
        raise ValueError("Password is required.")

    connection = db.connect()
    try:
        cursor = connection.execute(
            "UPDATE users SET password_hash = ?, force_password_change = 1 WHERE id = ?",
            (_hash_password(new_password), user_id),
        )
        if cursor.rowcount == 0:
            return None
        _revoke_user_refresh_tokens(connection, user_id)
        connection.commit()
        return get_user_by_id(user_id)
    finally:
        connection.close()


def revoke_refresh_token(raw_token):
    if not raw_token:
        return
    connection = db.connect()
    try:
        connection.execute(
            "UPDATE refresh_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (db.utcnow_text(), tokens.hash_refresh_token(raw_token)),
        )
        connection.commit()
    finally:
        connection.close()


def rotate_refresh_token(raw_token):
    token_hash = tokens.hash_refresh_token(raw_token)
    connection = db.connect()
    try:
        row = connection.execute(
            """
            SELECT id, user_id, expires_at, revoked_at
            FROM refresh_tokens
            WHERE token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None or row["revoked_at"]:
            return None

        expires_at = row["expires_at"] or ""
        if expires_at.endswith("Z"):
            expires_at = expires_at[:-1]
        if datetime.fromisoformat(expires_at) <= db.utcnow():
            return None

        connection.execute(
            "UPDATE refresh_tokens SET revoked_at = ? WHERE id = ?",
            (db.utcnow_text(), row["id"]),
        )
        connection.commit()

        user = _active_or_none(get_user_by_id(row["user_id"]))
        if user is None:
            return None
        return {
            "user": user,
            "refresh_token": issue_refresh_token(user["id"]),
        }
    finally:
        connection.close()


def unauthorized_response(*, html_redirect=False):
    if html_redirect and _request_wants_html() and not tokens.get_authorization_access_token(flask.request):
        response = flask.redirect(flask.url_for("accounts.login_page"))
    else:
        response = flask.make_response(flask.jsonify({"error": "Unauthorized"}), 401)

    if not tokens.get_authorization_access_token(flask.request):
        response = cookies.clear_auth_cookies(response)
    return response


def update_user(user_id, *, roles=None, active=None, acting_user_id=None):
    clean_roles = _clean_roles(roles) if roles is not None else None
    if active is not None and not isinstance(active, bool):
        raise ValueError("Active must be a boolean.")
    # The acting admin keeping their own admin role guarantees one active admin remains.
    if user_id == acting_user_id and (active is False or (clean_roles is not None and "admin" not in clean_roles)):
        raise ValueError("You cannot deactivate yourself or remove your own admin role.")

    connection = db.connect()
    try:
        if connection.execute("SELECT id FROM users WHERE id = ?", (user_id,)).fetchone() is None:
            return None
        if clean_roles is not None:
            connection.execute("DELETE FROM user_roles WHERE user_id = ?", (user_id,))
            for role in clean_roles:
                connection.execute(
                    "INSERT INTO user_roles (user_id, role) VALUES (?, ?)",
                    (user_id, role),
                )
        if active is not None:
            connection.execute("UPDATE users SET active = ? WHERE id = ?", (int(active), user_id))
            if not active:
                _revoke_user_refresh_tokens(connection, user_id)
        connection.commit()
        return get_user_by_id(user_id)
    finally:
        connection.close()


def user_has_permission(user, permission):
    if user is None:
        return False
    return permission in user.get("permissions", set())


def verify_password(password, stored_hash):
    try:
        scheme, iterations, salt_hex, digest_hex = stored_hash.split("$", 3)
    except ValueError:
        return False
    if scheme != "pbkdf2_sha256":
        return False
    candidate = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    ).hex()
    return hmac.compare_digest(candidate, digest_hex)
