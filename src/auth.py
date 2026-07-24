import flask
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
import stat
import tempfile
import threading
import cookies
import tokens

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from datetime import datetime, timedelta
from functools import wraps


DB_PATH = "/app/runs/tilletia.sqlite3"
ROBOFLOW_SETTINGS_PATH = "/app/runs/roboflow.json"
ROBOFLOW_MASTER_KEY_PATH = "/run/secrets/roboflow-master-key"
ROBOFLOW_KEY_VERSION = 1
ROBOFLOW_TOKEN_NONCE_BYTES = 12
SMTP_KEY_VERSION = 1
SMTP_TLS_MODES = ("starttls", "ssl", "none")
DEFAULT_SMTP_PORT = 587
PASSWORD_HASH_ITERATIONS = 200_000
VALID_ROLES = ("admin", "user")

ROLE_PERMISSIONS = {
    "admin": {
        "cameras:view",
        "dashboard:configure",
        "dashboard:view",
        "dashboard_settings:view",
        "models:manage",
        "models:view",
        "pipeline:start",
        "pipeline:stop",
        "roboflow:manage",
        "results:delete",
        "results:download",
        "results:inspect",
        "results:view",
        "smtp:manage",
        "status:view",
        "swagger:view",
        "upload:video",
        "users:manage",
    },
    "user": {
        "dashboard:view",
        "dashboard_settings:view",
        "pipeline:start",
        "pipeline:stop",
        "results:download",
        "results:inspect",
        "results:view",
        "status:view",
    },
}

DEFAULT_DASHBOARD_SETTINGS = {
    "analysis_number": "",
    "source_type": "camera",
    "camera_device": "",
    "camera_mode": {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "format": "MJPG",
    },
    "uploaded_path": "",
    "model_path": "",
    "vis_conf": 0.75,
    "captions_enabled": True,
    "grid_count_enabled": False,
    "grid_debug_enabled": False,
    "grid_score_threshold": 0.30,
    "ask_manual_spore_count": True,
}

_init_lock = threading.Lock()
_roboflow_settings_lock = threading.Lock()
_roboflow_memory_api_token = ""
_smtp_settings_lock = threading.Lock()
_smtp_memory_password = ""
LOGGER = logging.getLogger(__name__)


class RoboflowStorageError(RuntimeError):
    pass


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


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


def _has_column(connection, table_name, column_name):
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


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


def _utcnow():
    return datetime.utcnow()


def _utcnow_text():
    return _utcnow().isoformat() + "Z"


def _get_result_owner_from_legacy_metadata(hq_output_dir, run_id):
    metadata_path = os.path.join(hq_output_dir, run_id, "metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_input:
            metadata = json.load(metadata_input)
    except Exception:
        return None
    owner_username = metadata.get("owner")
    if not owner_username:
        return None
    owner = get_user_by_username(owner_username)
    return owner["id"] if owner else None


def authenticate_user(username, password):
    connection = _connect()
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


def build_page_context(user, **extra_context):
    settings = get_dashboard_settings()
    context = {
        "current_user": user,
        "dashboard_settings": settings,
        "dashboard_settings_json": json.dumps(settings),
        "can_configure_dashboard": user_has_permission(user, "dashboard:configure"),
        "can_delete_results": user_has_permission(user, "results:delete"),
        "can_view_models": user_has_permission(user, "models:view"),
        "can_manage_users": user_has_permission(user, "users:manage"),
        "can_manage_roboflow": user_has_permission(user, "roboflow:manage"),
        "can_upload_roboflow": (
            roboflow_configured() and user_has_permission(user, "roboflow:manage")
        ),
        "can_manage_smtp": user_has_permission(user, "smtp:manage"),
        "can_email_results": smtp_configured() and user_has_permission(user, "results:inspect"),
        "can_select_result_images": (
            user_has_permission(user, "roboflow:manage")
            or user_has_permission(user, "results:inspect")
        ),
        "can_view_result_owners": is_admin(user),
        "can_view_model_owners": is_admin(user),
    }
    context.update(extra_context)
    return context


def create_user(username, password, roles):
    username = (username or "").strip()
    if not username:
        raise ValueError("Username is required.")
    if not password:
        raise ValueError("Password is required.")

    clean_roles = sorted(set((role or "").strip().lower() for role in (roles or [])))
    if not clean_roles:
        raise ValueError("At least one role is required.")

    invalid_roles = [role for role in clean_roles if role not in VALID_ROLES]
    if invalid_roles:
        raise ValueError("Invalid roles.")

    connection = _connect()
    try:
        cursor = connection.execute(
            """
            INSERT INTO users (username, password_hash, active, force_password_change, created_at)
            VALUES (?, ?, 1, 1, ?)
            """,
            (username, _hash_password(password), _utcnow_text()),
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


def delete_result_owner(run_id):
    connection = _connect()
    try:
        connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        connection.commit()
    finally:
        connection.close()


def filter_results_for_user(user, hq_output_dir, results):
    if is_admin(user):
        return results
    return [item for item in results if user_can_access_result(user, hq_output_dir, item["id"])]


def forbidden_response():
    if _request_wants_html():
        return flask.make_response("Forbidden", 403)
    return flask.jsonify({"error": "Forbidden"}), 403


def get_dashboard_settings():
    connection = _connect()
    try:
        row = connection.execute("SELECT * FROM dashboard_settings WHERE id = 1").fetchone()
        if row is None:
            return DEFAULT_DASHBOARD_SETTINGS.copy()
        return {
            "analysis_number": row["analysis_number"],
            "source_type": row["source_type"],
            "camera_device": row["camera_device"],
            "camera_mode": {
                "width": row["camera_width"],
                "height": row["camera_height"],
                "fps": row["camera_fps"],
                "format": row["camera_format"],
            },
            "uploaded_path": row["uploaded_path"],
            "model_path": row["model_path"],
            "vis_conf": float(row["vis_conf"]),
            "captions_enabled": bool(row["captions_enabled"]),
            "grid_count_enabled": bool(row["grid_count_enabled"]),
            "grid_debug_enabled": bool(row["grid_debug_enabled"]),
            "grid_score_threshold": float(row["grid_score_threshold"]),
            "ask_manual_spore_count": bool(row["ask_manual_spore_count"]),
        }
    finally:
        connection.close()


def get_permissions_for_roles(roles):
    permissions = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, set()))
    return permissions


def _validate_roboflow_project_name(value):
    if not isinstance(value, str):
        raise ValueError("Roboflow project name must be a string.")
    project_name = value.strip()
    if not project_name:
        raise ValueError("Roboflow project name is required.")
    if len(project_name) > 255 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project_name) is None:
        raise ValueError("Use a valid Roboflow project ID, without the workspace or URL.")
    return project_name


def _load_roboflow_master_key():
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        file_descriptor = os.open(ROBOFLOW_MASTER_KEY_PATH, flags)
    except OSError:
        raise RoboflowStorageError("Secure Roboflow token storage is unavailable.") from None

    try:
        key_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(key_stat.st_mode)
            or stat.S_IMODE(key_stat.st_mode) & 0o077
        ):
            raise RoboflowStorageError("Secure Roboflow token storage is unavailable.")
        with os.fdopen(file_descriptor, "rb") as key_input:
            file_descriptor = None
            master_key = key_input.read(33)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    if len(master_key) != 32:
        raise RoboflowStorageError("Secure Roboflow token storage is unavailable.")
    return master_key


def _probe_roboflow_secure_storage():
    if not os.path.exists(ROBOFLOW_MASTER_KEY_PATH):
        return "unavailable", None
    try:
        master_key = _load_roboflow_master_key()
        nonce = os.urandom(ROBOFLOW_TOKEN_NONCE_BYTES)
        probe_aad = b"tilletia-app:roboflow-storage-probe:v1"
        encrypted_probe = AESGCM(master_key).encrypt(nonce, b"probe", probe_aad)
        if AESGCM(master_key).decrypt(nonce, encrypted_probe, probe_aad) != b"probe":
            raise RoboflowStorageError("Secure Roboflow token storage is unavailable.")
        return "available", master_key
    except (InvalidTag, OSError, ValueError, RoboflowStorageError):
        return "error", None


def _roboflow_token_aad(project_name):
    return b"tilletia-app:roboflow-settings:v1|" + project_name.encode("utf-8")


def _encrypt_roboflow_token(api_token, project_name, master_key):
    nonce = os.urandom(ROBOFLOW_TOKEN_NONCE_BYTES)
    ciphertext = AESGCM(master_key).encrypt(
        nonce,
        api_token.encode("utf-8"),
        _roboflow_token_aad(project_name),
    )
    return nonce, ciphertext


def _decrypt_roboflow_token(row, master_key):
    nonce = row["api_token_nonce"]
    ciphertext = row["api_token_ciphertext"]
    if nonce is None and ciphertext is None:
        return ""
    if nonce is None or ciphertext is None:
        raise RoboflowStorageError("The encrypted Roboflow token is invalid.")

    nonce = bytes(nonce)
    ciphertext = bytes(ciphertext)
    if (
        row["key_version"] != ROBOFLOW_KEY_VERSION
        or len(nonce) != ROBOFLOW_TOKEN_NONCE_BYTES
        or len(ciphertext) < 16
    ):
        raise RoboflowStorageError("The encrypted Roboflow token is invalid.")

    try:
        plaintext = AESGCM(master_key).decrypt(
            nonce,
            ciphertext,
            _roboflow_token_aad(row["project_name"]),
        )
        return plaintext.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError, ValueError):
        raise RoboflowStorageError("The encrypted Roboflow token could not be decrypted.") from None


def _read_plaintext_roboflow_settings():
    try:
        settings_stat = os.lstat(ROBOFLOW_SETTINGS_PATH)
    except FileNotFoundError:
        return None
    except OSError:
        raise RoboflowStorageError("The plaintext Roboflow token file could not be read.") from None

    if not stat.S_ISREG(settings_stat.st_mode):
        raise RoboflowStorageError("The plaintext Roboflow token path is invalid.")

    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    file_descriptor = None
    try:
        file_descriptor = os.open(ROBOFLOW_SETTINGS_PATH, flags)
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError("Not a regular file")
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "r", encoding="utf-8") as settings_input:
            file_descriptor = None
            serialized_settings = settings_input.read(65_537)
    except (OSError, UnicodeDecodeError):
        raise RoboflowStorageError("The plaintext Roboflow token file could not be read.") from None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    if len(serialized_settings) > 65_536:
        raise RoboflowStorageError("The plaintext Roboflow token file is invalid.")
    try:
        settings = json.loads(serialized_settings)
    except ValueError:
        raise RoboflowStorageError("The plaintext Roboflow token file is invalid.") from None
    if not isinstance(settings, dict):
        raise RoboflowStorageError("The plaintext Roboflow token file is invalid.")

    project_name = settings.get("project_name", "")
    api_token = settings.get("api_token", "")
    if not isinstance(project_name, str) or not isinstance(api_token, str):
        raise RoboflowStorageError("The plaintext Roboflow token file is invalid.")
    return {
        "project_name": project_name.strip(),
        "api_token": api_token.strip(),
        "plaintext_storage_confirmed": settings.get("plaintext_storage_confirmed") is True,
    }


def _write_plaintext_roboflow_settings(project_name, api_token):
    settings_dir = os.path.dirname(ROBOFLOW_SETTINGS_PATH)
    os.makedirs(settings_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=".roboflow-", dir=settings_dir)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as settings_output:
            file_descriptor = None
            json.dump(
                {
                    "schema_version": 1,
                    "project_name": project_name,
                    "api_token": api_token,
                    "plaintext_storage_confirmed": True,
                },
                settings_output,
            )
            settings_output.flush()
            os.fsync(settings_output.fileno())
        os.replace(temporary_path, ROBOFLOW_SETTINGS_PATH)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def _delete_plaintext_roboflow_settings():
    try:
        settings_stat = os.lstat(ROBOFLOW_SETTINGS_PATH)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(settings_stat.st_mode):
        raise RoboflowStorageError("The plaintext Roboflow token path is invalid.")
    try:
        os.unlink(ROBOFLOW_SETTINGS_PATH)
    except OSError:
        raise RoboflowStorageError("The plaintext Roboflow token file could not be removed.") from None


def _get_roboflow_settings_row(connection):
    return connection.execute(
        """
        SELECT project_name, api_token_nonce, api_token_ciphertext, key_version
        FROM roboflow_settings
        WHERE id = 1
        """
    ).fetchone()


def _upsert_roboflow_settings_row(connection, project_name, nonce, ciphertext):
    connection.execute(
        """
        INSERT INTO roboflow_settings (
            id, project_name, api_token_nonce, api_token_ciphertext, key_version, updated_at
        ) VALUES (1, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            project_name = excluded.project_name,
            api_token_nonce = excluded.api_token_nonce,
            api_token_ciphertext = excluded.api_token_ciphertext,
            key_version = excluded.key_version,
            updated_at = excluded.updated_at
        """,
        (
            project_name,
            nonce,
            ciphertext,
            ROBOFLOW_KEY_VERSION,
            _utcnow_text(),
        ),
    )


def _inspect_roboflow_settings(connection):
    row = _get_roboflow_settings_row(connection)
    project_name = row["project_name"] if row is not None else ""
    secure_storage_status, master_key = _probe_roboflow_secure_storage()
    api_token = ""
    storage_mode = "none"
    usable = False

    encrypted_token_exists = (
        row is not None
        and (row["api_token_nonce"] is not None or row["api_token_ciphertext"] is not None)
    )
    if encrypted_token_exists:
        storage_mode = "encrypted_unavailable"
        if secure_storage_status == "available":
            try:
                api_token = _decrypt_roboflow_token(row, master_key)
                storage_mode = "encrypted"
                usable = bool(api_token)
            except RoboflowStorageError:
                pass
    else:
        try:
            plaintext_settings = _read_plaintext_roboflow_settings()
        except RoboflowStorageError:
            plaintext_settings = None
            storage_mode = "plaintext_file_unavailable"

        if plaintext_settings is not None:
            project_name = project_name or plaintext_settings["project_name"]
            api_token = plaintext_settings["api_token"]
            storage_mode = (
                "plaintext_file"
                if plaintext_settings["plaintext_storage_confirmed"]
                else "plaintext_file_unconfirmed"
            )
            usable = bool(api_token) and plaintext_settings["plaintext_storage_confirmed"]
        elif _roboflow_memory_api_token:
            api_token = _roboflow_memory_api_token
            storage_mode = "memory"
            usable = True

    configured = encrypted_token_exists or bool(api_token) or storage_mode == "plaintext_file_unavailable"
    return {
        "project_name": project_name,
        "api_token": api_token,
        "api_token_configured": configured,
        "api_token_usable": usable,
        "secure_storage_status": secure_storage_status,
        "secure_storage_available": secure_storage_status == "available",
        "api_token_storage": storage_mode,
        "api_token_persistent": storage_mode in ("encrypted", "plaintext_file"),
    }


def get_roboflow_settings(*, include_api_token=False):
    with _roboflow_settings_lock:
        connection = _connect()
        try:
            settings = _inspect_roboflow_settings(connection)
        finally:
            connection.close()

    result = {
        key: value
        for key, value in settings.items()
        if key != "api_token"
    }
    if include_api_token:
        if settings["api_token_storage"] == "encrypted_unavailable":
            raise RoboflowStorageError(
                "The encrypted Roboflow token is unavailable; enter a replacement token."
            )
        if settings["api_token_storage"] == "plaintext_file_unconfirmed":
            raise RoboflowStorageError(
                "Confirm plaintext token storage or move the token to memory."
            )
        if settings["api_token_storage"] == "plaintext_file_unavailable":
            raise RoboflowStorageError(
                "The plaintext Roboflow token cannot be read; enter a replacement token."
            )
        result["api_token"] = settings["api_token"]
    return result


def roboflow_configured():
    settings = get_roboflow_settings()
    return bool(settings.get("project_name") and settings.get("api_token_usable"))


def update_roboflow_settings(payload):
    global _roboflow_memory_api_token

    if not isinstance(payload, dict):
        raise ValueError("Invalid Roboflow settings payload.")

    confirmation_supplied = "allow_insecure_file_storage" in payload
    allow_insecure_file_storage = payload.get("allow_insecure_file_storage", False)
    if confirmation_supplied and not isinstance(allow_insecure_file_storage, bool):
        raise ValueError("Plaintext storage confirmation must be a boolean.")

    with _roboflow_settings_lock:
        connection = _connect()
        try:
            current = _inspect_roboflow_settings(connection)
            project_name = current["project_name"]
            if "project_name" in payload:
                project_name = _validate_roboflow_project_name(payload.get("project_name"))
            else:
                project_name = _validate_roboflow_project_name(project_name)

            replacement_token = ""
            if "api_token" in payload:
                raw_api_token = payload.get("api_token")
                if not isinstance(raw_api_token, str):
                    raise ValueError("Roboflow API token must be a string.")
                replacement_token = raw_api_token.strip()

            api_token = replacement_token or current["api_token"]
            if not api_token:
                if current["api_token_storage"] == "encrypted_unavailable":
                    raise RoboflowStorageError(
                        "Secure storage is unavailable; enter a replacement Roboflow token."
                    )
                if current["api_token_storage"] == "plaintext_file_unavailable":
                    raise RoboflowStorageError(
                        "The plaintext Roboflow token cannot be read; enter a replacement token."
                    )
                raise ValueError("Roboflow API token is required.")

            secure_storage_status, master_key = _probe_roboflow_secure_storage()
            if secure_storage_status == "available":
                nonce, ciphertext = _encrypt_roboflow_token(
                    api_token,
                    project_name,
                    master_key,
                )
                _upsert_roboflow_settings_row(
                    connection,
                    project_name,
                    nonce,
                    ciphertext,
                )
                connection.commit()
                _roboflow_memory_api_token = ""
                _delete_plaintext_roboflow_settings()
            else:
                store_in_plaintext_file = (
                    allow_insecure_file_storage
                    if confirmation_supplied
                    else current["api_token_storage"] == "plaintext_file"
                )
                if store_in_plaintext_file:
                    _write_plaintext_roboflow_settings(project_name, api_token)
                    _upsert_roboflow_settings_row(connection, project_name, None, None)
                    connection.commit()
                    _roboflow_memory_api_token = ""
                else:
                    _upsert_roboflow_settings_row(connection, project_name, None, None)
                    connection.commit()
                    _roboflow_memory_api_token = api_token
                    _delete_plaintext_roboflow_settings()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    return get_roboflow_settings()


def _initialize_roboflow_token_storage():
    global _roboflow_memory_api_token

    with _roboflow_settings_lock:
        connection = _connect()
        try:
            row = _get_roboflow_settings_row(connection)
            encrypted_token_exists = (
                row is not None
                and (
                    row["api_token_nonce"] is not None
                    or row["api_token_ciphertext"] is not None
                )
            )
            secure_storage_status, master_key = _probe_roboflow_secure_storage()

            if encrypted_token_exists:
                if secure_storage_status == "available":
                    try:
                        _decrypt_roboflow_token(row, master_key)
                    except RoboflowStorageError:
                        return
                    try:
                        _delete_plaintext_roboflow_settings()
                    except RoboflowStorageError:
                        LOGGER.warning(
                            "Could not remove an obsolete Roboflow plaintext settings path."
                        )
                return

            try:
                plaintext_settings = _read_plaintext_roboflow_settings()
                if plaintext_settings is None:
                    return
                project_name = _validate_roboflow_project_name(
                    plaintext_settings["project_name"]
                    or (row["project_name"] if row is not None else "")
                )
            except (RoboflowStorageError, ValueError):
                return
            api_token = plaintext_settings["api_token"]
            if not api_token:
                _upsert_roboflow_settings_row(connection, project_name, None, None)
                connection.commit()
                _delete_plaintext_roboflow_settings()
                return

            if secure_storage_status == "available":
                nonce, ciphertext = _encrypt_roboflow_token(
                    api_token,
                    project_name,
                    master_key,
                )
                _upsert_roboflow_settings_row(
                    connection,
                    project_name,
                    nonce,
                    ciphertext,
                )
                connection.commit()
                migrated_row = _get_roboflow_settings_row(connection)
                if _decrypt_roboflow_token(migrated_row, master_key) != api_token:
                    raise RoboflowStorageError("Roboflow token migration verification failed.")
                _roboflow_memory_api_token = ""
                _delete_plaintext_roboflow_settings()
            elif plaintext_settings["plaintext_storage_confirmed"]:
                _upsert_roboflow_settings_row(connection, project_name, None, None)
                connection.commit()
                _roboflow_memory_api_token = ""
            else:
                _upsert_roboflow_settings_row(connection, project_name, None, None)
                connection.commit()
                _roboflow_memory_api_token = api_token
                _delete_plaintext_roboflow_settings()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()


def _smtp_password_aad():
    return b"tilletia-app:smtp-settings:v1"


def _encrypt_smtp_password(password, master_key):
    nonce = os.urandom(ROBOFLOW_TOKEN_NONCE_BYTES)
    ciphertext = AESGCM(master_key).encrypt(
        nonce,
        password.encode("utf-8"),
        _smtp_password_aad(),
    )
    return nonce, ciphertext


def _decrypt_smtp_password(row, master_key):
    nonce = row["password_nonce"]
    ciphertext = row["password_ciphertext"]
    if nonce is None and ciphertext is None:
        return ""
    if nonce is None or ciphertext is None:
        raise RoboflowStorageError("The encrypted SMTP password is invalid.")

    nonce = bytes(nonce)
    ciphertext = bytes(ciphertext)
    if (
        row["key_version"] != SMTP_KEY_VERSION
        or len(nonce) != ROBOFLOW_TOKEN_NONCE_BYTES
        or len(ciphertext) < 16
    ):
        raise RoboflowStorageError("The encrypted SMTP password is invalid.")

    try:
        plaintext = AESGCM(master_key).decrypt(nonce, ciphertext, _smtp_password_aad())
        return plaintext.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError, ValueError):
        raise RoboflowStorageError("The encrypted SMTP password could not be decrypted.") from None


def _normalize_smtp_tls_mode(value):
    mode = str(value or "").strip().lower()
    return mode if mode in SMTP_TLS_MODES else "starttls"


def _coerce_smtp_port(value):
    try:
        port = int(value)
    except (TypeError, ValueError):
        return DEFAULT_SMTP_PORT
    return port if 1 <= port <= 65535 else DEFAULT_SMTP_PORT


def _get_smtp_settings_row(connection):
    return connection.execute(
        """
        SELECT host, port, username, sender, tls_mode,
               password_nonce, password_ciphertext, key_version
        FROM smtp_settings
        WHERE id = 1
        """
    ).fetchone()


def _upsert_smtp_settings_row(connection, host, port, username, sender, tls_mode, nonce, ciphertext):
    connection.execute(
        """
        INSERT INTO smtp_settings (
            id, host, port, username, sender, tls_mode,
            password_nonce, password_ciphertext, key_version, updated_at
        ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET
            host = excluded.host,
            port = excluded.port,
            username = excluded.username,
            sender = excluded.sender,
            tls_mode = excluded.tls_mode,
            password_nonce = excluded.password_nonce,
            password_ciphertext = excluded.password_ciphertext,
            key_version = excluded.key_version,
            updated_at = excluded.updated_at
        """,
        (
            host,
            port,
            username,
            sender,
            tls_mode,
            nonce,
            ciphertext,
            SMTP_KEY_VERSION,
            _utcnow_text(),
        ),
    )


def _smtp_env_settings():
    host = (os.environ.get("TILLETIA_SMTP_HOST") or "").strip()
    username = (os.environ.get("TILLETIA_SMTP_USER") or "").strip()
    sender = (os.environ.get("TILLETIA_SMTP_FROM") or "").strip()
    return {
        "host": host,
        "port": _coerce_smtp_port(os.environ.get("TILLETIA_SMTP_PORT")),
        "username": username,
        "sender": sender or username,
        "tls_mode": _normalize_smtp_tls_mode(os.environ.get("TILLETIA_SMTP_TLS")),
        "password": os.environ.get("TILLETIA_SMTP_PASSWORD") or "",
    }


def _inspect_smtp_settings(connection):
    global _smtp_memory_password

    row = _get_smtp_settings_row(connection)
    env_settings = _smtp_env_settings()

    if row is None or not (row["host"] or "").strip():
        password = env_settings["password"]
        return {
            "source": "env" if env_settings["host"] else "none",
            "host": env_settings["host"],
            "port": env_settings["port"],
            "username": env_settings["username"],
            "sender": env_settings["sender"],
            "tls_mode": env_settings["tls_mode"],
            "password": password,
            "password_configured": bool(password),
            "password_storage": "env" if password else "none",
            "secure_storage_available": _probe_roboflow_secure_storage()[0] == "available",
        }

    secure_storage_status, master_key = _probe_roboflow_secure_storage()
    encrypted_password_exists = (
        row["password_nonce"] is not None or row["password_ciphertext"] is not None
    )

    password = ""
    password_storage = "none"
    if encrypted_password_exists:
        password_storage = "encrypted_unavailable"
        if secure_storage_status == "available":
            try:
                password = _decrypt_smtp_password(row, master_key)
                password_storage = "encrypted"
            except RoboflowStorageError:
                password = ""
                password_storage = "encrypted_unavailable"
    elif _smtp_memory_password:
        password = _smtp_memory_password
        password_storage = "memory"

    return {
        "source": "db",
        "host": (row["host"] or "").strip(),
        "port": _coerce_smtp_port(row["port"]),
        "username": (row["username"] or "").strip(),
        "sender": (row["sender"] or "").strip(),
        "tls_mode": _normalize_smtp_tls_mode(row["tls_mode"]),
        "password": password,
        "password_configured": bool(password) or encrypted_password_exists,
        "password_storage": password_storage,
        "secure_storage_available": secure_storage_status == "available",
    }


def get_smtp_settings(*, include_password=False):
    with _smtp_settings_lock:
        connection = _connect()
        try:
            settings = _inspect_smtp_settings(connection)
        finally:
            connection.close()

    result = {key: value for key, value in settings.items() if key != "password"}
    if include_password:
        result["password"] = settings["password"]
    return result


def smtp_configured():
    settings = get_smtp_settings()
    return bool(settings["host"] and settings["sender"])


def update_smtp_settings(payload):
    global _smtp_memory_password

    if not isinstance(payload, dict):
        raise ValueError("Invalid SMTP settings payload.")

    with _smtp_settings_lock:
        connection = _connect()
        try:
            current = _inspect_smtp_settings(connection)

            host = str(payload.get("host", current["host"]) or "").strip()
            if not host:
                raise ValueError("SMTP host is required.")

            sender_raw = payload.get("sender", current["sender"])
            sender = str(sender_raw or "").strip()
            if not sender:
                raise ValueError("Sender address is required.")

            username = str(payload.get("username", current["username"]) or "").strip()
            port = _coerce_smtp_port(payload.get("port", current["port"]))
            tls_mode = _normalize_smtp_tls_mode(payload.get("tls_mode", current["tls_mode"]))

            replacement_password = ""
            if "password" in payload:
                raw_password = payload.get("password")
                if not isinstance(raw_password, str):
                    raise ValueError("SMTP password must be a string.")
                replacement_password = raw_password
            password = replacement_password or current["password"]

            nonce = None
            ciphertext = None
            if password:
                secure_storage_status, master_key = _probe_roboflow_secure_storage()
                if secure_storage_status == "available":
                    nonce, ciphertext = _encrypt_smtp_password(password, master_key)
                    _smtp_memory_password = ""
                else:
                    _smtp_memory_password = password
            else:
                _smtp_memory_password = ""

            _upsert_smtp_settings_row(
                connection, host, port, username, sender, tls_mode, nonce, ciphertext
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    return get_smtp_settings()


def get_model_owner_username(model_type, model_name):
    connection = _connect()
    try:
        row = connection.execute(
            """
            SELECT users.username
            FROM model_owners
            JOIN users ON users.id = model_owners.owner_user_id
            WHERE model_owners.model_type = ? AND model_owners.model_name = ?
            """,
            (model_type, model_name),
        ).fetchone()
        return row["username"] if row else "system"
    finally:
        connection.close()


def get_result_owner_user_id(run_id):
    connection = _connect()
    try:
        row = connection.execute("SELECT owner_user_id FROM results WHERE run_id = ?", (run_id,)).fetchone()
        return row["owner_user_id"] if row else None
    finally:
        connection.close()


def get_user_by_id(user_id):
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT id, username, password_hash, active, force_password_change FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return _user_from_row(row, _get_roles(connection, row["id"])) if row else None
    finally:
        connection.close()


def get_user_by_username(username):
    connection = _connect()
    try:
        row = connection.execute(
            "SELECT id, username, password_hash, active, force_password_change FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return _user_from_row(row, _get_roles(connection, row["id"])) if row else None
    finally:
        connection.close()


def get_result_owner_username(hq_output_dir, run_id):
    owner_user_id = get_result_owner_user_id(run_id)
    if owner_user_id is not None:
        owner = get_user_by_id(owner_user_id)
        if owner is not None:
            return owner["username"]
    metadata_path = os.path.join(hq_output_dir, run_id, "metadata.json")
    if not os.path.isfile(metadata_path):
        return "unknown"
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_input:
            metadata = json.load(metadata_input)
    except Exception:
        return "unknown"
    return metadata.get("owner") or "unknown"


def list_users():
    connection = _connect()
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


def init_auth_storage():
    with _init_lock:
        connection = _connect()
        try:
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

                CREATE TABLE IF NOT EXISTS results (
                    run_id TEXT PRIMARY KEY,
                    owner_user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS model_owners (
                    model_type TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    owner_user_id INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (model_type, model_name)
                );

                CREATE TABLE IF NOT EXISTS dashboard_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    analysis_number TEXT NOT NULL DEFAULT '',
                    source_type TEXT NOT NULL DEFAULT 'camera',
                    camera_device TEXT NOT NULL DEFAULT '',
                    camera_width INTEGER NOT NULL DEFAULT 1280,
                    camera_height INTEGER NOT NULL DEFAULT 720,
                    camera_fps INTEGER NOT NULL DEFAULT 30,
                    camera_format TEXT NOT NULL DEFAULT 'MJPG',
                    uploaded_path TEXT NOT NULL DEFAULT '',
                    model_path TEXT NOT NULL DEFAULT '',
                    vis_conf REAL NOT NULL DEFAULT 0.75,
                    captions_enabled INTEGER NOT NULL DEFAULT 1,
                    grid_count_enabled INTEGER NOT NULL DEFAULT 0,
                    grid_debug_enabled INTEGER NOT NULL DEFAULT 0,
                    grid_score_threshold REAL NOT NULL DEFAULT 0.30,
                    ask_manual_spore_count INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS roboflow_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    project_name TEXT NOT NULL DEFAULT '',
                    api_token_nonce BLOB,
                    api_token_ciphertext BLOB,
                    key_version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    CHECK (
                        (api_token_nonce IS NULL) = (api_token_ciphertext IS NULL)
                    )
                );

                CREATE TABLE IF NOT EXISTS smtp_settings (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    host TEXT NOT NULL DEFAULT '',
                    port INTEGER NOT NULL DEFAULT 587,
                    username TEXT NOT NULL DEFAULT '',
                    sender TEXT NOT NULL DEFAULT '',
                    tls_mode TEXT NOT NULL DEFAULT 'starttls',
                    password_nonce BLOB,
                    password_ciphertext BLOB,
                    key_version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL,
                    CHECK (
                        (password_nonce IS NULL) = (password_ciphertext IS NULL)
                    )
                );
                """
            )

            if not _has_column(connection, "users", "force_password_change"):
                connection.execute(
                    "ALTER TABLE users ADD COLUMN force_password_change INTEGER NOT NULL DEFAULT 0"
                )

            if not _has_column(connection, "dashboard_settings", "ask_manual_spore_count"):
                connection.execute(
                    "ALTER TABLE dashboard_settings ADD COLUMN ask_manual_spore_count INTEGER NOT NULL DEFAULT 1"
                )

            if not _has_column(connection, "dashboard_settings", "captions_enabled"):
                connection.execute(
                    "ALTER TABLE dashboard_settings ADD COLUMN captions_enabled INTEGER NOT NULL DEFAULT 1"
                )

            if connection.execute("SELECT id FROM dashboard_settings WHERE id = 1").fetchone() is None:
                connection.execute(
                    """
                    INSERT INTO dashboard_settings (
                        id,
                        analysis_number,
                        source_type,
                        camera_device,
                        camera_width,
                        camera_height,
                        camera_fps,
                        camera_format,
                        uploaded_path,
                        model_path,
                        vis_conf,
                        captions_enabled,
                        grid_count_enabled,
                        grid_debug_enabled,
                        grid_score_threshold,
                        ask_manual_spore_count,
                        updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        1,
                        DEFAULT_DASHBOARD_SETTINGS["analysis_number"],
                        DEFAULT_DASHBOARD_SETTINGS["source_type"],
                        DEFAULT_DASHBOARD_SETTINGS["camera_device"],
                        DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["width"],
                        DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["height"],
                        DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["fps"],
                        DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["format"],
                        DEFAULT_DASHBOARD_SETTINGS["uploaded_path"],
                        DEFAULT_DASHBOARD_SETTINGS["model_path"],
                        DEFAULT_DASHBOARD_SETTINGS["vis_conf"],
                        int(DEFAULT_DASHBOARD_SETTINGS["captions_enabled"]),
                        int(DEFAULT_DASHBOARD_SETTINGS["grid_count_enabled"]),
                        int(DEFAULT_DASHBOARD_SETTINGS["grid_debug_enabled"]),
                        DEFAULT_DASHBOARD_SETTINGS["grid_score_threshold"],
                        int(DEFAULT_DASHBOARD_SETTINGS["ask_manual_spore_count"]),
                        _utcnow_text(),
                    ),
                )

            if connection.execute("SELECT id FROM users WHERE username = ?", ("admin",)).fetchone() is None:
                cursor = connection.execute(
                    """
                    INSERT INTO users (username, password_hash, active, force_password_change, created_at)
                    VALUES (?, ?, 1, 0, ?)
                    """,
                    ("admin", _hash_password("admin"), _utcnow_text()),
                )
                connection.execute(
                    "INSERT INTO user_roles (user_id, role) VALUES (?, ?)",
                    (cursor.lastrowid, "admin"),
                )
            else:
                connection.execute(
                    "UPDATE users SET force_password_change = 0 WHERE username = ?",
                    ("admin",),
                )

            connection.commit()
        finally:
            connection.close()

        _initialize_roboflow_token_storage()


def is_admin(user):
    return user_has_permission(user, "results:delete")


def issue_refresh_token(user_id):
    raw_token = tokens.generate_refresh_token()
    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO refresh_tokens (user_id, token_hash, expires_at, revoked_at, created_at)
            VALUES (?, ?, ?, NULL, ?)
            """,
            (
                user_id,
                tokens.hash_refresh_token(raw_token),
                (_utcnow() + timedelta(seconds=tokens.REFRESH_TOKEN_TTL_SECONDS)).isoformat() + "Z",
                _utcnow_text(),
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
    connection = _connect()
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
        if datetime.fromisoformat(expires_at) <= _utcnow():
            return None

        return get_user_by_id(row["user_id"])
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

    connection = _connect()
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
        user = get_user_by_id(user_id)
        if user is not None:
            flask.g._current_user = user
        return user

    cookie_access_token = tokens.get_cookie_access_token(flask.request)
    if cookie_access_token:
        user_id = tokens.verify_access_token(cookie_access_token)
        if user_id:
            user = get_user_by_id(user_id)
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


def resolve_dashboard_start_payload(raw_payload, user):
    settings = get_dashboard_settings()
    raw_payload = raw_payload or {}

    if not user_has_permission(user, "dashboard:configure") and raw_payload:
        raise PermissionError("Regular users cannot override dashboard settings.")

    mode = settings.get("camera_mode") or DEFAULT_DASHBOARD_SETTINGS["camera_mode"]
    return {
        "analysis_number": raw_payload.get("analysis_number", settings.get("analysis_number", "")),
        "source_type": raw_payload.get("source_type", settings.get("source_type", "camera")),
        "device": raw_payload.get("device", settings.get("camera_device", "")),
        "width": int(raw_payload.get("width", mode.get("width", 1280))),
        "height": int(raw_payload.get("height", mode.get("height", 720))),
        "fps": int(raw_payload.get("fps", mode.get("fps", 30))),
        "format": raw_payload.get("format", mode.get("format", "MJPG")),
        "video": raw_payload.get("video", settings.get("uploaded_path", "")),
        "model_path": raw_payload.get("model_path", settings.get("model_path", "")),
        "model_task": raw_payload.get("model_task"),
        "vis_conf": raw_payload.get("vis_conf", settings.get("vis_conf", 0.75)),
        "captions_enabled": raw_payload.get("captions_enabled", settings.get("captions_enabled", True)),
        "grid_count_enabled": raw_payload.get("grid_count_enabled", settings.get("grid_count_enabled", False)),
        "grid_debug_enabled": raw_payload.get("grid_debug_enabled", settings.get("grid_debug_enabled", False)),
        "grid_score_threshold": raw_payload.get("grid_score_threshold", settings.get("grid_score_threshold", 0.30)),
    }


def revoke_refresh_token(raw_token):
    if not raw_token:
        return
    connection = _connect()
    try:
        connection.execute(
            "UPDATE refresh_tokens SET revoked_at = ? WHERE token_hash = ? AND revoked_at IS NULL",
            (_utcnow_text(), tokens.hash_refresh_token(raw_token)),
        )
        connection.commit()
    finally:
        connection.close()


def rotate_refresh_token(raw_token):
    token_hash = tokens.hash_refresh_token(raw_token)
    connection = _connect()
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
        if datetime.fromisoformat(expires_at) <= _utcnow():
            return None

        connection.execute(
            "UPDATE refresh_tokens SET revoked_at = ? WHERE id = ?",
            (_utcnow_text(), row["id"]),
        )
        connection.commit()

        user = get_user_by_id(row["user_id"])
        if user is None:
            return None
        return {
            "user": user,
            "refresh_token": issue_refresh_token(user["id"]),
        }
    finally:
        connection.close()


def store_result_owner(run_id, owner_user_id):
    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO results (run_id, owner_user_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                created_at = excluded.created_at
            """,
            (run_id, owner_user_id, _utcnow_text()),
        )
        connection.commit()
    finally:
        connection.close()


def store_model_owner(model_type, model_name, owner_user_id):
    connection = _connect()
    try:
        connection.execute(
            """
            INSERT INTO model_owners (model_type, model_name, owner_user_id, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_type, model_name) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                created_at = excluded.created_at
            """,
            (model_type, model_name, owner_user_id, _utcnow_text()),
        )
        connection.commit()
    finally:
        connection.close()


def delete_model_owner(model_type, model_name):
    candidates = {model_name}
    if model_name.endswith("-fp16"):
        candidates.add(model_name[:-5])
    connection = _connect()
    try:
        connection.executemany(
            "DELETE FROM model_owners WHERE model_type = ? AND model_name = ?",
            [(model_type, candidate) for candidate in candidates],
        )
        connection.commit()
    finally:
        connection.close()


def unauthorized_response(*, html_redirect=False):
    if html_redirect and _request_wants_html() and not tokens.get_authorization_access_token(flask.request):
        response = flask.redirect(flask.url_for("login_page"))
    else:
        response = flask.make_response(flask.jsonify({"error": "Unauthorized"}), 401)

    if not tokens.get_authorization_access_token(flask.request):
        response = cookies.clear_auth_cookies(response)
    return response


def update_dashboard_settings(payload):
    current = get_dashboard_settings()
    updated = {
        "analysis_number": current["analysis_number"],
        "source_type": current["source_type"],
        "camera_device": current["camera_device"],
        "camera_mode": dict(current["camera_mode"]),
        "uploaded_path": current["uploaded_path"],
        "model_path": current["model_path"],
        "vis_conf": current["vis_conf"],
        "captions_enabled": current["captions_enabled"],
        "grid_count_enabled": current["grid_count_enabled"],
        "grid_debug_enabled": current["grid_debug_enabled"],
        "grid_score_threshold": current["grid_score_threshold"],
        "ask_manual_spore_count": current["ask_manual_spore_count"],
    }

    if "analysis_number" in payload:
        updated["analysis_number"] = str(payload.get("analysis_number") or "")
    if "source_type" in payload:
        source_type = str(payload.get("source_type") or "").strip().lower()
        if source_type not in ("camera", "file"):
            raise ValueError("Invalid source_type.")
        updated["source_type"] = source_type
    if "camera_device" in payload:
        updated["camera_device"] = str(payload.get("camera_device") or "")
    if "camera_mode" in payload and isinstance(payload.get("camera_mode"), dict):
        mode = payload["camera_mode"]
        updated["camera_mode"] = {
            "width": int(mode.get("width", updated["camera_mode"]["width"])),
            "height": int(mode.get("height", updated["camera_mode"]["height"])),
            "fps": int(mode.get("fps", updated["camera_mode"]["fps"])),
            "format": str(mode.get("format", updated["camera_mode"]["format"]) or "MJPG"),
        }
    if "uploaded_path" in payload:
        updated["uploaded_path"] = str(payload.get("uploaded_path") or "")
    if "model_path" in payload:
        updated["model_path"] = str(payload.get("model_path") or "")
    if "vis_conf" in payload:
        updated["vis_conf"] = float(payload.get("vis_conf"))
    if "captions_enabled" in payload:
        updated["captions_enabled"] = bool(payload.get("captions_enabled"))
    if "grid_count_enabled" in payload:
        updated["grid_count_enabled"] = bool(payload.get("grid_count_enabled"))
    if "grid_debug_enabled" in payload:
        updated["grid_debug_enabled"] = bool(payload.get("grid_debug_enabled"))
    if "grid_score_threshold" in payload:
        updated["grid_score_threshold"] = float(payload.get("grid_score_threshold"))
    if "ask_manual_spore_count" in payload:
        updated["ask_manual_spore_count"] = bool(payload.get("ask_manual_spore_count"))

    connection = _connect()
    try:
        connection.execute(
            """
            UPDATE dashboard_settings
            SET analysis_number = ?,
                source_type = ?,
                camera_device = ?,
                camera_width = ?,
                camera_height = ?,
                camera_fps = ?,
                camera_format = ?,
                uploaded_path = ?,
                model_path = ?,
                vis_conf = ?,
                captions_enabled = ?,
                grid_count_enabled = ?,
                grid_debug_enabled = ?,
                grid_score_threshold = ?,
                ask_manual_spore_count = ?,
                updated_at = ?
            WHERE id = 1
            """,
            (
                updated["analysis_number"],
                updated["source_type"],
                updated["camera_device"],
                updated["camera_mode"]["width"],
                updated["camera_mode"]["height"],
                updated["camera_mode"]["fps"],
                updated["camera_mode"]["format"],
                updated["uploaded_path"],
                updated["model_path"],
                updated["vis_conf"],
                int(updated["captions_enabled"]),
                int(updated["grid_count_enabled"]),
                int(updated["grid_debug_enabled"]),
                updated["grid_score_threshold"],
                int(updated["ask_manual_spore_count"]),
                _utcnow_text(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return updated


def user_can_access_result(user, hq_output_dir, run_id):
    if is_admin(user):
        return True

    owner_user_id = get_result_owner_user_id(run_id)
    if owner_user_id is None:
        owner_user_id = _get_result_owner_from_legacy_metadata(hq_output_dir, run_id)
    if owner_user_id is None:
        return False
    return owner_user_id == user["id"]


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
