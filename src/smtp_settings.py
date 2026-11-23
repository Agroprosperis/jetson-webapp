import os
import threading

import db
import secret_store

from secret_store import SecretStorageError


SMTP_SETTINGS_PATH = "/app/runs/smtp.json"
SMTP_KEY_VERSION = 1
SMTP_TLS_MODES = ("starttls", "ssl", "none")
DEFAULT_SMTP_PORT = 587
_PASSWORD_LABEL = "SMTP password"

_smtp_settings_lock = threading.Lock()
_smtp_memory_password = ""


def init_schema(connection):
    connection.executescript(
        """
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


def _smtp_password_aad():
    return b"tilletia-app:smtp-settings:v1"


def _encrypt_smtp_password(password, master_key):
    return secret_store.encrypt(password, _smtp_password_aad(), master_key)


def _decrypt_smtp_password(row, master_key):
    return secret_store.decrypt(
        row["password_nonce"],
        row["password_ciphertext"],
        key_version=row["key_version"],
        expected_key_version=SMTP_KEY_VERSION,
        aad=_smtp_password_aad(),
        master_key=master_key,
        label=_PASSWORD_LABEL,
    )


def _read_plaintext_smtp_password():
    settings = secret_store.read_plaintext_json(SMTP_SETTINGS_PATH, _PASSWORD_LABEL)
    if settings is None:
        return None

    password = settings.get("password", "")
    if not isinstance(password, str):
        raise SecretStorageError("The plaintext SMTP password file is invalid.")
    return {
        "password": password,
        "plaintext_storage_confirmed": settings.get("plaintext_storage_confirmed") is True,
    }


def _write_plaintext_smtp_password(password):
    secret_store.write_plaintext_json(
        SMTP_SETTINGS_PATH,
        ".smtp-",
        {
            "schema_version": 1,
            "password": password,
            "plaintext_storage_confirmed": True,
        },
    )


def _delete_plaintext_smtp_password():
    secret_store.delete_plaintext_file(SMTP_SETTINGS_PATH, _PASSWORD_LABEL)


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
            db.utcnow_text(),
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
            "password_persistent": bool(password),
            "secure_storage_available": secret_store.probe_secure_storage()[0] == "available",
        }

    secure_storage_status, master_key = secret_store.probe_secure_storage()
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
            except SecretStorageError:
                password = ""
                password_storage = "encrypted_unavailable"
    else:
        try:
            plaintext_settings = _read_plaintext_smtp_password()
        except SecretStorageError:
            plaintext_settings = None
            password_storage = "plaintext_file_unavailable"

        if plaintext_settings is not None:
            password = plaintext_settings["password"]
            password_storage = (
                "plaintext_file"
                if plaintext_settings["plaintext_storage_confirmed"]
                else "plaintext_file_unconfirmed"
            )
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
        "password_configured": (
            bool(password)
            or encrypted_password_exists
            or password_storage == "plaintext_file_unavailable"
        ),
        "password_storage": password_storage,
        "password_persistent": password_storage in ("encrypted", "plaintext_file"),
        "secure_storage_available": secure_storage_status == "available",
    }


def get_smtp_settings(*, include_password=False):
    with _smtp_settings_lock:
        connection = db.connect()
        try:
            settings = _inspect_smtp_settings(connection)
        finally:
            connection.close()

    result = {key: value for key, value in settings.items() if key != "password"}
    if include_password:
        if settings["password_storage"] in (
            "encrypted_unavailable",
            "plaintext_file_unconfirmed",
            "plaintext_file_unavailable",
        ):
            raise SecretStorageError("The stored SMTP password is not available.")
        result["password"] = settings["password"]
    return result


def smtp_configured():
    settings = get_smtp_settings()
    return bool(settings["host"] and settings["sender"])


def resolve_smtp_settings(overrides=None):
    """Merge the saved SMTP settings with per-request overrides."""
    overrides = overrides if isinstance(overrides, dict) else {}
    override_password = str(overrides.get("password") or "")
    saved = get_smtp_settings(include_password=not override_password)
    return {
        "host": str(overrides.get("host", saved["host"]) or "").strip(),
        "port": _coerce_smtp_port(overrides.get("port", saved["port"])),
        "username": str(overrides.get("username", saved["username"]) or "").strip(),
        "sender": str(overrides.get("sender", saved["sender"]) or "").strip(),
        "tls_mode": _normalize_smtp_tls_mode(overrides.get("tls_mode", saved["tls_mode"])),
        "password": override_password or saved.get("password", ""),
    }


def update_smtp_settings(payload):
    global _smtp_memory_password

    if not isinstance(payload, dict):
        raise ValueError("Invalid SMTP settings payload.")

    confirmation_supplied = "allow_insecure_file_storage" in payload
    allow_insecure_file_storage = payload.get("allow_insecure_file_storage", False)
    if confirmation_supplied and not isinstance(allow_insecure_file_storage, bool):
        raise ValueError("Plaintext storage confirmation must be a boolean.")

    with _smtp_settings_lock:
        connection = db.connect()
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
            store_in_plaintext_file = False
            if password:
                secure_storage_status, master_key = secret_store.probe_secure_storage()
                if secure_storage_status == "available":
                    nonce, ciphertext = _encrypt_smtp_password(password, master_key)
                else:
                    store_in_plaintext_file = (
                        allow_insecure_file_storage
                        if confirmation_supplied
                        else current["password_storage"] == "plaintext_file"
                    )

            _upsert_smtp_settings_row(
                connection, host, port, username, sender, tls_mode, nonce, ciphertext
            )
            connection.commit()

            if not password or ciphertext is not None:
                _smtp_memory_password = ""
                _delete_plaintext_smtp_password()
            elif store_in_plaintext_file:
                _write_plaintext_smtp_password(password)
                _smtp_memory_password = ""
            else:
                _smtp_memory_password = password
                _delete_plaintext_smtp_password()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    return get_smtp_settings()
