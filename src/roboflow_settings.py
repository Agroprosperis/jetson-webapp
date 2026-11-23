import logging
import re
import threading

import db
import secret_store

from secret_store import SecretStorageError


ROBOFLOW_SETTINGS_PATH = "/app/runs/roboflow.json"
ROBOFLOW_KEY_VERSION = 1
_TOKEN_LABEL = "Roboflow token"

_roboflow_settings_lock = threading.Lock()
_roboflow_memory_api_token = ""
LOGGER = logging.getLogger(__name__)


def init_schema(connection):
    connection.executescript(
        """
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
        """
    )


def _validate_roboflow_project_name(value):
    if not isinstance(value, str):
        raise ValueError("Roboflow project name must be a string.")
    project_name = value.strip()
    if not project_name:
        raise ValueError("Roboflow project name is required.")
    if len(project_name) > 255 or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", project_name) is None:
        raise ValueError("Use a valid Roboflow project ID, without the workspace or URL.")
    return project_name


def _roboflow_token_aad(project_name):
    return b"tilletia-app:roboflow-settings:v1|" + project_name.encode("utf-8")


def _encrypt_roboflow_token(api_token, project_name, master_key):
    return secret_store.encrypt(api_token, _roboflow_token_aad(project_name), master_key)


def _decrypt_roboflow_token(row, master_key):
    return secret_store.decrypt(
        row["api_token_nonce"],
        row["api_token_ciphertext"],
        key_version=row["key_version"],
        expected_key_version=ROBOFLOW_KEY_VERSION,
        aad=_roboflow_token_aad(row["project_name"]),
        master_key=master_key,
        label=_TOKEN_LABEL,
    )


def _read_plaintext_roboflow_settings():
    settings = secret_store.read_plaintext_json(ROBOFLOW_SETTINGS_PATH, _TOKEN_LABEL)
    if settings is None:
        return None

    project_name = settings.get("project_name", "")
    api_token = settings.get("api_token", "")
    if not isinstance(project_name, str) or not isinstance(api_token, str):
        raise SecretStorageError("The plaintext Roboflow token file is invalid.")
    return {
        "project_name": project_name.strip(),
        "api_token": api_token.strip(),
        "plaintext_storage_confirmed": settings.get("plaintext_storage_confirmed") is True,
    }


def _write_plaintext_roboflow_settings(project_name, api_token):
    secret_store.write_plaintext_json(
        ROBOFLOW_SETTINGS_PATH,
        ".roboflow-",
        {
            "schema_version": 1,
            "project_name": project_name,
            "api_token": api_token,
            "plaintext_storage_confirmed": True,
        },
    )


def _delete_plaintext_roboflow_settings():
    secret_store.delete_plaintext_file(ROBOFLOW_SETTINGS_PATH, _TOKEN_LABEL)


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
            db.utcnow_text(),
        ),
    )


def _inspect_roboflow_settings(connection):
    row = _get_roboflow_settings_row(connection)
    project_name = row["project_name"] if row is not None else ""
    secure_storage_status, master_key = secret_store.probe_secure_storage()
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
            except SecretStorageError:
                pass
    else:
        try:
            plaintext_settings = _read_plaintext_roboflow_settings()
        except SecretStorageError:
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
        connection = db.connect()
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
            raise SecretStorageError(
                "The encrypted Roboflow token is unavailable; enter a replacement token."
            )
        if settings["api_token_storage"] == "plaintext_file_unconfirmed":
            raise SecretStorageError(
                "Confirm plaintext token storage or move the token to memory."
            )
        if settings["api_token_storage"] == "plaintext_file_unavailable":
            raise SecretStorageError(
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
        connection = db.connect()
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
                    raise SecretStorageError(
                        "Secure storage is unavailable; enter a replacement Roboflow token."
                    )
                if current["api_token_storage"] == "plaintext_file_unavailable":
                    raise SecretStorageError(
                        "The plaintext Roboflow token cannot be read; enter a replacement token."
                    )
                raise ValueError("Roboflow API token is required.")

            secure_storage_status, master_key = secret_store.probe_secure_storage()
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


def initialize_roboflow_token_storage():
    global _roboflow_memory_api_token

    with _roboflow_settings_lock:
        connection = db.connect()
        try:
            row = _get_roboflow_settings_row(connection)
            encrypted_token_exists = (
                row is not None
                and (
                    row["api_token_nonce"] is not None
                    or row["api_token_ciphertext"] is not None
                )
            )
            secure_storage_status, master_key = secret_store.probe_secure_storage()

            if encrypted_token_exists:
                if secure_storage_status == "available":
                    try:
                        _decrypt_roboflow_token(row, master_key)
                    except SecretStorageError:
                        return
                    try:
                        _delete_plaintext_roboflow_settings()
                    except SecretStorageError:
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
            except (SecretStorageError, ValueError):
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
                    raise SecretStorageError("Roboflow token migration verification failed.")
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
