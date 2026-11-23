"""AES-GCM encryption with the TPM-provisioned master key, plus the owner-only
plaintext-file fallback shared by the Roboflow token and SMTP password settings."""

import json
import os
import stat
import tempfile

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


MASTER_KEY_PATH = "/run/secrets/roboflow-master-key"
NONCE_BYTES = 12
_PROBE_AAD = b"tilletia-app:roboflow-storage-probe:v1"
_MAX_PLAINTEXT_FILE_CHARS = 65_536


class SecretStorageError(RuntimeError):
    pass


def _read_only_flags():
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def load_master_key():
    try:
        file_descriptor = os.open(MASTER_KEY_PATH, _read_only_flags())
    except OSError:
        raise SecretStorageError("Secure secret storage is unavailable.") from None

    try:
        key_stat = os.fstat(file_descriptor)
        if (
            not stat.S_ISREG(key_stat.st_mode)
            or stat.S_IMODE(key_stat.st_mode) & 0o077
        ):
            raise SecretStorageError("Secure secret storage is unavailable.")
        with os.fdopen(file_descriptor, "rb") as key_input:
            file_descriptor = None
            master_key = key_input.read(33)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    if len(master_key) != 32:
        raise SecretStorageError("Secure secret storage is unavailable.")
    return master_key


def probe_secure_storage():
    if not os.path.exists(MASTER_KEY_PATH):
        return "unavailable", None
    try:
        master_key = load_master_key()
        nonce = os.urandom(NONCE_BYTES)
        encrypted_probe = AESGCM(master_key).encrypt(nonce, b"probe", _PROBE_AAD)
        if AESGCM(master_key).decrypt(nonce, encrypted_probe, _PROBE_AAD) != b"probe":
            raise SecretStorageError("Secure secret storage is unavailable.")
        return "available", master_key
    except (InvalidTag, OSError, ValueError, SecretStorageError):
        return "error", None


def encrypt(plaintext, aad, master_key):
    nonce = os.urandom(NONCE_BYTES)
    ciphertext = AESGCM(master_key).encrypt(nonce, plaintext.encode("utf-8"), aad)
    return nonce, ciphertext


def decrypt(nonce, ciphertext, *, key_version, expected_key_version, aad, master_key, label):
    if nonce is None and ciphertext is None:
        return ""
    if nonce is None or ciphertext is None:
        raise SecretStorageError(f"The encrypted {label} is invalid.")

    nonce = bytes(nonce)
    ciphertext = bytes(ciphertext)
    if (
        key_version != expected_key_version
        or len(nonce) != NONCE_BYTES
        or len(ciphertext) < 16
    ):
        raise SecretStorageError(f"The encrypted {label} is invalid.")

    try:
        plaintext = AESGCM(master_key).decrypt(nonce, ciphertext, aad)
        return plaintext.decode("utf-8")
    except (InvalidTag, UnicodeDecodeError, ValueError):
        raise SecretStorageError(f"The encrypted {label} could not be decrypted.") from None


def read_plaintext_json(path, label):
    try:
        settings_stat = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        raise SecretStorageError(f"The plaintext {label} file could not be read.") from None

    if not stat.S_ISREG(settings_stat.st_mode):
        raise SecretStorageError(f"The plaintext {label} path is invalid.")

    file_descriptor = None
    try:
        file_descriptor = os.open(path, _read_only_flags())
        opened_stat = os.fstat(file_descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            raise OSError("Not a regular file")
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "r", encoding="utf-8") as settings_input:
            file_descriptor = None
            serialized_settings = settings_input.read(_MAX_PLAINTEXT_FILE_CHARS + 1)
    except (OSError, UnicodeDecodeError):
        raise SecretStorageError(f"The plaintext {label} file could not be read.") from None
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)

    if len(serialized_settings) > _MAX_PLAINTEXT_FILE_CHARS:
        raise SecretStorageError(f"The plaintext {label} file is invalid.")
    try:
        settings = json.loads(serialized_settings)
    except ValueError:
        raise SecretStorageError(f"The plaintext {label} file is invalid.") from None
    if not isinstance(settings, dict):
        raise SecretStorageError(f"The plaintext {label} file is invalid.")
    return settings


def write_plaintext_json(path, temporary_prefix, data):
    settings_dir = os.path.dirname(path)
    os.makedirs(settings_dir, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(prefix=temporary_prefix, dir=settings_dir)
    try:
        os.fchmod(file_descriptor, 0o600)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as settings_output:
            file_descriptor = None
            json.dump(data, settings_output)
            settings_output.flush()
            os.fsync(settings_output.fileno())
        os.replace(temporary_path, path)
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def delete_plaintext_file(path, label):
    try:
        settings_stat = os.lstat(path)
    except FileNotFoundError:
        return
    if not stat.S_ISREG(settings_stat.st_mode):
        raise SecretStorageError(f"The plaintext {label} path is invalid.")
    try:
        os.unlink(path)
    except OSError:
        raise SecretStorageError(f"The plaintext {label} file could not be removed.") from None
