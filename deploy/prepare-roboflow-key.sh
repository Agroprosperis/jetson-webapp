#!/usr/bin/env bash
set -uo pipefail

RUNTIME_DIR="/run/tilletia/secrets"
RUNTIME_KEY="$RUNTIME_DIR/roboflow-master-key"
CREDENTIAL_DIR="/var/lib/tilletia-app/credentials"
CREDENTIAL_BLOB="$CREDENTIAL_DIR/roboflow-master-key.cred"
CREDENTIAL_NAME="roboflow-master-key"
encrypted_tmp=""
runtime_tmp=""

warn() {
  echo "Roboflow secure storage unavailable: $*" >&2
}

cleanup() {
  if [[ -n "$runtime_tmp" ]]; then
    rm -f -- "$runtime_tmp"
  fi
  if [[ -n "$encrypted_tmp" ]]; then
    rm -f -- "$encrypted_tmp"
  fi
}
trap cleanup EXIT
trap 'exit 0' HUP INT TERM
umask 077

if [[ -L "$RUNTIME_DIR" ]]; then
  warn "$RUNTIME_DIR must not be a symlink"
  exit 0
fi

if ! rm -f -- "$RUNTIME_KEY"; then
  warn "could not clear the previous runtime key"
  exit 0
fi
if [[ -e "$RUNTIME_KEY" || -L "$RUNTIME_KEY" ]]; then
  warn "the runtime key path is not a replaceable regular file"
  exit 0
fi

if ! command -v findmnt >/dev/null 2>&1; then
  warn "findmnt is not installed"
  exit 0
fi
runtime_fs="$(findmnt -n -o FSTYPE --target "$RUNTIME_DIR" 2>/dev/null || true)"
if [[ "$runtime_fs" != "ramfs" ]]; then
  warn "$RUNTIME_DIR is not mounted as ramfs"
  exit 0
fi

if ! command -v systemd-creds >/dev/null 2>&1; then
  warn "systemd-creds is not installed"
  exit 0
fi
if ! systemd-creds has-tpm2 >/dev/null 2>&1; then
  warn "systemd does not report an available TPM 2.0 device"
  exit 0
fi

if [[ -L "$CREDENTIAL_DIR" ]]; then
  warn "$CREDENTIAL_DIR must not be a symlink"
  exit 0
fi
if ! install -d -m 0700 "$CREDENTIAL_DIR"; then
  warn "could not create $CREDENTIAL_DIR"
  exit 0
fi

credential_source="$CREDENTIAL_BLOB"
new_credential=0
if [[ -e "$CREDENTIAL_BLOB" || -L "$CREDENTIAL_BLOB" ]]; then
  if [[ ! -f "$CREDENTIAL_BLOB" || -L "$CREDENTIAL_BLOB" ]]; then
    warn "$CREDENTIAL_BLOB is not a regular file"
    exit 0
  fi
  if ! chmod 0600 "$CREDENTIAL_BLOB"; then
    warn "could not protect $CREDENTIAL_BLOB"
    exit 0
  fi
else
  encrypted_tmp="$(mktemp "$CREDENTIAL_DIR/.roboflow-master-key.cred.XXXXXX")"
  if [[ -z "$encrypted_tmp" ]] || ! rm -f -- "$encrypted_tmp"; then
    warn "could not allocate an encrypted credential path"
    exit 0
  fi

  # Stay TPM-bound without a PCR policy so routine OS updates do not invalidate the key.
  if ! dd if=/dev/urandom bs=32 count=1 status=none |
      systemd-creds encrypt \
        --name="$CREDENTIAL_NAME" \
        --with-key=tpm2 \
        --tpm2-device=auto \
        --tpm2-pcrs= \
        - "$encrypted_tmp"; then
    warn "TPM-backed master-key provisioning failed"
    exit 0
  fi
  if [[ ! -f "$encrypted_tmp" || -L "$encrypted_tmp" ]] ||
      ! chmod 0600 "$encrypted_tmp"; then
    warn "the provisioned credential is not a protected regular file"
    exit 0
  fi
  credential_source="$encrypted_tmp"
  new_credential=1
fi

runtime_tmp="$(mktemp "$RUNTIME_DIR/.roboflow-master-key.XXXXXX")"
if [[ -z "$runtime_tmp" ]]; then
  warn "could not allocate volatile key storage"
  exit 0
fi
if ! systemd-creds decrypt \
    --name="$CREDENTIAL_NAME" \
    "$credential_source" - >"$runtime_tmp"; then
  warn "the TPM-backed master key could not be decrypted; the existing credential was preserved"
  exit 0
fi

key_size="$(stat -c %s -- "$runtime_tmp" 2>/dev/null || true)"
if [[ "$key_size" != "32" ]]; then
  warn "the decrypted master key is not exactly 32 bytes"
  exit 0
fi

if [[ "$new_credential" -eq 1 ]]; then
  if [[ -e "$CREDENTIAL_BLOB" || -L "$CREDENTIAL_BLOB" ]]; then
    warn "an encrypted credential appeared concurrently; neither credential was overwritten"
    exit 0
  fi
  if ! mv -n -- "$encrypted_tmp" "$CREDENTIAL_BLOB"; then
    warn "could not persist the encrypted credential"
    exit 0
  fi
  if [[ -e "$encrypted_tmp" ]]; then
    warn "an encrypted credential appeared concurrently; neither credential was overwritten"
    exit 0
  fi
  encrypted_tmp=""
fi

if ! chmod 0400 "$runtime_tmp" ||
    ! mv -f -- "$runtime_tmp" "$RUNTIME_KEY"; then
  warn "could not publish the decrypted key in volatile storage"
  exit 0
fi
runtime_tmp=""

echo "Roboflow TPM-backed secure storage is available."
exit 0
