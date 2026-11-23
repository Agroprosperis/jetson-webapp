#!/usr/bin/env bash
# Convert Agroprosperis local PFX into PEM files for Caddy (HTTPS :9443).
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo."
  exit 1
fi

TLS_DIR="${TILLETIA_TLS_DIR:-/etc/tilletia-app/tls}"
PFX_PATH="${TILLETIA_PFX_PATH:-$TLS_DIR/agroprosperis_local.pfx}"
PASSWORD_FILE="${TILLETIA_PFX_PASSWORD_FILE:-$TLS_DIR/pfx.password}"
FULLCHAIN_PATH="$TLS_DIR/fullchain.pem"
KEY_PATH="$TLS_DIR/key.pem"

if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required."
  exit 1
fi

install -d -m 0700 "$TLS_DIR"

if [[ ! -f "$PFX_PATH" ]]; then
  echo "Missing PFX: $PFX_PATH"
  echo "Copy agroprosperis_local.pfx there (from ap_local_cert.7z), then re-run."
  exit 1
fi

if [[ -n "${TILLETIA_PFX_PASSWORD:-}" ]]; then
  PFX_PASSWORD="$TILLETIA_PFX_PASSWORD"
elif [[ -f "$PASSWORD_FILE" ]]; then
  PFX_PASSWORD="$(tr -d '\r\n' < "$PASSWORD_FILE")"
else
  echo "Set TILLETIA_PFX_PASSWORD or create $PASSWORD_FILE (one line, PFX password)."
  exit 1
fi

if [[ -z "$PFX_PASSWORD" ]]; then
  echo "PFX password is empty."
  exit 1
fi

umask 077
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Leaf + intermediates as chain; private key separate.
openssl pkcs12 -in "$PFX_PATH" -passin "pass:$PFX_PASSWORD" -nokeys -clcerts \
  -out "$TMP_DIR/leaf.pem" 2>/dev/null
openssl pkcs12 -in "$PFX_PATH" -passin "pass:$PFX_PASSWORD" -nokeys -cacerts \
  -out "$TMP_DIR/ca.pem" 2>/dev/null || true
openssl pkcs12 -in "$PFX_PATH" -passin "pass:$PFX_PASSWORD" -nocerts -nodes \
  -out "$TMP_DIR/key.pem" 2>/dev/null

{
  cat "$TMP_DIR/leaf.pem"
  if [[ -s "$TMP_DIR/ca.pem" ]]; then
    cat "$TMP_DIR/ca.pem"
  fi
} > "$TMP_DIR/fullchain.pem"

install -m 0600 "$TMP_DIR/fullchain.pem" "$FULLCHAIN_PATH"
install -m 0600 "$TMP_DIR/key.pem" "$KEY_PATH"

# Keep password file mode tight if present / just created via env.
if [[ -n "${TILLETIA_PFX_PASSWORD:-}" && ! -f "$PASSWORD_FILE" ]]; then
  printf '%s\n' "$PFX_PASSWORD" > "$PASSWORD_FILE"
  chmod 0600 "$PASSWORD_FILE"
elif [[ -f "$PASSWORD_FILE" ]]; then
  chmod 0600 "$PASSWORD_FILE"
fi
chmod 0600 "$PFX_PATH" 2>/dev/null || true

openssl x509 -in "$FULLCHAIN_PATH" -noout -subject -dates >/dev/null

echo "Wrote:"
echo "  $FULLCHAIN_PATH"
echo "  $KEY_PATH"
echo "TLS material is ready for Caddy on :9443."
