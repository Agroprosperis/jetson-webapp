#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DATA_ROOT="$APP_DIR/data"
LOCAL_CONFIG_ROOT="$LOCAL_DATA_ROOT/config"
SYSTEM_SECRET_ROOT="/run/tilletia/secrets"
if [[ -n "${XDG_RUNTIME_DIR:-}" ]]; then
  LOCAL_RUNTIME_ROOT="$XDG_RUNTIME_DIR/tilletia-app"
else
  LOCAL_RUNTIME_ROOT="/tmp/tilletia-app-runtime-${UID}"
fi
LOCAL_SECRET_ROOT="$LOCAL_RUNTIME_ROOT/secrets"
STANDALONE_MODE=0

if [[ -f "/etc/tilletia-app/mediamtx.yml" ]]; then
  export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-/var/lib/tilletia-app}"
  export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-/etc/tilletia-app}"
  export TILLETIA_SECRET_ROOT="$SYSTEM_SECRET_ROOT"
else
  STANDALONE_MODE=1
  export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-$LOCAL_DATA_ROOT}"
  export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-$LOCAL_CONFIG_ROOT}"
  export TILLETIA_SECRET_ROOT="$LOCAL_SECRET_ROOT"
fi

if [[ -L "$TILLETIA_SECRET_ROOT" ]]; then
  echo "Refusing symlinked runtime secrets directory: $TILLETIA_SECRET_ROOT"
  exit 1
fi
install -d -m 0700 "$TILLETIA_SECRET_ROOT"

mkdir -p \
  "$TILLETIA_DATA_ROOT/model/ul" \
  "$TILLETIA_DATA_ROOT/model/rf" \
  "$TILLETIA_DATA_ROOT/output_hq" \
  "$TILLETIA_DATA_ROOT/runs"

if [[ -f "/etc/tilletia-app/mediamtx.yml" ]]; then
  export MEDIAMTX_CONFIG_PATH="/etc/tilletia-app/mediamtx.yml"
elif [[ -f "$APP_DIR/config/mediamtx.yml" ]]; then
  export MEDIAMTX_CONFIG_PATH="$APP_DIR/config/mediamtx.yml"
else
  export MEDIAMTX_CONFIG_PATH="/tmp/tilletia-app-mediamtx.yml"
  cat > "$MEDIAMTX_CONFIG_PATH" <<'EOF'
logLevel: info
paths: {}
EOF
fi

cleanup_runtime_secrets() {
  rm -f -- "$TILLETIA_SECRET_ROOT/roboflow-master-key" || true
  if [[ "$STANDALONE_MODE" -eq 1 ]]; then
    rmdir "$TILLETIA_SECRET_ROOT" >/dev/null 2>&1 || true
    rmdir "$LOCAL_RUNTIME_ROOT" >/dev/null 2>&1 || true
  fi
}

trap cleanup_runtime_secrets EXIT
docker compose -f "$COMPOSE_FILE" down --remove-orphans
trap - EXIT
cleanup_runtime_secrets
