#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOCAL_DATA_ROOT="$APP_DIR/data"
LOCAL_CONFIG_ROOT="$LOCAL_DATA_ROOT/config"

if [[ -f "/etc/tilletia-app/mediamtx.yml" ]]; then
  export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-/var/lib/tilletia-app}"
  export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-/etc/tilletia-app}"
else
  export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-$LOCAL_DATA_ROOT}"
  export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-$LOCAL_CONFIG_ROOT}"
fi

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

docker compose -f "$COMPOSE_FILE" down --remove-orphans
