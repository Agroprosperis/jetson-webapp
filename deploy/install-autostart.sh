#!/usr/bin/env bash
set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this script with sudo."
  exit 1
fi

MODE="${1:-install}"
if [[ "$MODE" != "install" && "$MODE" != "--uninstall" ]]; then
  echo "Usage: sudo ./install-autostart.sh [--uninstall]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
SYSTEMD_DIR="/etc/systemd/system"
APP_SERVICE_TEMPLATE="$SCRIPT_DIR/tilletia-app.service"
TIMEZONE_SYNC_SERVICE_TEMPLATE="$SCRIPT_DIR/tilletia-app-timezone-sync.service"
TIMEZONE_SYNC_PATH_TEMPLATE="$SCRIPT_DIR/tilletia-app-timezone-sync.path"
TIMEZONE_SYNC_SCRIPT="$SCRIPT_DIR/tilletia-app-timezone-sync.sh"
APP_SERVICE_PATH="$SYSTEMD_DIR/tilletia-app.service"
TIMEZONE_SYNC_SERVICE_PATH="$SYSTEMD_DIR/tilletia-app-timezone-sync.service"
TIMEZONE_SYNC_PATH_UNIT="$SYSTEMD_DIR/tilletia-app-timezone-sync.path"
ENV_DIR="/etc/tilletia-app"
MEDIAMTX_TEMPLATE="$APP_DIR/config/mediamtx.yml"
MEDIAMTX_CONFIG="$ENV_DIR/mediamtx.yml"
APP_IMAGE="tilletia-app:latest"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH"
  exit 1
fi

if [[ "$MODE" == "--uninstall" ]]; then
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
  if [[ -f "/etc/tilletia-app/mediamtx.yml" ]]; then
    export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-/var/lib/tilletia-app}"
    export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-/etc/tilletia-app}"
  else
    export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-$APP_DIR/data}"
    export TILLETIA_CONFIG_ROOT="${TILLETIA_CONFIG_ROOT:-$APP_DIR/data/config}"
  fi
  docker compose -f "$COMPOSE_FILE" down --remove-orphans || true

  systemctl disable --now tilletia-app-timezone-sync.path >/dev/null 2>&1 || true

  if systemctl list-unit-files | grep -q '^tilletia-app\.service'; then
    systemctl stop tilletia-app.service || true
    systemctl disable tilletia-app.service || true
  fi
  systemctl stop tilletia-app-timezone-sync.service >/dev/null 2>&1 || true

  rm -f "$APP_SERVICE_PATH"
  rm -f "$TIMEZONE_SYNC_SERVICE_PATH"
  rm -f "$TIMEZONE_SYNC_PATH_UNIT"
  systemctl daemon-reload
  systemctl reset-failed
  echo "tilletia-app.service removed. Data/config were kept intact."
  exit 0
fi

if [[ ! -f "$APP_SERVICE_TEMPLATE" ]]; then
  echo "Missing service template: $APP_SERVICE_TEMPLATE"
  exit 1
fi

if [[ ! -f "$TIMEZONE_SYNC_SERVICE_TEMPLATE" ]]; then
  echo "Missing timezone sync service template: $TIMEZONE_SYNC_SERVICE_TEMPLATE"
  exit 1
fi

if [[ ! -f "$TIMEZONE_SYNC_PATH_TEMPLATE" ]]; then
  echo "Missing timezone sync path template: $TIMEZONE_SYNC_PATH_TEMPLATE"
  exit 1
fi

if [[ ! -x "$TIMEZONE_SYNC_SCRIPT" ]]; then
  echo "Missing or non-executable timezone sync script: $TIMEZONE_SYNC_SCRIPT"
  exit 1
fi

if [[ ! -f "$MEDIAMTX_TEMPLATE" ]]; then
  echo "Missing mediamtx template: $MEDIAMTX_TEMPLATE"
  exit 1
fi

# Keep repo runtime layout complete even if only model/config are committed.
mkdir -p \
  "$APP_DIR/data/model/ul" \
  "$APP_DIR/data/model/rf" \
  "$APP_DIR/data/output_hq" \
  "$APP_DIR/data/runs"

mkdir -p "$ENV_DIR"
"$TIMEZONE_SYNC_SCRIPT" --config-root "$ENV_DIR" --force

if [[ ! -f "$MEDIAMTX_CONFIG" ]]; then
  cp "$MEDIAMTX_TEMPLATE" "$MEDIAMTX_CONFIG"
  chmod 0644 "$MEDIAMTX_CONFIG"
fi

mkdir -p \
  /var/lib/tilletia-app/model/ul \
  /var/lib/tilletia-app/model/rf \
  /var/lib/tilletia-app/output_hq \
  /var/lib/tilletia-app/runs

# Seed persisted runtime data from repository layout if present.
if [[ -d "$APP_DIR/data/model" ]]; then
  cp -an "$APP_DIR/data/model/." /var/lib/tilletia-app/model/
fi
if [[ -d "$APP_DIR/data/output_hq" ]]; then
  cp -an "$APP_DIR/data/output_hq/." /var/lib/tilletia-app/output_hq/
fi
if [[ -d "$APP_DIR/data/runs" ]]; then
  cp -an "$APP_DIR/data/runs/." /var/lib/tilletia-app/runs/
fi

echo "Building app image $APP_IMAGE..."
docker compose -f "$COMPOSE_FILE" build tilletia-app

sed "s|__DEPLOY_DIR__|$SCRIPT_DIR|g" "$APP_SERVICE_TEMPLATE" > "$APP_SERVICE_PATH"
sed "s|__DEPLOY_DIR__|$SCRIPT_DIR|g" "$TIMEZONE_SYNC_SERVICE_TEMPLATE" > "$TIMEZONE_SYNC_SERVICE_PATH"
cp "$TIMEZONE_SYNC_PATH_TEMPLATE" "$TIMEZONE_SYNC_PATH_UNIT"
chmod 0644 "$APP_SERVICE_PATH"
chmod 0644 "$TIMEZONE_SYNC_SERVICE_PATH" "$TIMEZONE_SYNC_PATH_UNIT"

systemctl daemon-reload
systemctl enable tilletia-app.service
systemctl restart tilletia-app.service
systemctl enable --now tilletia-app-timezone-sync.path

echo "Installed and started tilletia-app.service"
systemctl status --no-pager tilletia-app.service
systemctl status --no-pager tilletia-app-timezone-sync.path
