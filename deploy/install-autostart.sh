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
SERVICE_TEMPLATE="$SCRIPT_DIR/tilletia-app.service"
SERVICE_PATH="/etc/systemd/system/tilletia-app.service"
ENV_DIR="/etc/tilletia-app"
MEDIAMTX_TEMPLATE="$APP_DIR/config/mediamtx.yml"
MEDIAMTX_CONFIG="$ENV_DIR/mediamtx.yml"
BASE_IMAGE="opencv-gst:latest"
APP_IMAGE="tilletia-app-web:latest"
BASE_DOCKERFILE="$APP_DIR/docker/Dockerfile.desktop"

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
  export TILLETIA_DATA_ROOT="${TILLETIA_DATA_ROOT:-$APP_DIR/data}"
  docker compose -f "$COMPOSE_FILE" down || true

  if systemctl list-unit-files | grep -q '^tilletia-app\.service'; then
    systemctl stop tilletia-app.service || true
    systemctl disable tilletia-app.service || true
  fi

  rm -f "$SERVICE_PATH"
  systemctl daemon-reload
  systemctl reset-failed
  echo "tilletia-app.service removed. Data/config were kept intact."
  exit 0
fi

if [[ ! -f "$SERVICE_TEMPLATE" ]]; then
  echo "Missing service template: $SERVICE_TEMPLATE"
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

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Building base image $BASE_IMAGE from $BASE_DOCKERFILE..."
  docker build -t "$BASE_IMAGE" -f "$BASE_DOCKERFILE" "$APP_DIR"
fi

echo "Building app image $APP_IMAGE..."
docker compose -f "$COMPOSE_FILE" build tilletia-app-web

sed "s|__DEPLOY_DIR__|$SCRIPT_DIR|g" "$SERVICE_TEMPLATE" > "$SERVICE_PATH"
chmod 0644 "$SERVICE_PATH"

systemctl daemon-reload
systemctl enable tilletia-app.service
systemctl restart tilletia-app.service

echo "Installed and started tilletia-app.service"
systemctl status --no-pager tilletia-app.service
