#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
MEDIAMTX_CONFIG="/etc/tilletia-app/mediamtx.yml"
LOCAL_MEDIAMTX_CONFIG="$APP_DIR/config/mediamtx.yml"
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
TIMEZONE_SYNC_SCRIPT="$SCRIPT_DIR/tilletia-app-timezone-sync.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH"
  exit 1
fi

if [[ -f "$MEDIAMTX_CONFIG" ]]; then
  export MEDIAMTX_CONFIG_PATH="$MEDIAMTX_CONFIG"
  export TILLETIA_DATA_ROOT="/var/lib/tilletia-app"
  export TILLETIA_CONFIG_ROOT="/etc/tilletia-app"
  export TILLETIA_SECRET_ROOT="$SYSTEM_SECRET_ROOT"
else
  echo "No system config found at $MEDIAMTX_CONFIG. Starting in standalone mode."
  STANDALONE_MODE=1
  export MEDIAMTX_CONFIG_PATH="$LOCAL_MEDIAMTX_CONFIG"
  export TILLETIA_DATA_ROOT="$LOCAL_DATA_ROOT"
  export TILLETIA_CONFIG_ROOT="$LOCAL_CONFIG_ROOT"
  export TILLETIA_SECRET_ROOT="$LOCAL_SECRET_ROOT"
fi

if [[ -L "$TILLETIA_SECRET_ROOT" ]]; then
  echo "Refusing symlinked runtime secrets directory: $TILLETIA_SECRET_ROOT"
  exit 1
fi
install -d -m 0700 "$TILLETIA_SECRET_ROOT"

runtime_secret_fs=""
if command -v findmnt >/dev/null 2>&1; then
  runtime_secret_fs="$(findmnt -n -o FSTYPE --target "$TILLETIA_SECRET_ROOT" 2>/dev/null || true)"
fi
if [[ "$STANDALONE_MODE" -eq 1 || "$runtime_secret_fs" != "ramfs" ]]; then
  if [[ "$STANDALONE_MODE" -eq 0 ]]; then
    echo "Roboflow secure storage unavailable: $TILLETIA_SECRET_ROOT is not mounted as ramfs."
  fi
  # Never accept a master key from standalone or non-volatile storage.
  rm -f -- "$TILLETIA_SECRET_ROOT/roboflow-master-key"
fi

if [[ ! -f "$MEDIAMTX_CONFIG_PATH" ]]; then
  echo "Missing mediamtx config: $MEDIAMTX_CONFIG_PATH"
  exit 1
fi

mkdir -p \
  "$TILLETIA_DATA_ROOT/model/ul" \
  "$TILLETIA_DATA_ROOT/model/rf" \
  "$TILLETIA_DATA_ROOT/output_hq" \
  "$TILLETIA_DATA_ROOT/runs"

"$TIMEZONE_SYNC_SCRIPT" --config-root "$TILLETIA_CONFIG_ROOT" --force

docker compose -f "$COMPOSE_FILE" up --build --remove-orphans -d

echo "Started services"
docker compose -f "$COMPOSE_FILE" ps

if command -v curl >/dev/null 2>&1; then
  echo "Waiting for web app health on http://127.0.0.1:8000/login ..."
  healthy=0
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/login >/dev/null 2>&1; then
      healthy=1
      break
    fi
    sleep 1
  done

  if [[ "$healthy" -ne 1 ]]; then
    echo "Web app is not reachable on port 8000 after startup."
    echo "Last logs from tilletia-app:"
    docker compose -f "$COMPOSE_FILE" logs --tail=120 tilletia-app || true
    exit 1
  fi

  echo "Web app is healthy: http://127.0.0.1:8000/"
fi

if [[ -t 1 ]]; then
  echo "Interactive terminal detected. Attaching to service logs..."
  exec docker compose -f "$COMPOSE_FILE" logs -f
fi
