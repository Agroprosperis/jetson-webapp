#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
MEDIAMTX_CONFIG="/etc/tilletia-app/mediamtx.yml"
LOCAL_MEDIAMTX_CONFIG="$APP_DIR/config/mediamtx.yml"
LOCAL_DATA_ROOT="$APP_DIR/data"
BASE_IMAGE="opencv-gst:latest"
BASE_DOCKERFILE="$APP_DIR/docker/Dockerfile.desktop"

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is not installed or not in PATH"
  exit 1
fi

if [[ -f "$MEDIAMTX_CONFIG" ]]; then
  export MEDIAMTX_CONFIG_PATH="$MEDIAMTX_CONFIG"
  export TILLETIA_DATA_ROOT="/var/lib/tilletia-app"
else
  echo "No system config found at $MEDIAMTX_CONFIG. Starting in standalone mode."
  export MEDIAMTX_CONFIG_PATH="$LOCAL_MEDIAMTX_CONFIG"
  export TILLETIA_DATA_ROOT="$LOCAL_DATA_ROOT"
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

if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  echo "Base image $BASE_IMAGE not found, building from $BASE_DOCKERFILE..."
  docker build -t "$BASE_IMAGE" -f "$BASE_DOCKERFILE" "$APP_DIR"
fi

docker compose -f "$COMPOSE_FILE" up --build -d

echo "Started services"
docker compose -f "$COMPOSE_FILE" ps

if command -v curl >/dev/null 2>&1; then
  echo "Waiting for web app health on http://127.0.0.1:8000/api/config ..."
  healthy=0
  for _ in $(seq 1 30); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/api/config >/dev/null 2>&1; then
      healthy=1
      break
    fi
    sleep 1
  done

  if [[ "$healthy" -ne 1 ]]; then
    echo "Web app is not reachable on port 8000 after startup."
    echo "Last logs from tilletia-app-web:"
    docker compose -f "$COMPOSE_FILE" logs --tail=120 tilletia-app-web || true
    exit 1
  fi

  echo "Web app is healthy: http://127.0.0.1:8000/"
fi

if [[ -t 1 ]]; then
  echo "Interactive terminal detected. Attaching to service logs..."
  exec docker compose -f "$COMPOSE_FILE" logs -f
fi
