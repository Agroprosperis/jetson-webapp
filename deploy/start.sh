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

# Optional Caddy HTTPS on :9443 when TLS material + IDS hostname are present.
# HTTP :8000 always stays up (legacy mode).
TLS_DIR="${TILLETIA_TLS_DIR:-$TILLETIA_CONFIG_ROOT/tls}"
CADDYFILE_SRC="$APP_DIR/config/Caddyfile"
CADDYFILE_DST="${TILLETIA_CADDYFILE:-$TILLETIA_CONFIG_ROOT/Caddyfile}"
IDS_HOST_FILE="$TILLETIA_CONFIG_ROOT/ids-host"
ENABLE_TLS=0

if [[ -f "$TLS_DIR/fullchain.pem" && -f "$TLS_DIR/key.pem" ]]; then
  if [[ -n "${TILLETIA_IDS_HOST:-}" ]]; then
    :
  elif [[ -f "$IDS_HOST_FILE" ]]; then
    TILLETIA_IDS_HOST="$(tr -d '[:space:]' < "$IDS_HOST_FILE")"
  fi
  if [[ -n "${TILLETIA_IDS_HOST:-}" ]]; then
    if [[ -f "$CADDYFILE_SRC" ]]; then
      mkdir -p "$(dirname "$CADDYFILE_DST")"
      cp "$CADDYFILE_SRC" "$CADDYFILE_DST"
    fi
    if [[ -f "$CADDYFILE_DST" ]]; then
      ENABLE_TLS=1
      export TILLETIA_IDS_HOST
      export TILLETIA_CADDYFILE="$CADDYFILE_DST"
      export TILLETIA_TLS_DIR="$TLS_DIR"
      export COMPOSE_PROFILES=tls
    fi
  fi
fi

if [[ "$ENABLE_TLS" -ne 1 ]]; then
  echo "TLS/Caddy skipped (need $TLS_DIR/fullchain.pem, key.pem, and ids-host). HTTP :8000 only."
  unset COMPOSE_PROFILES || true
  docker compose -f "$COMPOSE_FILE" --profile tls stop tilletia-app-caddy >/dev/null 2>&1 || true
  docker rm -f tilletia-app-caddy >/dev/null 2>&1 || true
else
  echo "TLS/Caddy enabled for https://${TILLETIA_IDS_HOST}:9443 -> 127.0.0.1:8000"
fi

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

  if [[ "$ENABLE_TLS" -eq 1 ]]; then
    echo "Checking HTTPS front door on :9443 ..."
    https_ok=0
    for _ in $(seq 1 20); do
      if curl -fsSk --max-time 2 \
        --resolve "${TILLETIA_IDS_HOST}:9443:127.0.0.1" \
        "https://${TILLETIA_IDS_HOST}:9443/login" >/dev/null 2>&1; then
        https_ok=1
        break
      fi
      sleep 1
    done
    if [[ "$https_ok" -eq 1 ]]; then
      echo "HTTPS is healthy: https://${TILLETIA_IDS_HOST}:9443/"
    else
      echo "WARNING: HTTPS :9443 is not reachable yet. Check: docker logs tilletia-app-caddy"
      docker compose -f "$COMPOSE_FILE" logs --tail=80 tilletia-app-caddy || true
    fi
  fi
fi

if [[ -t 1 ]]; then
  echo "Interactive terminal detected. Attaching to service logs..."
  exec docker compose -f "$COMPOSE_FILE" logs -f
fi
