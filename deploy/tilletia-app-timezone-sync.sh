#!/usr/bin/env bash
set -euo pipefail

CONFIG_ROOT="/etc/tilletia-app"
FORCE=0

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --config-root)
      CONFIG_ROOT="${2:?Missing value for --config-root}"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    *)
      echo "Usage: $0 [--config-root PATH] [--force]" >&2
      exit 2
      ;;
  esac
done

STATE_FILE="$CONFIG_ROOT/timezone.state"
LOCALTIME_COPY="$CONFIG_ROOT/localtime"
TIMEZONE_COPY="$CONFIG_ROOT/timezone"

snapshot_timezone_state() {
  if [[ -e /etc/localtime ]]; then
    printf 'localtime-target='
    readlink -f /etc/localtime 2>/dev/null || true
    stat -Lc 'localtime-stat=%d:%i:%s:%Y' /etc/localtime 2>/dev/null || true
    sha256sum /etc/localtime 2>/dev/null || true
  fi

  if [[ -e /etc/timezone ]]; then
    stat -Lc 'timezone-stat=%d:%i:%s:%Y' /etc/timezone 2>/dev/null || true
    sha256sum /etc/timezone 2>/dev/null || true
  fi
}

copy_in_place() {
  local source_path="$1"
  local target_path="$2"

  if [[ ! -e "$source_path" ]]; then
    echo "Missing host timezone source: $source_path" >&2
    exit 1
  fi

  if [[ -e "$target_path" ]]; then
    if [[ ! -f "$target_path" || -L "$target_path" ]]; then
      echo "Refusing to overwrite non-regular timezone copy: $target_path" >&2
      exit 1
    fi
    cp "$source_path" "$target_path"
  else
    install -m 0644 "$source_path" "$target_path"
  fi
}

mkdir -p "$CONFIG_ROOT"

current_state="$(snapshot_timezone_state)"
previous_state="$(cat "$STATE_FILE" 2>/dev/null || true)"

if [[ "$FORCE" -eq 0 && "$current_state" == "$previous_state" && -f "$LOCALTIME_COPY" && -f "$TIMEZONE_COPY" ]]; then
  exit 0
fi

copy_in_place /etc/localtime "$LOCALTIME_COPY"
copy_in_place /etc/timezone "$TIMEZONE_COPY"
printf '%s\n' "$current_state" > "$STATE_FILE"

echo "Synced host timezone files into $CONFIG_ROOT."
