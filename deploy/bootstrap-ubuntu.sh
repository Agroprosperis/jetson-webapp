#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Agroprosperis/jetson-webapp.git}"
INSTALL_DIR="${INSTALL_DIR:-/opt/tilletia-app}"
SKIP_NVIDIA="${SKIP_NVIDIA:-0}"
SKIP_INSTALL="${SKIP_INSTALL:-0}"

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run with sudo."
  exit 1
fi

if [[ ! -f /etc/os-release ]]; then
  echo "Unsupported OS: missing /etc/os-release"
  exit 1
fi

# shellcheck disable=SC1091
source /etc/os-release
if [[ "${ID:-}" != "ubuntu" ]]; then
  echo "This script targets Ubuntu. Detected: ${ID:-unknown}"
  exit 1
fi

UBUNTU_CODENAME="${VERSION_CODENAME:-}"
UBUNTU_VERSION="${VERSION_ID:-}"
REQUIRED_CODENAME="${REQUIRED_CODENAME:-noble}"

if [[ -z "$UBUNTU_CODENAME" ]]; then
  echo "Could not detect Ubuntu codename from /etc/os-release."
  exit 1
fi

if [[ "$UBUNTU_CODENAME" != "$REQUIRED_CODENAME" ]]; then
  echo "Expected Ubuntu codename '$REQUIRED_CODENAME', got '$UBUNTU_CODENAME' ($PRETTY_NAME)."
  exit 1
fi

echo "Target OS: Ubuntu ${UBUNTU_VERSION} (${UBUNTU_CODENAME})"

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y ca-certificates curl git gnupg lsb-release

if ! command -v docker >/dev/null 2>&1; then
  install -m 0755 -d /etc/apt/keyrings
  curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --batch --yes --dearmor -o /etc/apt/keyrings/docker.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu ${UBUNTU_CODENAME} stable" \
    > /etc/apt/sources.list.d/docker.list
  apt-get update
  apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
  systemctl enable --now docker
else
  echo "Docker already installed."
fi

if [[ "$SKIP_NVIDIA" != "1" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
      install -m 0755 -d /etc/apt/keyrings
      curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --batch --yes --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit.gpg
      curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
      apt-get update
      apt-get install -y nvidia-container-toolkit
      nvidia-ctk runtime configure --runtime=docker
      systemctl restart docker
    else
      echo "NVIDIA Container Toolkit already installed."
    fi
  else
    echo "WARNING: nvidia-smi not found. Install NVIDIA drivers first, then re-run bootstrap."
  fi
else
  echo "Skipping NVIDIA Container Toolkit setup (SKIP_NVIDIA=1)."
fi

if [[ ! -d "$INSTALL_DIR/.git" ]]; then
  git clone "$REPO_URL" "$INSTALL_DIR"
else
  git -C "$INSTALL_DIR" fetch --all --prune
  git -C "$INSTALL_DIR" pull --ff-only
fi

if [[ "$SKIP_INSTALL" == "1" ]]; then
  echo "Repository ready at $INSTALL_DIR (SKIP_INSTALL=1)."
  exit 0
fi

cd "$INSTALL_DIR/deploy"
./install-autostart.sh

if command -v curl >/dev/null 2>&1; then
  for _ in $(seq 1 60); do
    if curl -fsS --max-time 2 http://127.0.0.1:8000/login >/dev/null 2>&1; then
      echo "OK: http://127.0.0.1:8000/login"
      exit 0
    fi
    sleep 2
  done
  echo "WARNING: service started but /login is not reachable yet."
  systemctl status --no-pager tilletia-app.service || true
  exit 1
fi

echo "Installed. Open http://<host-ip>:8000/login"
