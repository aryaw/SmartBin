#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[FE] Stopping frontend..."
cd "$PROJECT_DIR"
sudo docker compose stop frontend 2>/dev/null || true
echo "[FE] Frontend stopped."
