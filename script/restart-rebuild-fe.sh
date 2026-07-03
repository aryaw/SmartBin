#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[FE] Rebuilding frontend..."
cd "$PROJECT_DIR"
sudo docker compose build frontend
"$SCRIPT_DIR/stop-fe.sh"
"$SCRIPT_DIR/start-fe.sh"
echo "[FE] Rebuild complete."
