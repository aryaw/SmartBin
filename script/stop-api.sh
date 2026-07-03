#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[API] Stopping backend..."
cd "$PROJECT_DIR"
sudo docker compose stop backend 2>/dev/null || true
echo "[API] Backend stopped."
