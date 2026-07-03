#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
IMAGE_EXISTS=$(sudo docker images -q smartbin-frontend 2>/dev/null)
if [ -z "$IMAGE_EXISTS" ]; then
  echo "[FE] Image not found. Building first..."
  sudo docker compose build frontend
fi
echo "[FE] Freeing port 3000..."
sudo fuser -k 3000/tcp 2>/dev/null || true
sleep 1
echo "[FE] Starting frontend..."
sudo docker compose up -d frontend
echo "[FE] Frontend ready at http://localhost:3000"
