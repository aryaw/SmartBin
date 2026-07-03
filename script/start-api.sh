#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"
IMAGE_EXISTS=$(sudo docker images -q smartbin-backend 2>/dev/null)
if [ -z "$IMAGE_EXISTS" ]; then
  echo "[API] Image not found. Building first..."
  sudo docker compose build backend
fi
echo "[API] Freeing port 8000..."
sudo fuser -k 8000/tcp 2>/dev/null || true
sleep 1
echo "[API] Starting backend..."
sudo docker compose up -d backend || {
  echo "[API] Docker failed, starting directly..."
  cd "$PROJECT_DIR/backend"
  nohup .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > /tmp/backend.log 2>&1 &
  disown
  sleep 3
}
echo "[API] Backend ready at http://localhost:8000"
