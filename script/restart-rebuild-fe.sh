#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "$PROJECT_DIR/frontend/.env" ]; then
  set -a; source "$PROJECT_DIR/frontend/.env"; set +a
fi
USE_DOCKER="${USE_DOCKER:-true}"

if [ "$USE_DOCKER" = "true" ]; then
  echo "[FE] Rebuilding frontend (Docker)..."
  cd "$PROJECT_DIR"
  sudo docker compose build frontend
  "$SCRIPT_DIR/stop-fe.sh"
  "$SCRIPT_DIR/start-fe.sh"
  echo "[FE] Docker rebuild complete."
else
  echo "[FE] USE_DOCKER=false — running directly..."
  "$SCRIPT_DIR/stop-fe.sh" || true
  echo "[FE] Freeing port 3000..."
  sudo fuser -k 3000/tcp 2>/dev/null || true
  sleep 1
  cd "$PROJECT_DIR/frontend"
  if [ ! -d "node_modules" ]; then
    echo "[FE] Installing dependencies..."
    npm install
  fi
  echo "[FE] Starting Nuxt dev server..."
  setsid npm run dev > "$PROJECT_DIR/frontend/log/$(date +%d-%m-%Y)-frontend.log" 2>&1 &
  sleep 4
  echo "[FE] Frontend starting at http://localhost:3000"
fi
