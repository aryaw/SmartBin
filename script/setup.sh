#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== SmartBin One-Time Setup ==="

echo ""
echo "[1/3] Creating Python virtual environment..."
cd "$PROJECT_DIR/backend"
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
echo "  Done. (.venv created)"

echo ""
echo "[2/3] Building Docker images..."
cd "$PROJECT_DIR"
sudo docker compose build backend
sudo docker compose build frontend
echo "  Done."

echo ""
echo "[3/3] Creating log directories..."
mkdir -p "$PROJECT_DIR/backend/log" "$PROJECT_DIR/frontend/log"
echo "  Done."

echo ""
echo "=== Setup complete ==="
echo "Run:  ./script/start-api.sh   (Terminal 1)"
echo "      ./script/start-fe.sh    (Terminal 2)"
echo ""
echo "Or without Docker:"
echo "  cd backend  && .venv/bin/uvicorn app.main:app --reload"
echo "  cd frontend && npm run dev"
