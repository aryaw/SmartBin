#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "[API] Cleaning dataset (preserving raw)..."
sudo find "$PROJECT_DIR/backend/dataset" -mindepth 1 -maxdepth 1 ! -name 'raw' -exec rm -rf {} +
mkdir -p "$PROJECT_DIR/backend/dataset/raw"
echo "[API] Dataset cleaned"

echo "[API] Checking host .venv matches requirements.txt..."
cd "$PROJECT_DIR/backend"
MARKER=".venv/.requirements-checksum"
REQ_HASH=$(md5sum requirements.txt | cut -d' ' -f1)
PY312=""
for p in "$HOME/.pyenv/versions/3.12.9/bin/python3.12" "/usr/bin/python3.12" "/usr/local/bin/python3.12"; do
  [ -x "$p" ] && PY312="$p" && break
done
if [ -z "$PY312" ]; then
  echo "  Python 3.12 not found, skipping .venv sync"
elif [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$REQ_HASH" ]; then
  echo "  .venv up to date, skipping install"
else
  echo "  Syncing .venv..."
  "$PY312" -m venv .venv
  .venv/bin/pip install -r requirements.txt -q
  echo "  Resolving symlinks for Docker compat..."
  .venv/bin/python3.12 -c "
import os, sys
venv = '.venv'
for root, dirs, files in os.walk(venv):
    for f in files:
        path = os.path.join(root, f)
        if os.path.islink(path):
            target = os.readlink(path)
            if not target.startswith('/'):
                target = os.path.join(os.path.dirname(path), target)
            real = os.path.realpath(path)
            os.remove(path)
            os.system(f'cp -L {real} {path}')
    for d in dirs:
        path = os.path.join(root, d)
        if os.path.islink(path):
            real = os.path.realpath(path)
            os.remove(path)
            os.system(f'mkdir -p {path}')
            os.system(f'cp -rL {real}/. {path}/')
"
  echo "  Fixing pip shebangs for Docker..."
  find .venv/bin -type f -exec sed -i 's|^#!'"$PWD"'/backend/\.venv/bin/python3\.12|#!/app/.venv/bin/python3.12|' {} \;
  echo "$REQ_HASH" > "$MARKER"
fi

echo "[API] Verifying .venv has all required packages..."
VENV_OK=true
while IFS= read -r pkg; do
  [ -z "$pkg" ] && continue
  pkg_name=$(echo "$pkg" | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1 | cut -d'~' -f1 | xargs)
  [ -z "$pkg_name" ] && continue
  if ! .venv/bin/pip show "$pkg_name" >/dev/null 2>&1; then
    echo "  MISSING: $pkg_name — installing..."
    .venv/bin/pip install "$pkg" -q
    VENV_OK=false
  fi
done < requirements.txt
if [ "$VENV_OK" = true ]; then
  echo "  All packages verified"
fi

if [ -f "$PROJECT_DIR/backend/.env" ]; then
  set -a; source "$PROJECT_DIR/backend/.env"; set +a
fi
USE_DOCKER="${USE_DOCKER:-true}"

if [ "$USE_DOCKER" = "true" ]; then
  echo "[API] Stopping old backend container first..."
  sudo docker compose stop backend 2>/dev/null || true
  sudo docker compose rm -f backend 2>/dev/null || true
  echo "[API] Killing anything on port 8000..."
  sudo fuser -k 8000/tcp 2>/dev/null || true
  sleep 2
  echo "[API] Building Docker image..."
  cd "$PROJECT_DIR"
  sudo docker compose build backend 2>&1 || echo "[API] Build warning (non-fatal)"
  echo "[API] Starting backend via Docker..."
  if sudo docker compose up -d backend; then
    echo "[API] Waiting for backend to be ready..."
    for i in 1 2 3 4 5; do
      sleep 3
      if curl -s --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
        echo "[API] Docker backend running"
        break
      fi
      echo "[API] Attempt $i/5 - not ready yet"
    done
    if ! curl -s --max-time 3 http://localhost:8000/health >/dev/null 2>&1; then
      echo "[API] Backend not responding after 5 attempts, restarting..."
      sudo docker compose restart backend
      sleep 5
    fi
  else
    echo "[API] Docker start failed, starting directly..."
    cd "$PROJECT_DIR/backend"
    setsid .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > "$PROJECT_DIR/backend/log/$(date +%d-%m-%Y)-backend.log" 2>&1 &
    sleep 4
  fi
else
  echo "[API] USE_DOCKER=false — killing old process on 8000..."
  sudo fuser -k 8000/tcp 2>/dev/null || true
  sleep 2
  echo "[API] Starting directly..."
  cd "$PROJECT_DIR/backend"
  setsid .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 > "$PROJECT_DIR/backend/log/$(date +%d-%m-%Y)-backend.log" 2>&1 &
  sleep 4
fi

curl -s --max-time 3 http://localhost:8000/health >/dev/null 2>&1 && echo "[API] Backend ready on port 8000" || echo "[API] WARNING: Backend may not be running"
