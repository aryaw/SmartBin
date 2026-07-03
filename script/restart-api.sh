#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "[API] Restarting backend services..."
"$SCRIPT_DIR/stop-api.sh"
"$SCRIPT_DIR/start-api.sh"
echo "[API] Restart complete."
