#!/bin/bash
set -e

LOG_DIR="/app/log/$(date +%Y-%m-%d)"
mkdir -p "$LOG_DIR"

exec > >(tee -a "$LOG_DIR/container.log") 2>&1

echo "[LOG] Container started at $(date)"
echo "[LOG] Log dir: $LOG_DIR"

exec "$@"
