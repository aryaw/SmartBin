# Docker & Infrastructure PRD - SmartBin

---

## 1. Environment

| Resource | Spec |
|----------|------|
| OS | Ubuntu 25.10 (x86_64) |
| GPU | NVIDIA (VRAM limit 12GB) |
| RAM | 30 GB |
| Docker | 29.x |
| CUDA Driver | 595.x |
| NVIDIA Runtime | nvidia-container-toolkit (GPU passthrough) |

---

## 2. Build Strategy

**Host `.venv` (--copies) → copy into Docker → zero pip download.**

Key: `.venv` created with `--copies` flag so Python binary is a real file (no symlinks to host pyenv path). Symlinks resolved to real files in script. Shebangs rewritten for container path.

```
Host:
  pyenv Python 3.12.9 → .venv/ (--copies, portable)
  restart-rebuild-api.sh:
    1. Clean dataset dirs (preserve raw/)
    2. Sync .venv from requirements.txt
    3. Verify every package installed
    4. Build Docker

Docker build:
  COPY .venv .venv          # standalone Python 3.12 + all deps
  ENV PATH=/app/.venv/bin   # uses .venv's own pip
  pip install -r req.txt    # finds everything → no download
  --mount=type=cache        # fallback cache

Result: rebuild ~2s
```

---

## 3. Dockerfile

```dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04
WORKDIR /app
RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg libgl1 \
    && rm -rf /var/lib/apt/lists/*
COPY .venv .venv
ENV PATH=/app/.venv/bin:$PATH
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --break-system-packages -r requirements.txt
COPY . .
RUN mkdir -p uploads log static/result models runs && chmod +x log-wrapper.sh
EXPOSE 8000
ENTRYPOINT ["/app/log-wrapper.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 4. Docker Compose

```yaml
services:
  backend:
    build: ./backend
    container_name: smartbin-backend
    ports: ["8000:8000"]
    env_file: ./backend/.env
    environment:
      - DB_HOST=host.docker.internal
      - API_BASE=http://backend:8000
    extra_hosts:
      - "host.docker.internal:host-gateway"
    volumes:
      - ./backend/uploads:/app/uploads
      - ./backend/log:/app/log
      - ./backend/static:/app/static
      - ./backend/models:/app/models
      - ./backend/dataset:/app/dataset
      - ./datasource:/datasource
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, count: 1, capabilities: [gpu]}]
    restart: unless-stopped

  frontend:
    build: ./frontend
    container_name: smartbin-frontend
    ports: ["3000:3000"]
    environment:
      - API_BASE=http://backend:8000
    volumes:
      - ./frontend/log:/app/log
    depends_on: [backend]
    restart: unless-stopped
```

---

## 5. Volumes

| Host | Container | Purpose |
|------|-----------|---------|
| `./backend/uploads` | `/app/uploads` | Temp upload files (deleted after processing) |
| `./backend/log` | `/app/log` | Log-wrapper output |
| `./backend/static` | `/app/static` | Annotated results |
| `./backend/models` | `/app/models` | YOLO weights (`best.pt`) |
| `./backend/dataset` | `/app/dataset` | Image splits (raw/train/val/test) |
| `./datasource` | `/datasource` | TACO annotations JSON + CSV |
| `./frontend/log` | `/app/log` | Frontend logs |

---

## 6. Restart Scripts

| Script | Action |
|--------|--------|
| `restart-rebuild-api.sh` | Clean dataset (preserve `raw/`) → sync `.venv` (--copies) → resolve symlinks → fix shebangs → verify packages → build Docker → stop → start |
| `restart-rebuild-fe.sh` | Build frontend Docker → stop → start |
| `start-api.sh` | Build image if missing → free port 8000 → `docker compose up -d backend` (fallback direct uvicorn) |
| `start-fe.sh` | Build image if missing → free port 3000 → `docker compose up -d frontend` |
| `stop-api.sh` | `docker compose stop backend` |
| `stop-fe.sh` | `docker compose stop frontend` |
| `setup.sh` | One-time: create `.venv` → install deps → build Docker images → create log dirs |
