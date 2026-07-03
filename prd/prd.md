# SmartBin - Product Requirements Document

**Version:** 2.0
**Platform:** Web Application
**Stack:** FastAPI + Nuxt 3 + YOLO26n

---

## Dataset

Source: TACO (Trash Annotations in Context), 1500 images, 4784 annotations.

Class mapping: Organik (ID 0) = Food waste only (TACO category 25). Non-Organik (ID 1) = all other 59 categories.

Split: 70/15/15 from `raw/` via `POST /api/dataset/split`. Split endpoint self-cleans existing train/val/test before re-splitting. `raw/` never deleted by split or rebuild script.

---

## Training

Model: YOLO26n (`yolo26n.pt`), batch=16, imgsz=640, augment=mosaic+HSV+geometric+erasing.
GPU VRAM limit: 12GB. FP16 inference. Cache clear every 16 images.

CLI: `python train.py --model yolo26n.pt --data data.yaml --epochs 100 --batch 16 --imgsz 640`
Grid search: `--grid-search 10 20 40`

---

## Backend

FastAPI port 8000. 13+ REST endpoints across 4 routers: health, detect, annotation, datasource.

Lifespan: GPU init + model load on startup. Model: `models/best.pt`.

Build strategy: host `.venv` (Python 3.12 `--copies`) copied into Docker → `pip install` finds everything cached → zero download on rebuild (~2s).

---

## Frontend

Nuxt 3 port 3000. Tailwind CSS. Sidebar: 3 groups, 10 nav items, 10 unique routes.

- **Main:** Dashboard Report, Test Upload
- **Dataset:** All Raw Data, Train Data, Test Data, Evaluation Metrics
- **Annotations:** BoundingBox Result, Segmentation Result, Inference, Test Inference

All grids: ZoomModal + Pagination.

---

## Scripts

| Script | Action |
|--------|--------|
| `restart-rebuild-api.sh` | Clean dataset (preserve `raw/`) → sync `.venv` → verify packages → build Docker → stop → start |
| `restart-rebuild-fe.sh` | Build frontend Docker → stop → start |
| `start-api.sh` | Build image if missing → free port → `docker compose up -d backend` (fallback direct uvicorn) |
| `start-fe.sh` | Free port → `docker compose up -d frontend` |
| `stop-api.sh` | `docker compose stop backend` |
| `stop-fe.sh` | `docker compose stop frontend` |
| `setup.sh` | One-time: create `.venv`, build Docker images, create log dirs |

All `.sh` files in `script/`.
