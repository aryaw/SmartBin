# Backend PRD - SmartBin

**Stack:** Python 3.12, FastAPI, Uvicorn, Ultralytics YOLO, OpenCV, PyTorch (CUDA)

---

## 1. Tech Stack

| Komponen      | Teknologi                          |
|---------------|------------------------------------|
| Framework     | FastAPI 0.115                      |
| ASGI Server   | Uvicorn                            |
| Python        | 3.12 (host `.venv` + Docker)       |
| DL Framework  | PyTorch 2.x + CUDA                 |
| Model         | Ultralytics YOLO (`models/best.pt`) |
| CV Library    | OpenCV, Supervision, cvzone, Pillow |
| File Handling | python-multipart, aiofiles         |
| Validation    | Pydantic v2                        |
| HTTP Client   | httpx                              |
| Database      | PostgreSQL via asyncpg + SQLAlchemy 2.0 |
| DB Driver     | asyncpg                            |
| HEIC Support  | pi-heif                            |
| CORS          | `["http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:3000", "http://127.0.0.1:8000"]` |

---

## 2. GPU & Memory

- VRAM limit: 12GB via `VRAM_LIMIT_GB=12` env var
- `half=True` (FP16) for inference
- `imgsz=640`
- `torch.cuda.empty_cache()` every 16 images
- `torch.cuda.set_per_process_memory_fraction()` caps GPU usage
- Device configurable via `DEVICE` env var (default `cuda:0`)

---

## 3. Build Strategy (Docker)

1. Host `.venv` created with `python3.12 -m venv --copies .venv` (no symlinks → portable)
2. All packages installed from `requirements.txt`
3. Symlinks resolved to real files for Docker compat
4. Shebangs rewritten from host path to `/app/.venv/bin/python3.12`
5. `.venv` copied into Docker image (`COPY .venv .venv`)
6. `ENV PATH=/app/.venv/bin:$PATH`
7. `pip install --break-system-packages -r requirements.txt` - finds all packages already in `.venv` → zero download
8. `--mount=type=cache,target=/root/.cache/pip` - fallback cache

Rebuild: ~2s (no pip download, just copy `.venv` + source code).

---

## 4. Directory Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry (lifespan: GPU init + model load)
│   ├── core/
│   │   └── config.py           # Paths, CORS, DB, GPU config
│   ├── cli/
│   │   ├── train.py            # Training CLI with grid search
│   │   ├── test.py             # Evaluation CLI
│   │   └── predict.py          # CLI inference helper
│   ├── routes/
│   │   ├── health.py           # GET /health
│   │   ├── detect.py           # POST /api/detect, /detect/bulk, /result, /log
│   │   ├── annotation.py       # Dataset grid, split, pipelines, file serving, evaluate
│   │   └── datasource.py       # /api/datasource/* (TACO annotations proxy)
│   ├── services/
│   │   ├── detector.py         # YOLO inference (image/video) with cvzone bbox drawing
│   │   ├── annotation_service.py   # COCO pipeline (COCO→YOLO conversion viz)
│   │   ├── yolo_service.py         # YOLO train/val/test inference pipelines
│   │   ├── datapreparation_service.py  # Download + split TACO data
│   │   └── log_service.py       # Detection logging to PostgreSQL
│   ├── schemas/
│   │   └── detection.py        # Pydantic models for detect response
│   ├── utils/
│   │   ├── gpu_utils.py        # GPU init, memory limit, warmup
│   │   ├── file_utils.py       # Upload validation, save, cleanup
│   │   └── progress.py         # SSE progress emitter
│   ├── models/
│   │   └── database.py         # SQLAlchemy async engine + models
│   └── datapreparation/        # Dataset download & split scripts
├── dataset/
│   ├── raw/                    # 1500 images (TACO) - NEVER deleted
│   ├── train/images+labels/    # 70% split
│   ├── val/images+labels/      # 15% split
│   ├── test/images+labels/     # 15% split
│   ├── coco_gt_bbox{,_img}/      # COCO ground truth bbox + viz
│   ├── yolo_train_bbox{,_img}/   # YOLO train predictions + viz
│   ├── yolo_train_seg{,_img}/    # YOLO train segmentation + viz
│   ├── yolo_val_bbox{,_img}/     # YOLO val predictions + viz
│   └── yolo_test_bbox{,_img}/    # YOLO test predictions + viz
├── models/best.pt              # YOLO weights (trained)
├── .venv/                      # Python 3.12 (--copies, portable)
├── requirements.txt
├── data.yaml                   # Dataset config for YOLO
├── Dockerfile
├── log-wrapper.sh              # Entrypoint logging wrapper
└── static/result/              # Annotated detection outputs
```

---

## 5. API Endpoints (15+)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check + device info |
| POST | `/api/detect` | Upload file → detect objects |
| POST | `/api/detect/bulk` | Batch detection |
| GET | `/api/result/{filename}` | Serve annotated result |
| GET | `/api/log/{timestamp}` | Detection log from DB |
| GET | `/api/dataset/grid` | All dataset stats + images |
| GET | `/api/dataset/evaluate` | Model metrics (mAP, precision, recall) |
| GET | `/api/dataset/file/{source}/{file}` | Serve images/viz |
| POST | `/api/dataset/download` | Download TACO images → `raw/` |
| POST | `/api/dataset/split` | Self-cleaning 70/15/15 split from raw |
| POST | `/api/dataset/prepare` | Download + split combined |
| POST | `/api/dataset/reset` | Wipe all dataset dirs, recreate empty |
| POST | `/api/dataset/pipeline/coco` | Generate COCO annotation viz |
| POST | `/api/dataset/pipeline/yolo` | YOLO train inference |
| POST | `/api/dataset/pipeline/yolo/val` | YOLO val inference |
| POST | `/api/dataset/pipeline/yolo/test` | YOLO test inference |
| GET | `/api/dataset/annotation/{type}/{file}` | Annotation detail |
| GET | `/api/datasource/grid` | TACO annotations |
| GET | `/api/datasource/image` | Proxy + cache Flickr |
| GET | `/api/datasource/file/{file}` | Serve datasource files |
| GET | `/api/datasource/annotations` | Full JSON |

---

## 6. Training Config

```bash
# Single run
python -m app.cli.train --model yolo26n.pt --data data.yaml --epochs 100 --batch 16 --imgsz 640

# Grid search (best model auto-copied to models/best.pt)
python -m app.cli.train --model yolo26n.pt --data data.yaml --grid-search 10 20 40

# Evaluate on test set
python -m app.cli.test --model models/best.pt --data data.yaml --split test

# Predict single image
python -m app.cli.predict path/to/image.jpg
```

| Param | Value |
|-------|-------|
| Model | YOLO26n (`yolo26n.pt` from project root) |
| Batch | 16 |
| Imgsz | 640 |
| Augment | mosaic=1.0, HSV (h=0.015, s=0.7, v=0.4), scale=0.5, translate=0.1, degrees=10, shear=2, flipud=0.1, fliplr=0.5, erasing=0.4 |
| Patience | 20 |
| Device | auto (GPU if available) |
| Val | Runs automatically after training, prints per-class mAP |
