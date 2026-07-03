# SmartBin

Waste classification web app using YOLO deep learning. Detects and classifies waste into **Organik** (organic) and **Non-Organik** (inorganic) categories.

Built on the [TACO dataset](http://tacodataset.org/) - 1500 images, 4784 annotations, 60 COCO categories mapped to 2 classes.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12, Uvicorn |
| Frontend | Nuxt 3, Vue 3, TypeScript, Tailwind CSS |
| Model | Ultralytics YOLO (PyTorch) |
| Database | PostgreSQL + asyncpg / SQLAlchemy |
| Vision | OpenCV, Pillow, Supervision |
| Container | Docker, Docker Compose |

## Project Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry, lifespan, model load
│   ├── config.py               # Paths, CORS, GPU, DB config
│   ├── routes/
│   │   ├── annotation.py       # Grid, split, pipelines, evaluate, file serve
│   │   ├── detect.py           # Upload & detect, bulk, result, log
│   │   ├── datasource.py       # TACO proxy & serve
│   │   └── health.py           # /health
│   ├── services/
│   │   ├── detector.py         # YOLO inference
│   │   ├── yolo_service.py     # Train/val/test/seg pipelines
│   │   ├── annotation_service.py   # COCO→YOLO conversion
│   │   └── datapreparation_service.py
│   ├── schemas/
│   ├── utils/
│   └── models/
├── models/best.pt              # Trained weights
├── dataset/                    # Images + annotations (gitignored)
├── .env.example
├── train.py
└── data.yaml

frontend/
├── pages/                      # 13 routes (dashboard, dataset, eval, etc.)
├── components/                 # ZoomModal, YoloGrid, Pagination, etc.
├── layouts/default.vue         # Sidebar navigation
├── composables/                # useDetection, useFileUpload
├── nuxt.config.ts
└── .env.example
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.8+ (or CPU fallback)
- Docker 29.x + nvidia-container-toolkit
- Python 3.12, Node.js 22

### Setup

```bash
# Copy env files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit backend/.env - set DB_PASS and adjust GPU settings
```

### Run with scripts

```bash
./script/setup.sh                          # One-time: venv, deps, Docker build
./script/start-api.sh &                    # Backend → localhost:8000
./script/start-fe.sh &                     # Frontend → localhost:3000
```

### Run without Docker

```bash
# Backend
cd backend
.venv/bin/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
```

### Run with Docker Compose

```bash
docker compose up -d backend
docker compose up -d frontend
```

## Workflow

1. **Prepare Data** - Download TACO images via dashboard or `POST /api/dataset/download`, then split via `POST /api/dataset/split` (70/15/15 train/val/test, auto-dedup by MD5)
2. **Annotate** - Generate annotations via train-data page buttons: Generate Annotation (COCO bbox) and Generate Segmentation
3. **Train** - `python train.py --model yolo26n.pt --data data.yaml --epochs 100`
4. **Evaluate** - View mAP metrics at `/eval` or `GET /api/dataset/evaluate?split=all`
5. **Detect** - Upload images at `/dashboard` for real-time classification

## API Endpoints

### Detection
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detect` | Upload & detect (file + base64) |
| GET | `/api/result/{filename}` | Get annotated result |
| GET | `/api/log/{timestamp}` | Detection log |

### Dataset
| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/grid` | All images + annotations + stats |
| POST | `/api/dataset/download` | Download TACO to raw/ |
| POST | `/api/dataset/split` | Split raw → train/val/test (MD5 dedup) |
| POST | `/api/dataset/reset` | Wipe all splits |
| POST | `/api/dataset/pipeline/coco` | COCO→YOLO annotation convert |
| POST | `/api/dataset/pipeline/yolo` | YOLO inference on train |
| POST | `/api/dataset/pipeline/yolo/val` | YOLO inference on val |
| POST | `/api/dataset/pipeline/yolo/test` | YOLO inference on test |
| POST | `/api/dataset/pipeline/yolo/seg` | YOLO segmentation inference |
| GET | `/api/dataset/evaluate` | mAP50, precision, recall per split |

## Training

```bash
cd backend

# Basic
python train.py --model ../yolo26n.pt --data data.yaml --epochs 100 --batch 16 --imgsz 640

# Grid search over epochs
python train.py --model ../yolo26n.pt --data data.yaml --grid-search 10 20 40
```

## Environment

### Backend (`backend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `EPOCHS` | 150 | Training epochs |
| `DEVICE` | cuda:0 | GPU device |
| `VRAM_LIMIT_GB` | 12 | GPU memory limit |
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | smartbin | Database name |
| `DB_USER` | postgres | Database user |
| `DB_PASS` | - | Database password |

### Frontend (`frontend/.env`)
| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE` | http://localhost:8080 | Backend API URL |
