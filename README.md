# SmartBin

Waste instance segmentation web app using YOLOv26m-seg. Classifies waste into 18 subcategories across 4 main categories: Hazardous, Non-Recyclable, Organic, Recyclable. Provides full CMS pipeline from dataset exploration to model deployment.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.12, Uvicorn |
| Frontend | Nuxt 3, Vue 3, TypeScript, Tailwind CSS |
| Model | Ultralytics YOLOv26m-seg (PyTorch) |
| Database | PostgreSQL + asyncpg / SQLAlchemy |
| Vision | OpenCV, Pillow |
| Container | Docker, Docker Compose |

## Dataset

- **Source:** phenomsg/waste-classification (Kaggle)
- **~2,917 images**, 18 subcategories, 4 main categories
- Classification dataset (no masks) — pipeline generates pseudo-polygon masks via Otsu edge detection (~83%) and geometric fallback (~17%)
- Configurable via `DATASET_PATH` env var (default: `backend/dataset/raw`)

### 18 Subcategories

| ID | Subcategory | Category |
|----|-------------|----------|
| 0 | batteries | Hazardous |
| 1 | e-waste | Hazardous |
| 2 | paints | Hazardous |
| 3 | pesticides | Hazardous |
| 4 | ceramic_product | Non-Recyclable |
| 5 | diapers | Non-Recyclable |
| 6 | platics_bags_wrappers | Non-Recyclable |
| 7 | sanitary_napkin | Non-Recyclable |
| 8 | stroform_product | Non-Recyclable |
| 9 | coffee_tea_bags | Organic |
| 10 | egg_shells | Organic |
| 11 | food_scraps | Organic |
| 12 | kitchen_waste | Organic |
| 13 | yard_trimmings | Organic |
| 14 | cans_all_type | Recyclable |
| 15 | glass_containers | Recyclable |
| 16 | paper_products | Recyclable |
| 17 | plastic_bottles | Recyclable |

## Project Structure

```
backend/
├── app/
│   ├── main.py                   # FastAPI entry, lifespan
│   ├── core/
│   │   └── config.py             # Env-driven config (paths, GPU, DB)
│   ├── cli/
│   │   ├── train.py              # Training CLI with grid search
│   │   ├── test.py               # Evaluation CLI
│   │   └── predict.py            # Inference CLI
│   ├── routes/
│   │   ├── kaggle_cms.py         # Kaggle CMS pipeline endpoints
│   │   ├── annotation.py         # Dataset grid, split, pipelines
│   │   ├── detect.py             # Upload & detect, bulk, result
│   │   ├── datasource.py         # TACO proxy & serve
│   │   └── health.py             # /health endpoint
│   ├── services/
│   │   ├── kaggle_service.py     # Kaggle download, pseudo-mask gen
│   │   ├── detector.py           # YOLO inference
│   │   ├── yolo_service.py       # Train/val/test/seg pipelines
│   │   ├── annotation_service.py # COCO-to-YOLO conversion
│   │   ├── datapreparation_service.py
│   │   └── preparation_service.py
│   ├── schemas/
│   ├── utils/
│   └── models/
├── models/                       # YOLO weights (best.pt)
├── dataset/                      # Images (gitignored)
│   ├── raw/                      # Raw dataset (DATASET_PATH)
│   ├── train/val/test/           # YOLO splits
│   └── kaggle_waste/             # Kaggle-prepared data
├── log/                          # Backend logs
├── .env.example
└── data.yaml

frontend/
├── pages/
│   ├── raw/
│   │   ├── dataset.vue           # Load + Profiling
│   │   ├── preparation.vue       # Convert + Visualize
│   │   ├── training.vue          # Train + Results + Evaluate
│   │   └── deployment.vue        # Inference + Export + Verify
│   ├── dashboard.vue             # Live detection
│   └── ... (legacy pages)
├── components/                   # ZoomModal, YoloGrid, Pagination, etc.
├── layouts/default.vue           # Sidebar nav (Dataset, Preparation, Training, Deployment)
├── composables/                  # useDetection, useFileUpload, useToast
├── nuxt.config.ts
└── .env.example
```

## Quick Start

### Prerequisites

- NVIDIA GPU with CUDA 12.8+ (or CPU fallback)
- Python 3.12, Node.js 22
- Docker 29.x + nvidia-container-toolkit (optional)

### Setup

```bash
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env
# Edit backend/.env - set DB_PASS, DATASET_PATH, GPU settings
```

### Run without Docker

```bash
USE_DOCKER=false bash script/restart-rebuild-api.sh
USE_DOCKER=false bash script/restart-rebuild-fe.sh
```

Or manually:

```bash
# Backend
cd backend
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm run dev
```

### Run with Docker Compose

```bash
USE_DOCKER=true bash script/restart-rebuild-api.sh
USE_DOCKER=true bash script/restart-rebuild-fe.sh
```

Or manually:

```bash
docker compose up -d backend
docker compose up -d frontend
```

## CMS Pipeline

Four sidebar menus under `/raw`:

| Menu | Route | Processes |
|------|-------|-----------|
| Dataset | `/raw/dataset` | Load Dataset, Dataset Profiling |
| Preparation | `/raw/preparation` | Convert to YOLO-seg, Visualize Masks |
| Training | `/raw/training` | Train YOLOv26m-seg, Results & Curves, Evaluation |
| Deployment | `/raw/deployment` | Inference, Batch Inference, Export, Final Verification |

## API Endpoints

### Kaggle CMS Pipeline

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/download` | Load dataset from local path |
| GET | `/api/kaggle/download-status` | Check dataset existence |
| GET | `/api/kaggle/explore` | Class distribution + sample URLs |
| POST | `/api/kaggle/convert` | Generate YOLO-seg labels |
| GET | `/api/kaggle/viz` | Mask visualization sample images |
| POST | `/api/kaggle/train` | Train YOLO-seg model |
| GET | `/api/kaggle/results` | Training result images |
| GET | `/api/kaggle/evaluate` | Box + Mask evaluation metrics |
| GET | `/api/kaggle/categories` | Category hierarchy + recycling advice |
| POST | `/api/kaggle/inference` | Single image inference |
| POST | `/api/kaggle/inference/batch` | Batch inference on test set |
| POST | `/api/kaggle/export` | Export to ONNX/TorchScript |
| GET | `/api/kaggle/verify` | Final verification + per-class mask mAP |

### Detection

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detect` | Upload & detect |
| GET | `/api/result/{filename}` | Annotated result file |
| GET | `/api/log/{timestamp}` | Detection log |

### Dataset (legacy TACO pipeline)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/grid` | All images + annotations + stats |
| POST | `/api/dataset/download` | Download dataset |
| POST | `/api/dataset/split` | Split raw to train/val/test |
| POST | `/api/dataset/convert-seg` | Convert to YOLO-seg format |
| POST | `/api/dataset/pipeline/coco` | COCO-to-YOLO annotation convert |
| POST | `/api/dataset/pipeline/yolo` | YOLO inference on train |
| GET | `/api/dataset/evaluate` | mAP metrics per split |
| POST | `/api/dataset/reset` | Wipe all splits |

## Training

```bash
cd backend

# Full training
python -m app.cli.train --model yolo26m-seg.pt --data data.yaml --epochs 120 --batch 16

# Grid search
python -m app.cli.train --model yolo26m-seg.pt --data data.yaml --grid-search 10 20 40

# Evaluate
python -m app.cli.test --model models/best.pt --data data.yaml --split test

# Predict
python -m app.cli.predict path/to/image.jpg
```

## Environment

### Backend (`backend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_PATH` | backend/dataset | Dataset location (relative to project root) |
| `EPOCHS` | 150 | Training epochs |
| `YOLO_BATCH_SIZE` | 16 | Training batch size |
| `YOLO_MAX_IMAGES` | 0 | Max images for training (0=all) |
| `USE_DOCKER` | false | Run mode flag |
| `DEVICE` | cuda:0 | GPU device |
| `CUDA_VISIBLE_DEVICES` | 0 | Visible GPU IDs |
| `VRAM_LIMIT_GB` | 12 | GPU memory limit |
| `UPLOAD_DIR` | uploads | Upload directory (relative to backend/) |
| `LOG_DIR` | log | Log directory |
| `STATIC_DIR` | static | Static files directory |
| `RESULT_DIR` | static/result | Detection results directory |
| `MODEL_DIR` | models | Model weights directory |
| `MODEL_PATH` | models/best.pt | Model checkpoint path |
| `MAX_FILE_SIZE` | 209715200 | Max upload size (bytes) |
| `ANNOTATIONS_FILE` | ../datasource/annotations.json | Annotations path |
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `DB_NAME` | smartbin | Database name |
| `DB_USER` | postgres | Database user |
| `DB_PASS` | - | Database password |

### Frontend (`frontend/.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE` | http://localhost:8080 | Backend API URL |
| `USE_DOCKER` | false | Run mode flag |

## Architecture

YOLOv26m-seg features:
- MuSGD Optimizer (SGD + Muon hybrid)
- Semantic Segmentation Loss
- Multi-Scale Proto Modules
- NMS-Free End-to-End
- Pseudo-polygon mask generation (Otsu edge detection + geometric fallback)

## Scripts

| Script | Description |
|--------|-------------|
| `script/restart-rebuild-api.sh` | Rebuild and restart backend |
| `script/restart-rebuild-fe.sh` | Rebuild and restart frontend |
