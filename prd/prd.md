# SmartBin - Product Requirements Document

**Version:** 4.0
**Platform:** Web Application
**Stack:** FastAPI + Nuxt 3 + YOLOv26m-seg

---

## Dataset

Two data sources merged into `backend/dataset/raw/`:

### Source 1: TACO (Trash Annotations in Context)
- 1,500 images with COCO-format annotations (segmentation polygons, bounding boxes)
- 60 fine-grained categories (Aluminium foil, Battery, Plastic bottle, Food waste, etc.)
- Mapped to 2 classes: **Organik** (category 25: Food waste) and **Non-Organik** (all other 59 categories)
- Downloaded from Flickr via `waste_datasource/annotations.json`
- 7 Organik / 1,027 Non-Organik images after classification

### Source 2: Waste Classification Dataset (local)
- 2,939 images organized in subcategory folders under `backend/dataset/raw/`
- Original structure: `{Organic,Hazardous,Non-Recyclable,Recyclable}/{subcategory}/*.jpg`
- Mapped: Organic subcategories → Organik, everything else → Non-Organik
- 677 Organik / 2,262 Non-Organik images

### Merged Dataset
Total: **3,973 images** (684 Organik + 3,289 Non-Organik)
Stored at: `backend/dataset/raw/Organik/` and `backend/dataset/raw/Non-Organik/`

Both sources follow **Pergub Bali No.47/2019** classification standard.

---

## Pipeline

Single entry point on Dashboard:

| Button | Action | Endpoint |
|--------|--------|----------|
| Run Full Pipeline | Prepare data → Train YOLOv26m-seg → Validate → Copy model | POST /api/kaggle/pipeline/run-full?epochs=80&batch=16 |

Steps executed server-side:
1. Read images from `backend/dataset/raw/Organik/` and `Non-Organik/`
2. Generate pseudo-polygon segmentation masks (Otsu edge detection + geometric fallback)
3. Stratified train/val/test split (70/15/15)
4. Train YOLOv26m-seg with hyperparameters
5. Validate best model
6. Copy model to `backend/models/best.pt`

---

## Model

Architecture: YOLOv26m-seg (Medium Segmentation, 23.5M params)
- Pretrained: yolo26m-seg.pt (COCO)
- Input: 640x640
- Batch: 16
- Epochs: 80 (early stopping patience 40)
- Optimizer: SGD (MuSGD hybrid)
- Loss: CIoU (box) + BCE (cls) + DFL
- Augmentation: mosaic 1.0, mixup 0.2, copy_paste 0.15

Latest results:
- Box mAP@0.5: 80.4%
- Mask mAP@0.5: 49.7%
- Box Precision: 76.7%, Recall: 75.6%

---

## Backend

FastAPI port 8000. Routes:
| Prefix | Description |
|--------|-------------|
| `/api/detect` | Upload image/video → inference → annotated result |
| `/api/dataset/*` | Dataset management (COCO/YOLO conversion, splits) |
| `/api/kaggle/*` | Pipeline (download, train, evaluate, inference batch) |
| `/api/annotation/*` | Annotation tools, grid viewer, prepare/pipeline |
| `/health` | Health check + GPU status |

---

## Frontend

Nuxt 3 port 3000. Tailwind CSS. Pages:
| Route | Content |
|-------|---------|
| `/dashboard` | Upload file → detect → show results + Run Full Pipeline |
| `/test` | Multi-file upload + batch detect |
| `/result` | Detection result detail |
| `/train-eval` | Training evaluation report |
| `/val-result` | Validation results |
| `/test-result` | Test set results |
| `/inference-export` | Batch inference + export |

Sidebar: Main (Dashboard) + Report (Training Eval, Validation, Test Results, Inference)

---

## Scripts

| Script | Action |
|--------|--------|
| `backend/app/datapreparation/download_taco.py` | Download TACO dataset from annotations.json |
| `backend/app/datapreparation/prepare_data.py` | COCO→YOLO conversion, train/val/test split |
| `waste_datasource/prepare_dataset.py` | Download + classify TACO images into Organik/Non-Organik |
