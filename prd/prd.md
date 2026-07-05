# SmartBin - Product Requirements Document

**Version:** 3.0
**Platform:** Web Application
**Stack:** FastAPI + Nuxt 3 + YOLOv26m-seg

---

## Dataset

Source: phenomsg/waste-classification (Kaggle), ~2,917 images, 18 subcategories, 4 main categories.

Main categories: Hazardous, Non-Recyclable, Organic, Recyclable.

Classes (18): batteries, e-waste, paints, pesticides, ceramic_product, diapers, platics_bags_wrappers, sanitary_napkin, stroform_product, coffee_tea_bags, egg_shells, food_scraps, kitchen_waste, yard_trimmings, cans_all_type, glass_containers, paper_products, plastic_bottles.

Data stored at `backend/dataset/raw/` (configurable via `DATASET_PATH` env). Loaded into `backend/dataset/kaggle_waste/` via pipeline.

---

## Pipeline (4 Groups)

Four pages, one per pipeline group:

**Group 1: /dataset-prep — Dataset Preparation & Profiling**
| Button | Endpoint |
|--------|----------|
| Load Dataset | POST /api/kaggle/download?source=local |
| Dataset Profiling | GET /api/kaggle/explore |

**Group 2: /convert-viz — Convert & Visualize Masks**
| Button | Endpoint |
|--------|----------|
| Convert Masks | POST /api/kaggle/convert |
| Visualize Samples | GET /api/kaggle/viz |

**Group 3: /train-eval — Train, Results & Evaluation**
| Button | Endpoint |
|--------|----------|
| Train Model | POST /api/kaggle/train |
| Show Curves | GET /api/kaggle/results |
| Evaluate | GET /api/kaggle/evaluate |

**Group 4: /inference-export — Inference, Export & Verify**
| Button | Endpoint |
|--------|----------|
| Upload & Infer | POST /api/kaggle/inference |
| Batch Test | POST /api/kaggle/inference/batch |
| Export Model | POST /api/kaggle/export |
| Final Verify | GET /api/kaggle/verify |

---

## Model

Architecture: YOLOv26m-seg (Medium Segmentation)
- MuSGD Optimizer (SGD + Muon hybrid)
- Semantic Segmentation Loss
- Multi-Scale Proto Modules
- NMS-Free End-to-End
- No DFL (edge device support)
- ProgLoss + STAL (small object detection)

Training: 120 epochs, batch=16, imgsz=640, patience=20.
Augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.15.

---

## Backend

FastAPI port 8000. Routes:
- `/api/kaggle/*` — CMS pipeline (16 endpoints)
- `/api/dataset/*` — Dataset management (20 endpoints)
- `/api/detect` — Live inference
- `/health` — Health check

---

## Frontend

Nuxt 3 port 3000. Tailwind CSS.

Sidebar: Main (Dashboard) + Pipeline (4 groups).

Pages:
- `/dataset-prep` — Group 1: Load dataset + profiling report
- `/convert-viz` — Group 2: Convert masks + visualize samples
- `/train-eval` — Group 3: Train model + view curves + evaluate metrics
- `/inference-export` — Group 4: Upload infer + batch test + export + verify
- `/dashboard` — Live detection upload

---

## Scripts

| Script | Action |
|--------|--------|
| `restart-rebuild-api.sh` | Build backend Docker → stop → start |
| `restart-rebuild-fe.sh` | Build frontend Docker → stop → start |
| `start-api.sh` | Start backend container |
| `start-fe.sh` | Start frontend container |
| `stop-api.sh` | Stop backend |
| `stop-fe.sh` | Stop frontend |
| `setup.sh` | One-time Docker build |
