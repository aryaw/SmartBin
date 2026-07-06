# Dead Code Report

Generated: 06-07-2026

Scan of `backend/app/` and `frontend/` for code defined but never used.

**Note:** No database in use. `models/database.py` exists but `init_db`, `save_detection` never imported by any route or service. All detection logging is file-based via `log_service.py`.

---

## Backend

### 23 Unused Endpoints

| Route File | Endpoint | Notes |
|-----------|----------|-------|
| `routes/datasource.py` (all 5) | `GET /api/datasource/grid` | TACO legacy - frontend migrated to `/api/kaggle/*` |
| `routes/datasource.py` | `GET /api/datasource/image` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/file/{filename}` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/annotations` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/annotations/unofficial` | TACO legacy |
| `routes/health.py` | `GET /health` | No frontend caller |
| `routes/health.py` | `POST /health/reload` | No frontend caller |
| `routes/detect.py` | `GET /api/result/{filename}` | Frontend uses `/static/result/` directly |
| `routes/detect.py` | `GET /api/log/{timestamp}` | No frontend caller |
| `routes/annotation.py` | `GET /api/dataset/grid` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/prepare` | Superseded by `/api/kaggle/*` |
| `routes/annotation.py` | `POST /api/dataset/pipeline/coco` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/pipeline/yolo` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/pipeline/yolo/val` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/convert-seg` | Superseded by `/api/kaggle/convert` |
| `routes/annotation.py` | `POST /api/dataset/split-stratified` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/download` | Frontend calls `/api/kaggle/download` |
| `routes/annotation.py` | `POST /api/dataset/split` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/pipeline/yolo/test` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/pipeline/yolo/train` | No frontend caller |
| `routes/annotation.py` | `POST /api/dataset/reset` | No frontend caller |
| `routes/annotation.py` (all 4) | `GET /api/dataset/annotation/.../` | No frontend caller |
| `routes/kaggle_cms.py` | `GET /api/kaggle/download-status` | No frontend caller |

### 4 Unused Service Functions

| File | Function | Notes |
|------|----------|-------|
| `services/preparation_service.py:14` | `prepare_dataset()` | Never imported - superseded |
| `services/datapreparation_service.py:110` | `run_download_only()` | Never imported |
| `services/yolo_service.py:127` | `_clean_outputs()` | Defined but never called |
| `services/yolo_service.py:337` | `_save_viz()` | Defined but never called |

### 2 Unused Service Files

| File | Notes |
|------|-------|
| `services/preparation_service.py` | Single function `prepare_dataset()` never called |
| `models/database.py` | `init_db()`, `save_detection()` never imported. DB config exists but unused |

### 1 Unused Model

| File | Notes |
|------|-------|
| `models/database.py` | SQLAlchemy async model + `Detection` table. No route or service imports it |

---

## Frontend

### 14 Unused Pages

| Page File | Route | Notes |
|-----------|-------|-------|
| `pages/dataset.vue` | `/dataset` | Not linked from nav or any page |
| `pages/dataset-prep.vue` | `/dataset-prep` | Not linked |
| `pages/convert-viz.vue` | `/convert-viz` | Not linked |
| `pages/train-data.vue` | `/train-data` | Not linked |
| `pages/test-data.vue` | `/test-data` | Not linked |
| `pages/eval.vue` | `/eval` | Not linked |
| `pages/segmentation-result.vue` | `/segmentation-result` | Not linked |
| `pages/boundingbox-result.vue` | `/boundingbox-result` | Not linked |
| `pages/kaggle-cms.vue` | `/kaggle-cms` | Not linked |
| `pages/test.vue` | `/test` | Not linked |
| `pages/result.vue` | `/result` | Not linked |
| `pages/raw/dataset.vue` | `/raw/dataset` | Not linked |
| `pages/raw/preparation.vue` | `/raw/preparation` | Not linked |
| `pages/raw/training.vue` | `/raw/training` | Not linked |
| `pages/raw/deployment.vue` | `/raw/deployment` | Not linked |

**Pages actually linked from sidebar (active):** `/dashboard`, `/train-eval`, `/val-result`, `/test-result`, `/inference-export`, `/visualization`

### 2 Unused Components

| Component File | Notes |
|---------------|-------|
| `components/FileUpload.vue` | All pages inline file upload logic instead |
| `components/DetectionResult.vue` | No page imports it |

### 2 Unused Composables

| Composable File | Notes |
|----------------|-------|
| `composables/useDetection.ts` | Orphaned - file upload inlined in pages |
| `composables/useFileUpload.ts` | Same |

**Note:** `useVisualization.ts` is used by `visualization.vue`. `useToast.ts` is widely used. `LoadingOverlay.vue` is used only by `visualization.vue`.

---

## Summary

| Category | Count |
|----------|-------|
| Backend endpoints | 23 |
| Service functions | 4 |
| Service files | 2 |
| Models (unused) | 1 |
| Frontend pages | 14 |
| Components | 2 |
| Composables | 2 |
| **Total** | **48** |
