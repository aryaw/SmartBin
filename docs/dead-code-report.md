# Dead Code Report

Generated: 06-07-2026

Scan of `backend/app/` and `frontend/` for code defined but never used.

---

## Backend

### 17 Unused Endpoints

| Route File | Endpoint | Notes |
|-----------|----------|-------|
| `routes/datasource.py` (all 5) | `GET /api/datasource/grid` | TACO legacy — frontend migrated to `/api/kaggle/*` |
| `routes/datasource.py` | `GET /api/datasource/image` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/file/{filename}` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/annotations` | TACO legacy |
| `routes/datasource.py` | `GET /api/datasource/annotations/unofficial` | TACO legacy |
| `routes/health.py:18` | `GET /health` | No frontend caller |
| `routes/health.py:38` | `POST /health/reload` | No frontend caller |
| `routes/detect.py:63` | `POST /api/detect/bulk` | No frontend caller |
| `routes/detect.py:117` | `GET /api/result/{filename}` | Frontend uses `/static/result/` directly |
| `routes/detect.py:126` | `GET /api/log/{timestamp}` | No frontend caller |
| `routes/annotation.py:307` | `POST /api/dataset/prepare` | Superseded by `/api/kaggle/*` |
| `routes/annotation.py:348` | `POST /api/dataset/convert-seg` | Superseded by `/api/kaggle/convert` |
| `routes/annotation.py:359` | `POST /api/dataset/split-stratified` | No frontend caller |
| `routes/annotation.py:396` | `POST /api/dataset/download` | Frontend calls `/api/kaggle/download` |
| `routes/annotation.py:427` | `POST /api/dataset/split` | No frontend caller |
| `routes/annotation.py:522` | `POST /api/dataset/pipeline/yolo/train` | No frontend caller |
| `routes/annotation.py:623` | `POST /api/dataset/reset` | No frontend caller |
| `routes/annotation.py:677` | `GET /api/dataset/annotation/convert/{filename}` | No frontend caller |
| `routes/annotation.py:690` | `GET /api/dataset/annotation/real/{filename}` | No frontend caller |
| `routes/annotation.py:703` | `GET /api/dataset/annotation/val/{filename}` | No frontend caller |
| `routes/annotation.py:716` | `GET /api/dataset/annotation/test/{filename}` | No frontend caller |
| `routes/kaggle_cms.py:162` | `GET /api/kaggle/download-status` | No frontend caller |

### 3 Unused Service Functions

| File | Function | Notes |
|------|----------|-------|
| `services/preparation_service.py:14` | `prepare_dataset()` | Never imported — superseded |
| `services/datapreparation_service.py:110` | `run_download_only()` | Never imported |
| `services/yolo_service.py:127` | `_clean_outputs()` | Defined but never called |

### 1 Unused Service File

| File | Notes |
|------|-------|
| `services/preparation_service.py` | Single function `prepare_dataset()` never called |

---

## Frontend

### 13 Unused Pages

| Page File | Route | Notes |
|-----------|-------|-------|
| `pages/dataset.vue` | `/dataset` | Not linked from any nav or other page |
| `pages/dataset-prep.vue` | `/dataset-prep` | Not linked from any nav or other page |
| `pages/convert-viz.vue` | `/convert-viz` | Not linked from any nav or other page |
| `pages/train-eval.vue` | `/train-eval` | Not linked from any nav or other page |
| `pages/inference-export.vue` | `/inference-export` | Not linked from any nav or other page |
| `pages/kaggle-cms.vue` | `/kaggle-cms` | Not linked from any nav or other page |
| `pages/train-data.vue` | `/train-data` | Not linked from any nav or other page |
| `pages/test-data.vue` | `/test-data` | Not linked from any nav or other page |
| `pages/val-result.vue` | `/val-result` | Not linked from any nav or other page |
| `pages/test-result.vue` | `/test-result` | Not linked from any nav or other page |
| `pages/eval.vue` | `/eval` | Not linked from any nav or other page |
| `pages/segmentation-result.vue` | `/segmentation-result` | Not linked from any nav or other page |
| `pages/boundingbox-result.vue` | `/boundingbox-result` | Not linked from any nav or other page |

### 2 Unused Components

| Component File | Notes |
|---------------|-------|
| `components/FileUpload.vue` | All pages inline file upload logic instead |
| `components/LoadingOverlay.vue` | No page imports it |

### 2 Unused Composables

| Composable File | Notes |
|----------------|-------|
| `composables/useDetection.ts` | Orphaned — file upload inlined in pages |
| `composables/useFileUpload.ts` | Same |

---

## Summary

| Category | Count |
|----------|-------|
| Backend endpoints | 22 |
| Service functions | 3 |
| Service files | 1 |
| Frontend pages | 13 |
| Components | 2 |
| Composables | 2 |
| **Total** | **43** |
