# Route Rename Plan: /api/kaggle/* → /api/pipeline/*

## Rationale

Current prefix `/api/kaggle/*` tied to data source (Kaggle). Project is not Kaggle-specific — it's a waste classification pipeline. New prefix `/api/pipeline/*` reflects the actual domain: data → model → deploy pipeline.

## Route Mapping

### Group: Data Pipeline (`/api/pipeline/data/*`)

| Method | Current (kaggle) | Proposed (pipeline) | Purpose |
|--------|------------------|---------------------|---------|
| POST | `/api/kaggle/pipeline/run-full` | `/api/pipeline/run` | End-to-end: load → convert → train → eval |
| POST | `/api/kaggle/download` | `/api/pipeline/data/load` | Load raw dataset from local source |
| GET | `/api/kaggle/download-status` | `/api/pipeline/data/status` | Check dataset loading progress |
| GET | `/api/kaggle/explore` | `/api/pipeline/data/profile` | Dataset profiling: distribution, samples |
| POST | `/api/kaggle/convert` | `/api/pipeline/data/convert` | Convert classification → YOLO-seg masks |
| GET | `/api/kaggle/viz` | `/api/pipeline/data/viz` | Visualize polygon mask overlays |
| GET | `/api/kaggle/viz/image/{filename}` | `/api/pipeline/data/viz/{filename}` | Get specific viz image |
| GET | `/api/kaggle/categories` | `/api/pipeline/data/categories` | List all subcategories with metadata |

### Group: Model Pipeline (`/api/pipeline/model/*`)

| Method | Current (kaggle) | Proposed (pipeline) | Purpose |
|--------|------------------|---------------------|---------|
| POST | `/api/kaggle/train` | `/api/pipeline/model/train` | Start model training |
| GET | `/api/kaggle/results` | `/api/pipeline/model/results` | Training curves and metrics |
| GET | `/api/kaggle/results/image/{filename}` | `/api/pipeline/model/results/{filename}` | Specific training chart |
| GET | `/api/kaggle/evaluate` | `/api/pipeline/model/evaluate` | Evaluate on test set |
| POST | `/api/kaggle/export` | `/api/pipeline/model/export` | Export to ONNX/TorchScript |

### Group: Deploy Pipeline (`/api/pipeline/deploy/*`)

| Method | Current (kaggle) | Proposed (pipeline) | Purpose |
|--------|------------------|---------------------|---------|
| POST | `/api/kaggle/inference` | `/api/pipeline/deploy/infer` | Single image inference with mask |
| POST | `/api/kaggle/inference/batch` | `/api/pipeline/deploy/infer-batch` | Batch inference on test set |
| GET | `/api/kaggle/verify` | `/api/pipeline/deploy/verify` | Final per-class verification report |

## Other Route Prefixes (unchanged)

| Prefix | Purpose |
|--------|---------|
| `/api/dataset/*` | Dataset file serving |
| `/api/detect/*` | Live detection endpoint |
| `/health` | Health check |
| `/api/annotation/*` | COCO annotation pipeline |

## Implementation Order

1. Create new router `pipeline_cms.py` with `/api/pipeline/*` routes
2. Copy logic from `kaggle_cms.py` to new router
3. Update frontend `API_BASE` calls to new routes
4. Deprecate old `/api/kaggle/*` routes with redirect or removal
