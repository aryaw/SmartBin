# API PRD - SmartBin

**Base URL:** `http://localhost:8000`

---

## Kaggle CMS Pipeline (/api/kaggle)

Grouped by frontend page:

**Group 1 (/dataset-prep):**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/download` | Load dataset from local path (DATASET_PATH env) |
| GET | `/api/kaggle/download-status` | Check if dataset + labels exist |
| GET | `/api/kaggle/explore` | Class distribution (Organik/Non-Organik with Anorganik/Residu subtypes) + sample image URLs |

**Group 2 (/convert-viz):**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/convert` | Generate YOLO-seg polygon masks |
| GET | `/api/kaggle/viz` | 12 random sample images with mask overlays |
| GET | `/api/kaggle/viz/image/{filename}` | Single image with mask overlay |

**Group 3 (/train-eval):**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/train` | Train YOLOv26m-seg (params: epochs, batch, lr0) |
| GET | `/api/kaggle/results` | Training result plot images |
| GET | `/api/kaggle/results/image/{filename}` | Single result image |
| GET | `/api/kaggle/evaluate` | Box + Mask metrics |

**Group 4 (/inference-export):**
| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/inference` | Upload image -> mask overlay + detections |
| POST | `/api/kaggle/inference/batch` | Batch inference on test set + Organik/Non-Organik recycling advice with Anorganik/Residu mapping |
| POST | `/api/kaggle/export` | Export model (ONNX/TorchScript) |
| GET | `/api/kaggle/categories` | Class hierarchy + recycling advice |
| GET | `/api/kaggle/verify` | Per-class mask mAP final verification (Organik/Non-Organik) |

---

## Dataset Management (/api/dataset)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/grid` | All images + annotations + stats |
| GET | `/api/dataset/evaluate?split=all` | Model metrics per split |
| POST | `/api/dataset/download?source=local` | Load from local dataset path |
| POST | `/api/dataset/convert-seg` | Generate YOLO-seg masks from raw dataset |
| POST | `/api/dataset/split-stratified` | Stratified 70/15/15 split |
| POST | `/api/dataset/split` | Random 70/15/15 split from raw |
| POST | `/api/dataset/reset` | Reset all dataset dirs |
| POST | `/api/dataset/pipeline/coco` | COCO annotation viz |
| POST | `/api/dataset/pipeline/yolo` | YOLO train inference |
| POST | `/api/dataset/pipeline/yolo/val` | YOLO val inference |
| POST | `/api/dataset/pipeline/yolo/seg` | YOLO segmentation inference |
| POST | `/api/dataset/pipeline/yolo/test` | YOLO test inference |

---

## Detection

### POST /api/detect
Upload image/video → detect Organik/Non-Organik objects.

### POST /api/detect/bulk
Batch detection.

### GET /api/result/{filename}
Serve annotated result file.

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok","device":"cuda:0"}` |
| POST | `/health/reload` | Reload model from disk |
