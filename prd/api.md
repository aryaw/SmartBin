# API PRD - SmartBin

**Base URL:** `http://localhost:8000`

---

## Detection (/api)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/detect` | Upload image/video → YOLO inference → annotated result + summary |
| POST | `/api/detect/bulk` | Batch upload multiple files |
| GET | `/api/result/{filename}` | Get annotated result image |
| GET | `/api/log/{timestamp}` | Get detection log |

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, GPU info, model loaded |

## Kaggle CMS Pipeline (/api/kaggle)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/kaggle/pipeline/run-full` | Full pipeline: prepare data → train → validate → copy model |
| POST | `/api/kaggle/download` | Download/load dataset from source |
| POST | `/api/kaggle/convert` | Convert to YOLO-seg masks |
| POST | `/api/kaggle/train` | Train model with params |
| POST | `/api/kaggle/inference` | Single image inference |
| POST | `/api/kaggle/inference/batch` | Batch inference |
| POST | `/api/kaggle/export` | Export trained model |
| GET | `/api/kaggle/download-status` | Check dataset status |
| GET | `/api/kaggle/explore` | Dataset profiling |
| GET | `/api/kaggle/viz` | Sample mask visualizations |
| GET | `/api/kaggle/train/status` | Training status |
| GET | `/api/kaggle/results` | Training results/curves |
| GET | `/api/kaggle/evaluate` | Evaluate model on val set |
| GET | `/api/kaggle/verify` | Final verification |

## Dataset Management (/api/dataset)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/grid` | Dataset image grid |
| POST | `/api/dataset/pipeline/coco` | COCO pipeline |
| POST | `/api/dataset/pipeline/yolo` | YOLO bbox pipeline |
| POST | `/api/dataset/pipeline/yolo/seg` | YOLO seg pipeline |
| POST | `/api/dataset/pipeline/yolo/train` | YOLO training pipeline |
| POST | `/api/dataset/split-stratified` | Stratified dataset split |
| POST | `/api/dataset/split` | Dataset split |
| GET | `/api/dataset/evaluate` | Dataset evaluation |

## Annotations (/api/annotation)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/annotation/grid` | Annotation grid |
| POST | `/api/annotation/prepare` | Prepare annotations |
| POST | `/api/annotation/pipeline/coco` | COCO pipeline |
| POST | `/api/annotation/pipeline/yolo` | YOLO pipeline |
| POST | `/api/annotation/pipeline/yolo/seg` | YOLO seg pipeline |
| POST | `/api/annotation/convert-seg` | Convert to seg format |

## Datasource (/api)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/grid` | Datasource grid |
| GET | `/api/image` | Get image |
| GET | `/api/annotations` | Get annotations |
| GET | `/api/annotations/unofficial` | Get unofficial annotations |
