# API PRD - SmartBin

**Base URL:** `http://localhost:8000`

---

## Detection

### POST /api/detect
Upload image/video → detect Organik/Non-Organik objects.

Request: `multipart/form-data`, field `file`

Response:
```json
{
  "success": true,
  "file_type": "image",
  "detected_objects": [{"label": "...", "category": "Organik", "confidence": 0.98, "bbox": [x1,y1,x2,y2]}],
  "summary": {"organik": 1, "non_organik": 1, "total": 2},
  "result_url": "/static/result/...",
  "recommendation": "Buang ke Tempat Sampah Organik"
}
```

### POST /api/detect/bulk
Batch detection. Accepts multiple files. Returns per-file results + error list.

### GET /api/result/{filename}
Serve annotated result file.

### GET /api/log/{timestamp}
Read detection log from DB.

---

## Dataset

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/dataset/grid` | All images + annotations + stats across raw/train/val/test |
| GET | `/api/dataset/evaluate?split=all` | Model metrics per split (mAP, precision, recall) |
| GET | `/api/dataset/file/{source}/{filename}` | Serve dataset images/viz |
| POST | `/api/dataset/download` | Download TACO images → `raw/` |
| POST | `/api/dataset/split` | Self-cleans train/val/test → re-splits raw 70/15/15. `raw/` untouched |
| POST | `/api/dataset/prepare` | Download + split combined |
| POST | `/api/dataset/reset` | Remove all dataset dirs, recreate empty |

---

## Pipelines

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/dataset/pipeline/coco` | Generate COCO annotation visualization files |
| POST | `/api/dataset/pipeline/yolo` | YOLO inference on train images |
| POST | `/api/dataset/pipeline/yolo/val` | YOLO inference on val images |
| POST | `/api/dataset/pipeline/yolo/seg` | YOLO segmentation inference on train images |
| POST | `/api/dataset/pipeline/yolo/test` | YOLO inference on test images |

---

## Annotation Details

| Method | Path | Returns |
|--------|------|---------|
| GET | `/api/dataset/annotation/convert/{file}` | COCO annotation (class, bbox) |
| GET | `/api/dataset/annotation/real/{file}` | YOLO train prediction (class, bbox, conf) |
| GET | `/api/dataset/annotation/val/{file}` | YOLO val prediction (class, bbox, conf) |
| GET | `/api/dataset/annotation/test/{file}` | YOLO test prediction (class, bbox, conf) |

---

## Datasource

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/datasource/grid` | TACO annotations (1500 images, 4784 annotations) with image URL + category mapping |
| GET | `/api/datasource/image?url=` | Proxy + cache Flickr images to `static/datasource_cache/` |
| GET | `/api/datasource/file/{filename}` | Serve raw datasource files (JSON, CSV) |
| GET | `/api/datasource/annotations` | Full raw annotations JSON |
| GET | `/api/datasource/annotations/unofficial` | Unofficial annotations JSON |

---

## Health

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | `{"status":"ok","device":"cuda:0"}` |

---

## Errors

| Code | Description |
|------|-------------|
| 400 | Invalid file format / no raw images to split / raw directory not found |
| 404 | Source/file not found |
| 500 | Detection/pipeline failed |
| 502 | Failed to fetch datasource image |
