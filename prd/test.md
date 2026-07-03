# Test Plan - SmartBin

---

## Dataset

| Path | Content |
|------|---------|
| `backend/dataset/raw/` | 1500 raw images from TACO |
| `backend/dataset/train/images/` | 1050 training images |
| `backend/dataset/train/labels/` | 1050 YOLO .txt annotation files |
| `backend/dataset/val/images/` | 225 validation images |
| `backend/dataset/val/labels/` | 225 YOLO .txt annotation files |
| `backend/dataset/test/images/` | 225 test images |
| `backend/dataset/test/labels/` | 225 YOLO .txt annotation files |

## API Verification

| Endpoint | Expected |
|----------|----------|
| `GET /health` | `{"status":"ok","device":"cuda:0"}` |
| `GET /api/dataset/grid` | Raw/Train/Val/Test counts |
| `GET /api/datasource/grid` | 1500 images, 4784 annotations |
| `POST /api/dataset/download` | Download success count |
| `POST /api/dataset/split` | Train/Val/Test counts returned |
| `GET /api/dataset/evaluate?split=all` | Metrics per split |

## GPU Memory

- `VRAM_LIMIT_GB=12` env var
- `half=True` (FP16) inference
- Cache clear every 16 images
- Training batch size: 16

## Build Verification

1. `restart-rebuild-api.sh` syncs `.venv`, verifies all packages, builds Docker
2. Docker health check passes within 15s
3. `restart-rebuild-fe.sh` builds frontend
4. Both containers running and accessible

## Frontend Pages

| Route | Features |
|-------|----------|
| `/dashboard` | Upload + detect + Prepare Data |
| `/raw` | Grid + zoom + Download + Split |
| `/train-data` | Grid + zoom + Generate Annotation + Segmentation |
| `/eval` | Metrics table + val images |
| `/boundingbox-result` | COCO bbox table + zoom |
| `/segmentation-result` | Segmentation polygon table + zoom |
| `/val-result` | Run val + results + zoom |
| `/test-result` | Run test + results + zoom |
