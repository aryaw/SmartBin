# Test Plan - SmartBin

---

## Backend Verification

| Check | Expected |
|-------|----------|
| `python3 -m py_compile app/routes/*.py` | All 5 route files compile |
| `python3 -m py_compile app/services/*.py` | All 6 service files compile |
| `python3 -m py_compile app/cli/*.py` | All 3 CLI files compile |

## API Verification

| Endpoint | Expected |
|----------|----------|
| `GET /health` | `{"status":"ok"}` |
| `GET /api/kaggle/download-status` | Dataset exists or not |
| `GET /api/kaggle/explore` | Class distribution with 2 classes (Organik/Non-Organik) |
| `POST /api/kaggle/train` | Training starts with yolo26m-seg.pt |

## Frontend Pages

| Route | Content |
|-------|---------|
| `/dataset-prep` | Group 1: Load Dataset + Profiling |
| `/convert-viz` | Group 2: Convert Masks + Visualize |
| `/train-eval` | Group 3: Train + Curves + Evaluate |
| `/inference-export` | Group 4: Infer + Batch + Export + Verify |
| `/dashboard` | File upload + detect |

## Data Flow

```
backend/dataset/raw/
  ├── Organik/
  └── Non-Organik/  (subtypes: Anorganik recyclable, Residu landfill)
        │
        ▼ (POST /api/kaggle/download?source=local)
backend/dataset/kaggle_waste/
  ├── train/ (2041 images + labels)
  ├── val/ (438 images + labels)
  ├── test/ (438 images + labels)
  └── data.yaml (2 classes)
        │
        ▼ (POST /api/kaggle/train)
backend/models/best.pt (YOLOv26m-seg)
```

## Build Verification

1. All 16 Python files compile
2. 42 API endpoints registered
3. Frontend rebuild required to see sidebar changes
