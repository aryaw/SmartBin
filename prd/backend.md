# Backend PRD - SmartBin

**Stack:** Python 3.12, FastAPI, Uvicorn, Ultralytics YOLO, OpenCV, PyTorch (CUDA)

---

## 1. Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework | FastAPI 0.115 |
| ASGI Server | Uvicorn |
| Python | 3.12 |
| DL Framework | PyTorch 2.x + CUDA |
| Model | Ultralytics YOLO (yolo26m-seg.pt) |
| CV Library | OpenCV, Pillow |
| Validation | Pydantic v2 |
| Database | PostgreSQL via asyncpg + SQLAlchemy 2.0 |

---

## 2. Directory Structure

```
backend/
├── app/
│   ├── main.py                 # FastAPI entry
│   ├── core/config.py          # Paths, CORS, GPU config
│   ├── cli/
│   │   ├── train.py            # Training with optimizer + mask params
│   │   ├── test.py             # Evaluation CLI
│   │   └── predict.py          # CLI inference
│   ├── routes/
│   │   ├── kaggle_cms.py       # /api/kaggle/* (16 pipeline endpoints)
│   │   ├── annotation.py       # /api/dataset/* (dataset management)
│   │   ├── detect.py           # POST /api/detect
│   │   ├── health.py           # GET /health
│   │   └── datasource.py       # TACO datasource
│   ├── services/
│   │   ├── kaggle_service.py   # Dataset loading + mask generation
│   │   ├── detector.py         # YOLO inference singleton
│   │   ├── yolo_service.py     # Train/val/test pipelines
│   │   ├── annotation_service.py   # COCO conversion
│   │   ├── datapreparation_service.py
│   │   └── log_service.py
│   ├── utils/
│   │   ├── gpu_utils.py        # GPU init, memory limit
│   │   └── progress.py         # SSE progress emitter
│   └── schemas/detection.py    # Pydantic models
├── dataset/
│   ├── kaggle_waste/           # 18-class dataset (train/val/test)
│   ├── raw/                    # Raw images
│   └── train/val/test/         # Split datasets
├── models/best.pt              # Trained YOLO weights
├── data.yaml                   # Dataset config
└── requirements.txt
```

---

## 3. Kaggle CMS Endpoints by Pipeline Group

### Group 1: /dataset-prep
| POST /api/kaggle/download | Load from local dataset path |
| GET /api/kaggle/download-status | Check dataset existence |
| GET /api/kaggle/explore | Class distribution report |

### Group 2: /convert-viz
| POST /api/kaggle/convert | Generate polygon masks |
| GET /api/kaggle/viz | Sample mask visualizations |

### Group 3: /train-eval
| POST /api/kaggle/train | Train YOLOv26m-seg |
| GET /api/kaggle/results | Training plots |
| GET /api/kaggle/evaluate | Box + Mask metrics |

### Group 4: /inference-export
| POST /api/kaggle/inference | Single image with masks |
| POST /api/kaggle/inference/batch | Batch test set inference |
| GET /api/kaggle/categories | Recycling advice mapping |
| POST /api/kaggle/export | ONNX/TorchScript export |
| GET /api/kaggle/verify | Per-class verification |

---

## 4. Training Config

```
Model: yolo26m-seg.pt (COCO pretrained)
Task: Instance Segmentation (18 classes)
Epochs: 120 (patience=20)
Batch: 16
Imgsz: 640
Optimizer: SGD (triggers MuSGD internally)
Loss: box=7.5, cls=1.5, mask_ratio=4, overlap_mask=True
Augmentation: mosaic=1.0, mixup=0.2, copy_paste=0.15
```

## 5. Polygon Mask Generation

3 strategies (tried in order):
1. Edge detection (Otsu threshold + contour) — ~83%
2. Elliptical polygon — ~10%
3. Rounded rectangle — ~7%

Min contour area threshold: 20% of image (reduces noise).
