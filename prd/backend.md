# Backend PRD - SmartBin

**Stack:** Python 3.12, FastAPI, Uvicorn, Ultralytics YOLO, OpenCV, PyTorch (CUDA)

---

## 1. Tech Stack

| Komponen | Teknologi |
|----------|-----------|
| Framework | FastAPI |
| ASGI Server | Uvicorn |
| Python | 3.12 |
| DL Framework | PyTorch 2.12 + CUDA 13.0 |
| Model | Ultralytics YOLO (yolo26m-seg) |
| CV Library | OpenCV, Pillow |
| Validation | Pydantic v2 |

## 2. Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app, lifespan (GPU init + model load)
│   ├── core/
│   │   └── config.py        # Paths, categories, CORS, DB config
│   ├── routes/
│   │   ├── detect.py        # POST /api/detect, /api/detect/bulk
│   │   ├── health.py        # GET /health
│   │   ├── annotation.py    # Dataset annotation pipelines
│   │   ├── datasource.py    # Datasource grid/images
│   │   └── kaggle_cms.py    # Kaggle CMS pipeline endpoints
│   ├── services/
│   │   ├── detector.py      # YOLO model load, image/video inference
│   │   ├── yolo_service.py  # YOLO pipeline (inference, viz, seg)
│   │   ├── kaggle_service.py# Dataset download, prepare_from_local
│   │   ├── annotation_service.py
│   │   ├── datapreparation_service.py
│   │   ├── log_service.py
│   │   └── preparation_service.py
│   ├── schemas/
│   │   └── detection.py     # Pydantic models (DetectResponse, etc.)
│   ├── utils/
│   │   ├── file_utils.py    # Upload validation, save, cleanup
│   │   ├── gpu_utils.py     # GPU init, warmup, device selection
│   │   └── progress.py      # SSE progress emitter
│   └── datapreparation/
│       ├── download_taco.py # TACO dataset downloader
│       └── prepare_data.py  # COCO→YOLO conversion, split
├── models/
│   └── best.pt              # Trained model (auto-copied after pipeline)
├── dataset/
│   └── raw/
│       ├── Organik/         # 684 organic waste images
│       └── Non-Organik/     # 3,289 non-organic waste images
├── requirements.txt
└── .env
```

## 3. Inference Flow

1. User uploads image via POST /api/detect
2. Server validates file type (.jpg, .jpeg, .png, .mp4, .avi, .mov) and size (max 200MB)
3. Image saved to uploads/, inference runs on GPU
4. YOLO model returns detected objects with class, confidence, bbox
5. Annotated image drawn with bounding boxes + labels
6. Result saved to static/result/, uploaded file deleted
7. Response: detected_objects[], summary {organik, non_organik, total}, result_url, recommendation
8. Detection logged to file

## 4. Key Config (config.py)

ORGANIC_CATEGORIES = {25}  # only Food waste

MODEL_PATH = models/best.pt
DATASET_PATH = backend/dataset
DEVICE = cuda:0

## 5. Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_PATH | models/best.pt | Path to model weights |
| DATASET_PATH | backend/dataset | Dataset root directory |
| DEVICE | cuda:0 | GPU device |
| UPLOAD_DIR | uploads | Upload directory |
| EPOCHS | 150 | Training epochs |
| BATCH_SIZE | 16 | Training batch size |
