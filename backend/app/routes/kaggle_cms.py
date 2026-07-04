import asyncio
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from ultralytics import YOLO

from app.cli.train import train_one
from app.core.config import BASE_DIR, MODEL_PATH
from app.services.kaggle_service import download_and_prepare, _generate_mask

router = APIRouter(prefix="/api/kaggle", tags=["Kaggle CMS"])

KAGGLE_DIR = BASE_DIR / "dataset" / "kaggle_waste"
TRAIN_RUN_DIR = BASE_DIR / "runs" / "detect" / "api_train"

_executor = ThreadPoolExecutor(max_workers=1)


def _draw_polygon(img_bgr, poly):
    """Draw normalized polygon coordinates on BGR image, return copy."""
    h, w = img_bgr.shape[:2]
    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
    pts[:, 0] *= w
    pts[:, 1] *= h
    pts = pts.astype(np.int32)
    overlay = img_bgr.copy()
    cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
    cv2.fillPoly(overlay, [pts], (0, 255, 0))
    return cv2.addWeighted(overlay, 0.5, img_bgr, 0.5, 0)


@router.post("/download")
async def download():
    try:
        result = download_and_prepare()
        return result
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


@router.post("/convert")
async def convert():
    if not KAGGLE_DIR.exists():
        raise HTTPException(400, "Dataset not found. Download first.")

    counts = {"train": 0, "val": 0, "test": 0}

    for split in ("train", "val", "test"):
        img_dir = KAGGLE_DIR / split / "images"
        lbl_dir = KAGGLE_DIR / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue
            try:
                img = Image.open(img_path).convert("RGB")
                poly, _ = _generate_mask(img, randomize=(split == "train"))
                coords = " ".join(f"{v:.6f}" for v in poly)

                existing_lbl = lbl_dir / f"{img_path.stem}.txt"
                cls_id = "0"
                if existing_lbl.exists():
                    content = existing_lbl.read_text().strip()
                    if content:
                        cls_id = content.split()[0]

                (lbl_dir / f"{img_path.stem}.txt").write_text(f"{cls_id} {coords}\n")
                counts[split] += 1
            except Exception:
                continue

    return counts


@router.get("/download-status")
async def download_status():
    exists = KAGGLE_DIR.exists()
    train_count = val_count = test_count = 0
    labels_exist = False

    if exists:
        train_dir = KAGGLE_DIR / "train" / "images"
        val_dir = KAGGLE_DIR / "val" / "images"
        test_dir = KAGGLE_DIR / "test" / "images"

        if train_dir.is_dir():
            train_count = sum(1 for f in train_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if val_dir.is_dir():
            val_count = sum(1 for f in val_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))
        if test_dir.is_dir():
            test_count = sum(1 for f in test_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png"))

        lbl_dir = KAGGLE_DIR / "train" / "labels"
        labels_exist = lbl_dir.is_dir() and any(lbl_dir.iterdir())

    return {
        "dataset_exists": exists,
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "labels_exist": labels_exist,
    }


@router.get("/viz")
async def viz():
    img_dir = KAGGLE_DIR / "train" / "images"
    if not img_dir.is_dir():
        raise HTTPException(404, "Training images not found")

    images = sorted(
        f for f in img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    samples = random.sample(images, min(12, len(images)))
    urls = [f"/api/kaggle/viz/image/{img.name}" for img in samples]
    return {"images": urls, "total": len(images), "count": len(urls)}


@router.get("/viz/image/{filename}")
async def viz_image(filename: str):
    img_path = KAGGLE_DIR / "train" / "images" / filename
    if not img_path.exists():
        raise HTTPException(404, "Image not found")

    lbl_path = KAGGLE_DIR / "train" / "labels" / f"{img_path.stem}.txt"

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise HTTPException(500, "Failed to read image")

    if lbl_path.exists():
        content = lbl_path.read_text().strip()
        if content:
            parts = content.split()
            poly = [float(x) for x in parts[1:]]
            if len(poly) >= 4:
                img_bgr = _draw_polygon(img_bgr, poly)

    fd, out_path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(out_path, img_bgr)
    return FileResponse(out_path, media_type="image/jpeg", filename=filename)


@router.post("/train")
async def train(
    epochs: int = Query(50, description="Number of epochs"),
    batch: int = Query(16, description="Batch size"),
    imgsz: int = Query(640, description="Image size"),
    lr0: float = Query(0.001, description="Initial learning rate"),
):
    if not KAGGLE_DIR.exists():
        raise HTTPException(400, "Dataset not found. Download first.")

    data_yaml = KAGGLE_DIR / "data.yaml"
    if not data_yaml.exists():
        raise HTTPException(400, "Data config not found. Run download first.")

    loop = asyncio.get_event_loop()

    def _train():
        best_path, map50 = train_one(
            pretrained="yolo11m.pt",
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=30,
            device="cuda:0",
            name="api_train",
            lr0=lr0,
        )
        if best_path and Path(best_path).exists():
            MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(best_path), str(MODEL_PATH))
        return {
            "best_model_path": str(best_path),
            "map50": map50,
            "saved_to": str(MODEL_PATH),
        }

    try:
        result = await loop.run_in_executor(_executor, _train)
        return result
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")


@router.get("/results")
async def results():
    if not TRAIN_RUN_DIR.exists():
        return {"available": False, "images": []}

    candidates = [
        "results.png", "labels.jpg", "F1_curve.png", "PR_curve.png",
        "confusion_matrix.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg",
    ]
    base_url = "/api/kaggle/results/image"
    images = []
    for fname in candidates:
        fpath = TRAIN_RUN_DIR / fname
        if fpath.exists():
            images.append({"name": fname, "url": f"{base_url}/{fname}"})

    return {"available": True, "images": images}


@router.get("/results/image/{filename}")
async def results_image(filename: str):
    fpath = TRAIN_RUN_DIR / filename
    if not fpath.exists():
        raise HTTPException(404, "Result file not found")
    return FileResponse(str(fpath))


@router.get("/evaluate")
async def evaluate():
    if not MODEL_PATH.exists():
        raise HTTPException(400, f"No trained model found at {MODEL_PATH}")

    data_yaml = KAGGLE_DIR / "data.yaml"
    if not data_yaml.exists():
        raise HTTPException(400, "Dataset data.yaml not found")

    try:
        model = YOLO(str(MODEL_PATH))
        val_results = model.val(data=str(data_yaml), split="test")
        metrics = val_results.results_dict
        return {
            "mAP50": metrics.get("metrics/mAP50(B)", 0),
            "mAP50_95": metrics.get("metrics/mAP50-95(B)", 0),
            "precision": metrics.get("metrics/precision(B)", 0),
            "recall": metrics.get("metrics/recall(B)", 0),
        }
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")


@router.post("/inference")
async def inference(file: UploadFile = File(...)):
    if not MODEL_PATH.exists():
        raise HTTPException(400, "No trained model found")

    ext = Path(file.filename).suffix.lower()
    if ext not in (".jpg", ".jpeg", ".png"):
        raise HTTPException(400, "Unsupported image format. Use jpg/jpeg/png.")

    contents = await file.read()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    tmp.write(contents)
    tmp_path = tmp.name
    tmp.close()

    try:
        model = YOLO(str(MODEL_PATH))
        results = model(tmp_path)
        annotated = results[0].plot()

        out_path = tmp_path + "_annotated.jpg"
        cv2.imwrite(out_path, annotated)

        return FileResponse(
            out_path,
            media_type="image/jpeg",
            filename=f"annotated_{file.filename}",
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
