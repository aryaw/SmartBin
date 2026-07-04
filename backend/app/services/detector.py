import time
import logging
from datetime import datetime
from pathlib import Path

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

from app.config import MODEL_PATH, RESULT_DIR
from app.utils.gpu_utils import get_device, warmup_model

logger = logging.getLogger(__name__)

CATEGORY_MAP = {
    0: ("Organik", "Organik"),
    1: ("Non-Organik", "Non-Organik"),
}

RECOMMENDATION_MAP = {
    "Organik": "Buang ke Tempat Sampah Organik",
    "Non-Organik": "Buang ke Tempat Sampah Non-Organik",
}

COLOR_MAP = {
    0: (0, 180, 80),
    1: (80, 120, 255),
}

CONF_THRESHOLD = 0.25

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if MODEL_PATH.stat().st_size < 1_000_000:
        logger.warning(f"Model suspiciously small ({MODEL_PATH.stat().st_size} bytes)")
    _model = YOLO(str(MODEL_PATH))
    warmup_model(_model)
    _validate_model()
    return _model


def _validate_model():
    if _model is None:
        raise RuntimeError("Model not loaded")
    if not hasattr(_model, "names") or len(_model.names) == 0:
        raise RuntimeError("Model has no class names")
    logger.info(f"Model loaded: {MODEL_PATH.name}, classes: {_model.names}")


def reload_model():
    global _model
    _model = None
    return load_model()


def _draw_detections(img: np.ndarray, boxes, detected: list) -> np.ndarray:
    overlay = img.copy()
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        if conf < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        w, h = x2 - x1, y2 - y1
        color = COLOR_MAP.get(cls_id, (200, 200, 200))
        label = CATEGORY_MAP.get(cls_id, ("Unknown",))[0]

        cvzone.cornerRect(overlay, (x1, y1, w, h), t=2, rt=0, colorR=color)
        cvzone.putTextRect(
            overlay, f"{label} {conf:.2f}",
            (max(0, x1), max(35, y1 - 10)),
            scale=0.8, thickness=1,
            colorR=color, colorT=(255, 255, 255),
            font=cv2.FONT_HERSHEY_SIMPLEX,
        )

        detected.append({
            "label": label,
            "category": CATEGORY_MAP.get(cls_id, ("Unknown", "Unknown"))[1],
            "confidence": round(conf, 4),
            "bbox": [round(v, 2) for v in [x1, y1, x2, y2]],
        })

    return cv2.addWeighted(overlay, 1, img, 0, 0)


def _run_inference_stream(model, frame):
    try:
        return model(frame, device=get_device(), verbose=False, stream=True)
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(f"GPU OOM in video inference, falling back to CPU: {e}")
            import torch
            torch.cuda.empty_cache()
            return model(frame, device="cpu", verbose=False, stream=True)
        raise


def _run_inference(model, img):
    try:
        return model(img, device=get_device(), verbose=False)[0]
    except RuntimeError as e:
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            logger.warning(f"GPU OOM, falling back to CPU: {e}")
            import torch
            torch.cuda.empty_cache()
            return model(img, device="cpu", verbose=False)[0]
        raise


def detect_image(image_path: Path) -> dict:
    model = load_model()
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    results = _run_inference(model, img)

    detected = []
    annotated = _draw_detections(img, results.boxes, detected)

    organik_count = sum(1 for d in detected if d["category"] == "Organik")
    non_organik_count = sum(1 for d in detected if d["category"] == "Non-Organik")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = image_path.stem
    out_filename = f"{ts}_{stem}_annotated.jpg"
    out_path = RESULT_DIR / out_filename
    cv2.imwrite(str(out_path), annotated)

    return {
        "detected_objects": detected,
        "summary": {
            "organik": organik_count,
            "non_organik": non_organik_count,
            "total": len(detected),
        },
        "result_url": f"/static/result/{out_filename}",
        "recommendation": _get_recommendation(organik_count, non_organik_count),
    }


def detect_video(video_path: Path) -> dict:
    model = load_model()
    cap = cv2.VideoCapture(str(video_path))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = video_path.stem
    out_filename = f"{ts}_{stem}_annotated.mp4"
    out_path = RESULT_DIR / out_filename
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_skip = max(1, fps // 2)
    frame_count = 0
    processed_count = 0
    all_detected = {}
    organik_total = 0
    non_organik_total = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = _run_inference_stream(model, frame)
            for r in results:
                det = []
                annotated = _draw_detections(frame, r.boxes, det)
                writer.write(annotated)
                processed_count += 1

                for d in det:
                    key = f"{d['label']}|{d['category']}"
                    if key not in all_detected:
                        all_detected[key] = {
                            "label": d["label"],
                            "category": d["category"],
                            "confidence": 0.0,
                            "count": 0,
                        }
                    all_detected[key]["count"] += 1
                    all_detected[key]["confidence"] = max(
                        all_detected[key]["confidence"], d["confidence"]
                    )

                    if d["category"] == "Organik":
                        organik_total += 1
                    else:
                        non_organik_total += 1
        else:
            writer.write(frame)

        frame_count += 1

    cap.release()
    writer.release()

    detected_objects = [
        {"label": v["label"], "category": v["category"], "confidence": round(v["confidence"], 4), "count": v["count"]}
        for v in all_detected.values()
    ]

    return {
        "detected_objects": detected_objects,
        "frames_processed": processed_count,
        "summary": {
            "organik": organik_total,
            "non_organik": non_organik_total,
            "total": organik_total + non_organik_total,
        },
        "result_url": f"/static/result/{out_filename}",
        "recommendation": _get_recommendation(organik_total, non_organik_total),
    }


def _get_recommendation(organik: int, non_organik: int) -> str:
    if organik > 0 and non_organik > 0:
        return "Pisahkan sampah sesuai kategori sebelum dibuang."
    if organik > 0:
        return "Buang ke Tempat Sampah Organik"
    if non_organik > 0:
        return "Buang ke Tempat Sampah Non-Organik"
    return "Tidak ada sampah terdeteksi."
