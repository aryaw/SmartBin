import os, time, json
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, ImageDraw, ImageFont

from app.core.config import BASE_DIR, MODEL_PATH
from app.services.detector import load_model
from app.utils.gpu_utils import get_device, try_log_gpu

DATASET_DIR = BASE_DIR / "dataset"
BATCH_SIZE = int(os.environ.get("YOLO_BATCH_SIZE", "16"))
MAX_IMAGES = int(os.environ.get("YOLO_MAX_IMAGES", "0"))

ORGANIK_COLOR = (34, 139, 34)   # green
NON_ORGANIK_COLOR = (200, 50, 50)  # red
LABELS = {0: "Organik", 1: "Non-Organik"}


def _run_yolo_inference(img_dir=None, out_dir=None):
    img_dir = img_dir or (DATASET_DIR / "train" / "images")
    yolo_out_dir = out_dir or (DATASET_DIR / "yolo_train_bbox")
    yolo_out_dir.mkdir(parents=True, exist_ok=True)

    model = load_model()
    device = get_device()

    total_pred = 0
    img_count = 0
    organik_total = 0
    non_organik_total = 0

    files = []
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if f.stat().st_size == 0:
            continue
        files.append(f)

    if MAX_IMAGES > 0:
        files = files[:MAX_IMAGES]

    for idx, f in enumerate(files):
        results = model(str(f), device=device, verbose=False, imgsz=640, half=True)[0]
        stem = f.stem
        out_path = yolo_out_dir / f"{stem}.txt"

        org = 0
        non = 0
        lines = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            lines.append(f"{cls_id} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")
            if cls_id == 0:
                org += 1
            else:
                non += 1

        if lines:
            out_path.write_text("".join(lines))
            total_pred += len(lines)
            organik_total += org
            non_organik_total += non
            img_count += 1

        if (idx + 1) % BATCH_SIZE == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "images_processed": img_count,
        "total_predictions": total_pred,
        "organik": organik_total,
        "non_organik": non_organik_total,
    }


def _export_yolo_viz(img_dir=None, pred_dir=None, out_viz_dir=None):
    img_dir = img_dir or (DATASET_DIR / "train" / "images")
    yolo_pred_dir = pred_dir or (DATASET_DIR / "yolo_train_bbox")
    viz_dir = out_viz_dir or (DATASET_DIR / "yolo_train_bbox_img")
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    WHITE = (255, 255, 255)

    ok = 0
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if f.stat().st_size == 0:
            continue
        stem = f.stem
        pred_file = yolo_pred_dir / f"{stem}.txt"
        if not pred_file.exists():
            continue

        img = Image.open(f).convert("RGB")
        draw = ImageDraw.Draw(img)
        for idx, line in enumerate(pred_file.read_text().strip().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            conf = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:])
            color = ORGANIK_COLOR if cls_id == 0 else NON_ORGANIK_COLOR
            label = f"#{idx} Cls:{cls_id} {LABELS.get(cls_id, 'Unknown')} {conf:.0%}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            bbox = draw.textbbox((int(x1) + 2, int(y1) - 18), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((int(x1) + 2, int(y1) - 18), label, fill=WHITE, font=font)

        img.save(str(viz_dir / f"{stem}.jpg"), "JPEG")
        ok += 1

    return {"viz_images": ok}


def _clean_outputs():
    import shutil
    for d in [DATASET_DIR / "yolo_train_bbox", DATASET_DIR / "yolo_train_bbox_img"]:
        if d.exists():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)


def run_yolo_pipeline(progress_callback=None):
    logs = []
    t0 = time.time()

    if progress_callback:
        progress_callback({"step": "yolo_inference", "status": "start", "message": "Running YOLO inference on train images..."})
    t = time.time()
    r1 = _run_yolo_inference()
    logs.append({"step": "yolo_inference_train", "duration_s": round(time.time() - t, 2), **r1, **try_log_gpu()})
    if progress_callback:
        progress_callback({"step": "yolo_inference", "status": "done", "message": "YOLO inference complete"})

    if progress_callback:
        progress_callback({"step": "yolo_viz", "status": "start", "message": "Rendering YOLO visualizations..."})
    t = time.time()
    r2 = _export_yolo_viz()
    logs.append({"step": "yolo_viz", "duration_s": round(time.time() - t, 2), **r2, **try_log_gpu()})
    if progress_callback:
        progress_callback({"step": "yolo_viz", "status": "done", "message": "YOLO visualizations rendered"})

    return {
        "pipeline": "yolo",
        "total_duration_s": round(time.time() - t0, 2),
        "logs": logs,
    }


def run_yolo_val_pipeline():
    val_img_dir = DATASET_DIR / "val" / "images"
    val_ann_dir = DATASET_DIR / "yolo_val_bbox"
    val_viz_dir = DATASET_DIR / "yolo_val_bbox_img"
    logs = []
    t0 = time.time()

    t = time.time()
    r1 = _run_yolo_inference(img_dir=val_img_dir, out_dir=val_ann_dir)
    logs.append({"step": "yolo_inference_val", "duration_s": round(time.time() - t, 2), **r1, **try_log_gpu()})

    t = time.time()
    r2 = _export_yolo_viz(img_dir=val_img_dir, pred_dir=val_ann_dir, out_viz_dir=val_viz_dir)
    logs.append({"step": "yolo_viz_val", "duration_s": round(time.time() - t, 2), **r2, **try_log_gpu()})

    return {
        "pipeline": "yolo_val",
        "total_duration_s": round(time.time() - t0, 2),
        "logs": logs,
    }


def _box_to_polygon(cls_id, x1, y1, x2, y2):
    return f"{cls_id} {x1:.6f} {y1:.6f} {x2:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x1:.6f} {y2:.6f}\n"

def run_yolo_seg_pipeline():
    img_dir = DATASET_DIR / "train" / "images"
    ann_dir = DATASET_DIR / "yolo_train_seg"
    viz_dir = DATASET_DIR / "yolo_train_seg_img"
    logs = []
    t0 = time.time()

    ann_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    from PIL import Image, ImageDraw, ImageFont
    from app.core.config import MODEL_PATH
    from ultralytics import YOLO
    import torch

    model = YOLO(str(MODEL_PATH))
    device = get_device()

    total_pred = 0
    img_count = 0
    files = [f for f in sorted(img_dir.iterdir())
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.stat().st_size > 0]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    ORGANIK_COLOR = (34, 139, 34)
    NON_ORGANIK_COLOR = (200, 50, 50)
    LABELS = {0: "Organik", 1: "Non-Organik"}

    for idx, f in enumerate(files):
        results = model(str(f), device=device, verbose=False, imgsz=640, half=True, retina_masks=True)[0]
        stem = f.stem
        out_path = ann_dir / f"{stem}.txt"
        img_w, img_h = results.orig_shape[1], results.orig_shape[0]

        lines = []
        if results.masks is not None:
            cls_ids = results.boxes.cls.int().tolist() if results.boxes is not None else []
            segments = results.masks.xyn
            for i, seg in enumerate(segments):
                cls_id = cls_ids[i] if i < len(cls_ids) else 0
                pts = seg.flatten().tolist()
                if len(pts) < 4:
                    continue
                lines.append(f"{cls_id} " + " ".join(f"{p:.6f}" for p in pts) + "\n")
        elif results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxyn[0].tolist()
                lines.append(_box_to_polygon(cls_id, x1, y1, x2, y2))

        if lines:
            out_path.write_text("".join(lines))
            total_pred += len(lines)
            img_count += 1

            img = Image.open(f).convert("RGB")
            draw = ImageDraw.Draw(img)
            for li, line in enumerate(lines, 1):
                parts = line.strip().split()
                cls_id = int(parts[0])
                pts = list(map(float, parts[1:]))
                color = ORGANIK_COLOR if cls_id == 0 else NON_ORGANIK_COLOR
                polygon = [(pts[i] * img_w, pts[i + 1] * img_h) for i in range(0, len(pts), 2)]
                draw.polygon(polygon, outline=color, width=2)
                label = f"#{li} {LABELS.get(cls_id, 'Unknown')}"
                tx, ty = polygon[0]
                bbox = draw.textbbox((int(tx) + 2, int(ty) - 18), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((int(tx) + 2, int(ty) - 18), label, fill=(255, 255, 255), font=font)
            img.save(str(viz_dir / f"{stem}.jpg"), "JPEG")

        if (idx + 1) % 16 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    r1 = {"images_processed": img_count, "total_predictions": total_pred}
    logs.append({"step": "yolo_seg_inference", "duration_s": round(time.time() - t, 2), **r1, **try_log_gpu()})

    return {
        "pipeline": "yolo_seg",
        "total_duration_s": round(time.time() - t0, 2),
        "logs": logs,
        "processed": img_count,
        "predictions": total_pred,
    }


def run_yolo_test_pipeline():
    test_img_dir = DATASET_DIR / "test" / "images"
    test_ann_dir = DATASET_DIR / "yolo_test_bbox"
    test_viz_dir = DATASET_DIR / "yolo_test_bbox_img"
    logs = []
    t0 = time.time()

    test_ann_dir.mkdir(parents=True, exist_ok=True)
    test_viz_dir.mkdir(parents=True, exist_ok=True)

    t = time.time()
    from PIL import Image, ImageDraw, ImageFont
    from app.core.config import MODEL_PATH
    from ultralytics import YOLO
    import torch

    model = YOLO(str(MODEL_PATH))
    device = get_device()

    total_pred = 0
    img_count = 0
    files = [f for f in sorted(test_img_dir.iterdir())
             if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.stat().st_size > 0]

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    WHITE = (255, 255, 255)
    ORGANIK_COLOR = (34, 139, 34)
    NON_ORGANIK_COLOR = (200, 50, 50)
    LABELS = {0: "Organik", 1: "Non-Organik"}

    for idx, f in enumerate(files):
        results = model(str(f), device=device, verbose=False, imgsz=640, half=True)[0]
        stem = f.stem
        out_path = test_ann_dir / f"{stem}.txt"

        lines = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            lines.append(f"{cls_id} {conf:.4f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}\n")

        if lines:
            out_path.write_text("".join(lines))
            total_pred += len(lines)
            img_count += 1

            img = Image.open(f).convert("RGB")
            draw = ImageDraw.Draw(img)
            for li, line in enumerate(lines, 1):
                parts = line.split()
                cls_id = int(parts[0])
                conf = float(parts[1])
                x1, y1, x2, y2 = map(float, parts[2:])
                color = ORGANIK_COLOR if cls_id == 0 else NON_ORGANIK_COLOR
                label = f"#{li} {LABELS.get(cls_id, 'Unknown')} {conf:.0%}"
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                bbox = draw.textbbox((int(x1) + 2, int(y1) - 18), label, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((int(x1) + 2, int(y1) - 18), label, fill=WHITE, font=font)
            img.save(str(test_viz_dir / f"{stem}.jpg"), "JPEG")

        if (idx + 1) % 16 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    r1 = {"images_processed": img_count, "total_predictions": total_pred}
    logs.append({"step": "yolo_inference_test", "duration_s": round(time.time() - t, 2), **r1, **try_log_gpu()})

    return {
        "pipeline": "yolo_test",
        "total_duration_s": round(time.time() - t0, 2),
        "logs": logs,
        "processed": img_count,
        "predictions": total_pred,
    }
