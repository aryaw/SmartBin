import json, math, time, os
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import BASE_DIR, ORGANIC_CATEGORIES, VIZ_DIR
from app.utils.gpu_utils import try_log_gpu

DATASET_DIR = BASE_DIR / "dataset"
ANNOTATIONS_FILE = BASE_DIR.parent / "datasource" / "annotations.json"

ORGANIK_COLOR = (34, 139, 34)
NON_ORGANIK_COLOR = (200, 50, 50)
LABELS = {0: "Organik", 1: "Non-Organik"}


def _export_coco():
    train_img_dir = DATASET_DIR / "train" / "images"
    coco_out_dir = DATASET_DIR / "coco_gt_bbox"
    coco_out_dir.mkdir(parents=True, exist_ok=True)

    data = json.loads(ANNOTATIONS_FILE.read_text())

    img_lookup = {}
    for img in data["images"]:
        key = img["file_name"].replace("/", "_")
        key_lower = key.lower()
        if key_lower in img_lookup:
            key_lower = f"{key_lower}_{img['id']}"
        img_lookup[key_lower] = img

    train_keys = set(f.name.lower() for f in train_img_dir.iterdir()
                     if f.suffix.lower() in {".jpg", ".jpeg", ".png"})

    matched_ids = set()
    matched_files = []
    for key in train_keys:
        img = img_lookup.get(key)
        if img:
            matched_ids.add(img["id"])
            matched_files.append((key, img))

    filtered = {
        "info": data["info"],
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": [img for img in data["images"] if img["id"] in matched_ids],
        "annotations": [ann for ann in data["annotations"] if ann["image_id"] in matched_ids],
    }

    (coco_out_dir / "train_annotations.json").write_text(json.dumps(filtered, indent=2))

    # Build annotation index per image (with coords scaled to actual image size)
    ann_idx = defaultdict(list)
    for ann in filtered["annotations"]:
        ann_idx[ann["image_id"]].append(ann)

    # Write per-image YOLO annotation .txt (normalized x_center y_center width height)
    ann_count = 0
    for key, img in matched_files:
        fpath = train_img_dir / img["file_name"].replace("/", "_")
        if not fpath.exists() or fpath.stat().st_size == 0:
            continue
        actual = Image.open(fpath)
        aw, ah = actual.size
        actual.close()
        sx = aw / img["width"]
        sy = ah / img["height"]

        anns = ann_idx.get(img["id"], [])
        if not anns:
            continue
        out_txt = coco_out_dir / f"{Path(fpath.name).stem}.txt"
        with open(out_txt, "w") as f:
            for ann in anns:
                cat_map = 0 if ann["category_id"] in ORGANIC_CATEGORIES else 1
                x, y, bw, bh = ann["bbox"]
                x_center = ((x + bw / 2) * sx) / aw
                y_center = ((y + bh / 2) * sy) / ah
                w_norm = (bw * sx) / aw
                h_norm = (bh * sy) / ah
                f.write(f"{cat_map} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                ann_count += 1

    return {
        "train_images": len(matched_ids),
        "coco_annotations": len(filtered["annotations"]),
        "annotation_files": ann_count,
    }


def _export_coco_viz():
    train_img_dir = DATASET_DIR / "train" / "images"
    coco_ann_dir = DATASET_DIR / "coco_gt_bbox"
    viz_dir = DATASET_DIR / "coco_gt_bbox_img"
    viz_dir.mkdir(parents=True, exist_ok=True)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except Exception:
        font = ImageFont.load_default()

    WHITE = (255, 255, 255)

    ok = 0
    for f in sorted(train_img_dir.iterdir()):
        if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        if f.stat().st_size == 0:
            continue
        stem = f.stem
        ann_file = coco_ann_dir / f"{stem}.txt"
        if not ann_file.exists():
            continue

        img = Image.open(f).convert("RGB")
        draw = ImageDraw.Draw(img)
        for idx, line in enumerate(ann_file.read_text().strip().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            cls_id = int(parts[0])
            x_center, y_center, w_norm, h_norm = map(float, parts[1:])
            img_w, img_h = img.size
            x1 = (x_center - w_norm / 2) * img_w
            y1 = (y_center - h_norm / 2) * img_h
            x2 = (x_center + w_norm / 2) * img_w
            y2 = (y_center + h_norm / 2) * img_h
            color = ORGANIK_COLOR if cls_id == 0 else NON_ORGANIK_COLOR
            label = f"#{idx} {LABELS.get(cls_id, 'Unknown')}"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            bbox = draw.textbbox((int(x1) + 2, int(y1) - 18), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((int(x1) + 2, int(y1) - 18), label, fill=WHITE, font=font)

        img.save(str(viz_dir / f"{stem}.jpg"), "JPEG")
        ok += 1

    return {"viz_images": ok}


def visualize_pseudo_mask_pipeline(sample_path=None, output_dir=None):
    """Generate pseudo-mask pipeline visualization showing edge detection steps."""
    if output_dir is None:
        output_dir = VIZ_DIR / "pseudo_mask_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_path is None:
        sample_path = BASE_DIR.parent / "waste_datasource" / "inferencedata" / "batch_1_000000.jpg"

    img = cv2.imread(str(sample_path))
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
    img = cv2.resize(img, (640, 640))
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mean_val = np.mean(thresh)
    if mean_val > 127:
        thresh = cv2.bitwise_not(thresh)
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img.copy()
    if contours:
        largest = max(contours, key=cv2.contourArea)
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        cv2.drawContours(result_img, [approx], -1, (0, 255, 0), 3)

    steps = {
        "01_original": img,
        "02_grayscale": cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        "03_gaussian_blur": cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        "04_otsu_threshold": cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR),
        "05_morph_close": cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR),
        "06_contour_result": result_img,
    }
    saved = []
    for name, step_img in steps.items():
        path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), step_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(str(path))
    return {"saved": len(saved), "output_dir": str(output_dir), "files": saved}


def visualize_augmentation_pipeline(sample_path=None, output_dir=None):
    """Generate augmentation visualization samples showing effect of each augmentation."""
    if output_dir is None:
        output_dir = VIZ_DIR / "augmentation_demo"
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_path is None:
        sample_path = BASE_DIR.parent / "waste_datasource" / "inferencedata" / "batch_1_000000.jpg"

    img = cv2.imread(str(sample_path))
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {sample_path}")
    img = cv2.resize(img, (640, 640))

    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 18) % 180
    hsv_hue = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.8
    hsv_sat = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    h, w = img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w // 2, h // 2), 15, 1.0)
    rot = cv2.warpAffine(img, M_rot, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    flip_h = cv2.flip(img, 1)

    M_scale = cv2.getRotationMatrix2D((w // 2, h // 2), 0, 0.5)
    scaled = cv2.warpAffine(img, M_scale, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    shear_m = np.float32([[1, math.tan(math.radians(5)), 0], [0, 1, 0]])
    sheared = cv2.warpAffine(img, shear_m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    augs = {
        "01_original": img,
        "02_hsv_hue": hsv_hue,
        "03_hsv_saturation": hsv_sat,
        "04_rotate_15deg": rot,
        "05_flip_horizontal": flip_h,
        "06_scale_50pct": scaled,
        "07_shear_5deg": sheared,
    }
    saved = []
    for name, aug_img in augs.items():
        path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(str(path))
    return {"saved": len(saved), "output_dir": str(output_dir), "files": saved}


def run_coco_pipeline(progress_callback=None):
    logs = []
    t0 = time.time()

    if progress_callback:
        progress_callback({"step": "coco", "status": "start", "message": "Exporting COCO annotations..."})
    t = time.time()
    r1 = _export_coco()
    logs.append({"step": "export_coco", "duration_s": round(time.time() - t, 2), **r1, **try_log_gpu()})
    if progress_callback:
        progress_callback({"step": "coco", "status": "done", "message": "COCO annotations exported"})

    if progress_callback:
        progress_callback({"step": "coco_viz", "status": "start", "message": "Rendering COCO visualizations..."})
    t = time.time()
    r2 = _export_coco_viz()
    logs.append({"step": "export_coco_viz", "duration_s": round(time.time() - t, 2), **r2, **try_log_gpu()})
    if progress_callback:
        progress_callback({"step": "coco_viz", "status": "done", "message": "COCO visualizations rendered"})

    return {
        "pipeline": "coco",
        "total_duration_s": round(time.time() - t0, 2),
        "logs": logs,
    }
