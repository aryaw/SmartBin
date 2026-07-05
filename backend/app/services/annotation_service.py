import json, time, os
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont

from app.core.config import BASE_DIR, ORGANIC_CATEGORIES
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
