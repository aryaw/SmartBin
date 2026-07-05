import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from sklearn.model_selection import train_test_split

from app.core.config import BASE_DIR

KAGGLE_DS = "phenomsg/waste-classification"
SEED = 42


def _ensure_cv2():
    pass


def _generate_edge_polygon(img, n_points=24):
    try:
        img_np = np.array(img.convert("RGB"))
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(thresh) > 127:
            thresh = cv2.bitwise_not(thresh)
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 0.20 * h * w:
            return None
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        if len(approx) < 6:
            indices = np.linspace(0, len(largest) - 1, n_points, dtype=int)
            approx = largest[indices]
        if len(approx) > n_points:
            indices = np.linspace(0, len(approx) - 1, n_points, dtype=int)
            approx = approx[indices]
        points = []
        for pt in approx:
            px, py = pt[0] if len(pt.shape) > 1 else pt
            points.extend([max(0.0, min(1.0, float(px) / w)), max(0.0, min(1.0, float(py) / h))])
        return points
    except Exception:
        return None


def _generate_ellipse_polygon(n_points=20, randomize=True):
    cx = 0.5 + random.uniform(-0.04, 0.04) if randomize else 0.5
    cy = 0.5 + random.uniform(-0.04, 0.04) if randomize else 0.5
    rx = random.uniform(0.37, 0.46) if randomize else 0.42
    ry = random.uniform(0.37, 0.46) if randomize else 0.42
    rot = random.uniform(-0.1, 0.1) if randomize else 0
    irr = random.uniform(0.02, 0.06) if randomize else 0
    points = []
    for i in range(n_points):
        a = 2 * math.pi * i / n_points + rot
        r = 1.0 + random.uniform(-irr, irr) if irr > 0 else 1.0
        x = max(0.005, min(0.995, cx + rx * r * math.cos(a)))
        y = max(0.005, min(0.995, cy + ry * r * math.sin(a)))
        points.extend([x, y])
    return points


def _generate_rounded_rect_polygon(n_points=20, randomize=True):
    mx = random.uniform(0.06, 0.14) if randomize else 0.08
    my = random.uniform(0.06, 0.14) if randomize else 0.08
    cr = random.uniform(0.04, 0.10) if randomize else 0.06
    x_min, x_max = mx, 1.0 - mx
    y_min, y_max = my, 1.0 - my
    ppc = max(3, n_points // 4)
    points = []
    for corners in [
        [(x_max - cr, y_min + cr, -math.pi / 2, math.pi / 2)],
        [(x_max - cr, y_max - cr, 0, math.pi / 2)],
        [(x_min + cr, y_max - cr, math.pi / 2, math.pi / 2)],
        [(x_min + cr, y_min + cr, math.pi, math.pi / 2)],
    ]:
        cx, cy, a_start, a_range = corners[0]
        for i in range(ppc):
            a = a_start + a_range * i / (ppc - 1)
            x = max(0.005, min(0.995, cx + cr * math.cos(a)))
            y = max(0.005, min(0.995, cy + cr * math.sin(a)))
            points.extend([x, y])
    return points


def _generate_mask(img, randomize=True):
    poly = _generate_edge_polygon(img)
    if poly is not None:
        return poly, "edge"
    if random.random() < 0.6:
        return _generate_ellipse_polygon(randomize=randomize), "ellipse"
    return _generate_rounded_rect_polygon(randomize=randomize), "rect"


def download_and_prepare():
    _ensure_cv2()
    random.seed(SEED)
    np.random.seed(SEED)

    from kagglehub import dataset_download

    out_dir = BASE_DIR / "dataset" / "kaggle_waste"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    print("Downloading Kaggle waste-classification dataset...")
    dl_path = dataset_download(KAGGLE_DS)
    dl_path = Path(dl_path) / "waste-classification"

    CLASS_NAMES = []
    CLASS_MAP = {}
    all_paths = []

    categories = sorted(os.listdir(dl_path))
    cid = 0
    for cat in categories:
        inner = dl_path / cat / cat
        if not inner.is_dir():
            continue
        subs = sorted(os.listdir(inner))
        for sub in subs:
            sub_path = inner / sub
            if not sub_path.is_dir():
                continue
            CLASS_NAMES.append(sub)
            CLASS_MAP[sub] = cid
            for f in sorted(os.listdir(sub_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    all_paths.append((str(sub_path / f), cid, sub))
            cid += 1

    NC = len(CLASS_NAMES)
    print(f"Found {len(all_paths)} images across {NC} subcategories")

    paths = [x[0] for x in all_paths]
    cls_ids = [x[1] for x in all_paths]

    train_p, temp_p, train_id, temp_id = train_test_split(
        paths, cls_ids, test_size=0.3, random_state=SEED, stratify=cls_ids
    )
    val_p, test_p, val_id, test_id = train_test_split(
        temp_p, temp_id, test_size=0.5, random_state=SEED, stratify=temp_id
    )

    for split, split_paths, split_ids in [
        ("train", train_p, train_id),
        ("val", val_p, val_id),
        ("test", test_p, test_id),
    ]:
        img_dir = out_dir / split / "images"
        lbl_dir = out_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {"edge": 0, "fallback": 0, "total": 0}
    for split_name, split_paths, split_ids in [
        ("train", train_p, train_id),
        ("val", val_p, val_id),
        ("test", test_p, test_id),
    ]:
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        do_rand = split_name == "train"

        for i, (img_path, cls_id) in enumerate(zip(split_paths, split_ids)):
            try:
                with Image.open(img_path) as img:
                    if img.mode in ("RGBA", "P", "LA", "L"):
                        img = img.convert("RGB")
                    ext = Path(img_path).suffix.lower()
                    if ext not in (".jpg", ".jpeg", ".png"):
                        ext = ".jpg"
                    new_name = f"{split_name}_{i:05d}{ext}"
                    img.save(str(img_dir / new_name), quality=95)

                    poly, poly_type = _generate_mask(img, randomize=do_rand)
                    if poly_type == "edge":
                        stats["edge"] += 1
                    else:
                        stats["fallback"] += 1
                    stats["total"] += 1

                    coords = " ".join(f"{v:.6f}" for v in poly)
                    lbl_path = lbl_dir / f"{Path(new_name).stem}.txt"
                    lbl_path.write_text(f"{cls_id} {coords}\n")
            except Exception as e:
                pass

    data_yaml = {
        "path": str(out_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": NC,
        "names": {i: n for i, n in enumerate(CLASS_NAMES)},
    }
    yaml_path = out_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    names_path = out_dir / "class_names.json"
    with open(names_path, "w") as f:
        json.dump({"nc": NC, "names": CLASS_NAMES}, f, indent=2)

    print(f"Done. {stats['total']} images processed ({stats['edge']} edge, {stats['fallback']} fallback)")
    return {
        "pipeline": "kaggle_download",
        "total_images": stats["total"],
        "edge_masks": stats["edge"],
        "fallback_masks": stats["fallback"],
        "classes": NC,
        "splits": {
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
        },
        "output_dir": str(out_dir),
    }


def prepare_from_local(source_dir, output_dir=None):
    """Read Waste_Classification_Dataset from local path, generate YOLO-seg masks, split."""
    _ensure_cv2()
    random.seed(SEED)
    np.random.seed(SEED)

    source_path = Path(source_dir)
    if not source_path.is_dir():
        raise FileNotFoundError(f"Source not found: {source_dir}")

    out_dir = Path(output_dir or BASE_DIR / "dataset" / "kaggle_waste")
    if out_dir.exists():
        shutil.rmtree(out_dir)

    CLASS_NAMES = []
    CLASS_MAP = {}
    all_paths = []

    categories = sorted(os.listdir(source_path))
    cid = 0
    for cat in categories:
        inner = source_path / cat / cat
        if not inner.is_dir():
            inner = source_path / cat
            if not inner.is_dir():
                continue
        subs = sorted(os.listdir(inner))
        for sub in subs:
            sub_path = inner / sub
            if not sub_path.is_dir():
                continue
            CLASS_NAMES.append(sub)
            CLASS_MAP[sub] = cid
            for f in sorted(os.listdir(sub_path)):
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                    all_paths.append((str(sub_path / f), cid, sub))
            cid += 1

    ORGANIC_SUBS = {"coffee_tea_bags", "egg_shells", "food_scraps", "kitchen_waste", "yard_trimmings"}
    BIN_MAP = {i: 0 if CLASS_NAMES[i] in ORGANIC_SUBS else 1 for i in range(len(CLASS_NAMES))}

    NC = len(CLASS_NAMES)
    if NC == 0:
        raise ValueError(f"No subcategories found in {source_dir}")

    print(f"Found {len(all_paths)} images across {NC} subcategories from {source_dir}")

    paths = [x[0] for x in all_paths]
    cls_ids = [x[1] for x in all_paths]

    train_p, temp_p, train_id, temp_id = train_test_split(
        paths, cls_ids, test_size=0.3, random_state=SEED, stratify=cls_ids
    )
    val_p, test_p, val_id, test_id = train_test_split(
        temp_p, temp_id, test_size=0.5, random_state=SEED, stratify=temp_id
    )

    for split in ("train", "val", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"edge": 0, "fallback": 0, "total": 0}
    for split_name, split_paths, split_ids in [
        ("train", train_p, train_id),
        ("val", val_p, val_id),
        ("test", test_p, test_id),
    ]:
        img_dir = out_dir / split_name / "images"
        lbl_dir = out_dir / split_name / "labels"
        do_rand = split_name == "train"

        for i, (img_path, cls_id) in enumerate(zip(split_paths, split_ids)):
            try:
                with Image.open(img_path) as img:
                    if img.mode in ("RGBA", "P", "LA", "L"):
                        img = img.convert("RGB")
                    ext = Path(img_path).suffix.lower()
                    if ext not in (".jpg", ".jpeg", ".png"):
                        ext = ".jpg"
                    new_name = f"{split_name}_{i:05d}{ext}"
                    img.save(str(img_dir / new_name), quality=95)

                    poly, poly_type = _generate_mask(img, randomize=do_rand)
                    if poly_type == "edge":
                        stats["edge"] += 1
                    else:
                        stats["fallback"] += 1
                    stats["total"] += 1

                    coords = " ".join(f"{v:.6f}" for v in poly)
                    lbl_path = lbl_dir / f"{Path(new_name).stem}.txt"
                    lbl_path.write_text(f"{BIN_MAP[cls_id]} {coords}\n")
            except Exception:
                pass

    data_yaml = {
        "path": str(out_dir),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": {0: "Organik", 1: "Non-Organik"},
    }
    with open(out_dir / "data.yaml", "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    BIN_CLASS_NAMES = ["Organik", "Non-Organik"]
    with open(out_dir / "class_names.json", "w") as f:
        json.dump({"nc": 2, "names": BIN_CLASS_NAMES}, f, indent=2)

    print(f"Done. {stats['total']} images processed ({stats['edge']} edge, {stats['fallback']} fallback)")
    return {
        "pipeline": "local_prepare",
        "source": str(source_dir),
        "total_images": stats["total"],
        "edge_masks": stats["edge"],
        "fallback_masks": stats["fallback"],
        "classes": 2,
        "splits": {
            "train": len(train_p),
            "val": len(val_p),
            "test": len(test_p),
        },
        "output_dir": str(out_dir),
    }


if __name__ == "__main__":
    result = download_and_prepare()
    print(json.dumps(result, indent=2))
