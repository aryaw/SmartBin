import json, os, shutil, random
from pathlib import Path
from collections import defaultdict
from fastapi import HTTPException

from app.config import BASE_DIR

random.seed(42)

DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"
ANNOTATIONS_FILE = BASE_DIR.parent / "datasource" / "annotations.json"

ORGANIK_IDS = {25}


def prepare_dataset(train_pct: float = 0.70, val_pct: float = 0.15, test_pct: float = 0.15):
    if not ANNOTATIONS_FILE.exists():
        raise HTTPException(400, f"Annotations not found at {ANNOTATIONS_FILE}")

    if not RAW_DIR.exists() or not any(RAW_DIR.iterdir()):
        raise HTTPException(400, f"No raw images in {RAW_DIR}. Run download first.")

    data = json.loads(ANNOTATIONS_FILE.read_text())

    cat_map = {}
    for cat in data["categories"]:
        cat_map[cat["id"]] = 0 if cat["id"] in ORGANIK_IDS else 1

    images = {img["id"]: img for img in data["images"]}
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    annotated_ids = list(img_anns.keys())
    random.shuffle(annotated_ids)

    n = len(annotated_ids)
    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)

    splits = {
        "train": annotated_ids[:train_end],
        "val": annotated_ids[train_end:val_end],
        "test": annotated_ids[val_end:],
    }

    results = {}
    for split_name, img_ids in splits.items():
        img_dir = DATASET_DIR / split_name / "images"
        label_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        org_count = 0
        non_count = 0
        img_count = 0

        for img_id in img_ids:
            img = images[img_id]
            src_name = img["file_name"].replace("/", "_")
            src_path = RAW_DIR / src_name
            if not src_path.exists():
                continue

            shutil.copy2(str(src_path), str(img_dir / src_name))
            img_count += 1

            h, w = img["height"], img["width"]
            label_name = Path(src_name).stem + ".txt"
            label_path = label_dir / label_name
            with open(label_path, "w") as f:
                for ann in img_anns[img_id]:
                    cls_id = cat_map[ann["category_id"]]
                    if cls_id == 0:
                        org_count += 1
                    else:
                        non_count += 1
                    x, y, bw, bh = ann["bbox"]
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw/w:.6f} {bh/h:.6f}\n")

        results[split_name] = {
            "images": img_count,
            "labels_organik": org_count,
            "labels_non_organik": non_count,
        }

    # Update data.yaml
    yaml_path = BASE_DIR / "data.yaml"
    yaml_path.write_text(
        f"train: dataset/train/images\n"
        f"val: dataset/val/images\n"
        f"test: dataset/test/images\n\n"
        f"nc: 2\n"
        f"names: ['Organik', 'Non-Organik']\n"
    )

    return results
