import json, os, time, shutil, random
from pathlib import Path
from urllib.request import urlopen, Request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

from app.core.config import BASE_DIR, ORGANIC_CATEGORIES

DATASOURCE_DIR = BASE_DIR.parent / "datasource"
ANNOTATIONS_FILE = DATASOURCE_DIR / "annotations.json"
DATASET_DIR = BASE_DIR / "dataset"
RAW_DIR = DATASET_DIR / "raw"

SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_WORKERS = 12


def _download_images():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    data = json.loads(ANNOTATIONS_FILE.read_text())
    images = data["images"]
    annotated_ids = set(a["image_id"] for a in data["annotations"])
    to_download = [i for i in images if i["id"] in annotated_ids]

    def download(img):
        iid = img["id"]
        fname = img["file_name"]
        flickr_url = img.get("flickr_url", "")
        if not flickr_url:
            return iid, "no_url"
        z_url = flickr_url.replace("_o.png", "_z.jpg")
        out_path = RAW_DIR / fname.replace("/", "_")
        if out_path.exists():
            return iid, "exists"
        for _ in range(3):
            try:
                req = Request(z_url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urlopen(req, timeout=30)
                out_path.write_bytes(resp.read())
                return iid, "ok"
            except Exception:
                time.sleep(2)
        return iid, "fail"

    success = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(download, img): img["id"] for img in to_download}
        for f in as_completed(futures):
            _, status = f.result()
            if status in ("ok", "exists"):
                success += 1
            else:
                failed += 1
    return success, failed


def _prepare_splits():
    data = json.loads(ANNOTATIONS_FILE.read_text())
    cat_map = {}
    for cat in data["categories"]:
        cat_map[cat["id"]] = 0 if cat["id"] in ORGANIC_CATEGORIES else 1
    images = {img["id"]: img for img in data["images"]}
    img_anns = defaultdict(list)
    for ann in data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    annotated_ids = list(img_anns.keys())
    random.shuffle(annotated_ids)
    n = len(annotated_ids)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])
    splits = [("train", annotated_ids[:train_end]), ("val", annotated_ids[train_end:val_end]), ("test", annotated_ids[val_end:])]

    counts = {}
    for split_name, ids in splits:
        img_dir = DATASET_DIR / split_name / "images"
        lbl_dir = DATASET_DIR / split_name / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        c = {0: 0, 1: 0}
        for img_id in ids:
            img = images[img_id]
            src_name = img["file_name"].replace("/", "_")
            src_path = RAW_DIR / src_name
            if not src_path.exists():
                # Try case-insensitive match (TACO has .jpg vs .JPG)
                candidates = list(RAW_DIR.glob(f"{src_path.stem}.*"))
                if candidates:
                    src_path = candidates[0]
                else:
                    continue
            shutil.copy2(str(src_path), str(img_dir / src_name))
            label_name = Path(src_name).stem + ".txt"
            h, w = img["height"], img["width"]
            with open(lbl_dir / label_name, "w") as f:
                for ann in img_anns[img_id]:
                    cls_id = cat_map[ann["category_id"]]
                    c[cls_id] += 1
                    x, y, bw, bh = ann["bbox"]
                    x_center = (x + bw / 2) / w
                    y_center = (y + bh / 2) / h
                    bw_norm = bw / w
                    bh_norm = bh / h
                    f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
        counts[split_name] = {"images": len(ids), "organik": c[0], "non_organik": c[1]}
    return counts


def run_download_only():
    t0 = time.time()
    success, failed = _download_images()
    return {"pipeline": "download", "duration_s": round(time.time() - t0, 2), "success": success, "failed": failed}


def run_datapreparation():
    logs = []
    t0 = time.time()

    t = time.time()
    success, failed = _download_images()
    logs.append({"step": "download", "duration_s": round(time.time() - t, 2), "success": success, "failed": failed})

    t = time.time()
    counts = _prepare_splits()
    logs.append({"step": "split", "duration_s": round(time.time() - t, 2), "counts": counts})

    return {"pipeline": "datapreparation", "total_duration_s": round(time.time() - t0, 2), "logs": logs}
