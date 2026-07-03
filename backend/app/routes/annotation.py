import asyncio
import json
import random
import shutil
import time

import torch
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from PIL import Image
from ultralytics import YOLO

from app.config import BASE_DIR, MODEL_PATH
from app.services.annotation_service import run_coco_pipeline
from app.services.yolo_service import run_yolo_pipeline, run_yolo_val_pipeline, run_yolo_seg_pipeline
from app.utils.gpu_utils import get_device

router = APIRouter(prefix="/api/dataset", tags=["Dataset"])

DATASET_DIR = BASE_DIR / "dataset"
ANNOTATIONS_FILE = BASE_DIR.parent / "datasource" / "annotations.json"
CONVERT_ANN_DIR = DATASET_DIR / "coco_gt_bbox"
CONVERT_VIZ_DIR = DATASET_DIR / "coco_gt_bbox_img"
REAL_ANN_DIR = DATASET_DIR / "yolo_train_bbox"
REAL_VIZ_DIR = DATASET_DIR / "yolo_train_bbox_img"
VAL_REAL_ANN_DIR = DATASET_DIR / "yolo_val_bbox"
VAL_REAL_VIZ_DIR = DATASET_DIR / "yolo_val_bbox_img"
TEST_REAL_ANN_DIR = DATASET_DIR / "yolo_test_bbox"
TEST_REAL_VIZ_DIR = DATASET_DIR / "yolo_test_bbox_img"
SEG_ANN_DIR = DATASET_DIR / "yolo_train_seg"
SEG_VIZ_DIR = DATASET_DIR / "yolo_train_seg_img"
TRAIN_IMG_DIR = DATASET_DIR / "train" / "images"
TRAIN_LBL_DIR = DATASET_DIR / "train" / "labels"
VAL_IMG_DIR = DATASET_DIR / "val" / "images"
VAL_LBL_DIR = DATASET_DIR / "val" / "labels"
TEST_IMG_DIR = DATASET_DIR / "test" / "images"
TEST_LBL_DIR = DATASET_DIR / "test" / "labels"
RAW_DIR = DATASET_DIR / "raw"


@router.get("/grid")
async def dataset_grid():
    raw_images = []
    if RAW_DIR.exists():
        for f in sorted(RAW_DIR.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                raw_images.append({
                    "filename": f.name,
                    "url": f"/api/dataset/file/raw/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    coco_annotations = {}
    if ANNOTATIONS_FILE.exists():
        data = json.loads(ANNOTATIONS_FILE.read_text())
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in coco_annotations:
                coco_annotations[img_id] = []
            x, y, w, h = ann["bbox"]
            cat_map = 0 if ann["category_id"] == 25 else 1
            coco_annotations[img_id].append({
                "class_id": cat_map,
                "category": "Organik" if cat_map == 0 else "Non-Organik",
                "bbox": [round(v, 2) for v in [x, y, w, h]],
            })
        img_lookup = {img["id"]: img for img in data.get("images", [])}

    def _build_coco_for_split(img_dir, source_name, viz_dir=None):
        items = []
        if not img_dir.exists():
            return items
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = f.stem
            img_id = None
            anns = []
            viz_url = None
            if viz_dir:
                viz_file = viz_dir / f"{stem}.jpg"
                if viz_file.exists():
                    viz_url = f"/api/dataset/file/{viz_dir.name}/{stem}.jpg"
            if ANNOTATIONS_FILE.exists():
                key = f"{stem}.jpg"
                for iid, img in img_lookup.items():
                    if img["file_name"].replace("/", "_").lower() == key.lower():
                        img_id = iid
                        break
                if img_id and img_id in coco_annotations:
                    anns = coco_annotations[img_id]
            items.append({
                "filename": f.name,
                "image_url": f"/api/dataset/file/{source_name}/{f.name}",
                "annotations": anns,
                "viz_url": viz_url,
                "annotation_count": len(anns),
            })
        return items

    coco_images = _build_coco_for_split(TRAIN_IMG_DIR, "train", CONVERT_VIZ_DIR)

    def scan_annotations(img_dir, source_name, ann_dir, viz_dir, has_conf=False, is_yolo=False, is_seg=False):
        items = []
        if not img_dir.exists():
            return items
        for f in sorted(img_dir.iterdir()):
            if f.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = f.stem
            preds = []
            ann_file = ann_dir / f"{stem}.txt"
            if ann_file.exists():
                for line in ann_file.read_text().strip().splitlines():
                    parts = line.split()
                    if is_yolo:
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        cx, cy, w_n, h_n = map(float, parts[1:5])
                        iw, ih = Image.open(f).size
                        x1 = (cx - w_n / 2) * iw
                        y1 = (cy - h_n / 2) * ih
                        x2 = (cx + w_n / 2) * iw
                        y2 = (cy + h_n / 2) * ih
                        entry = {
                            "class_id": cls_id,
                            "category": "Organik" if cls_id == 0 else "Non-Organik",
                            "bbox": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                            "yolo_bbox": [round(cx, 6), round(cy, 6), round(w_n, 6), round(h_n, 6)],
                        }
                    elif has_conf:
                        if len(parts) < 6:
                            continue
                        cls_id = int(parts[0])
                        conf = float(parts[1])
                        x1, y1, x2, y2 = map(float, parts[2:6])
                        entry = {
                            "class_id": cls_id,
                            "category": "Organik" if cls_id == 0 else "Non-Organik",
                            "bbox": [round(v, 2) for v in [x1, y1, x2, y2]],
                            "confidence": round(conf, 4),
                        }
                    elif is_seg:
                        if len(parts) < 5:
                            continue
                        cls_id = int(parts[0])
                        seg_pts = list(map(float, parts[1:]))
                        entry = {
                            "class_id": cls_id,
                            "category": "Organik" if cls_id == 0 else "Non-Organik",
                            "segmentation": [round(v, 6) for v in seg_pts],
                        }
                    else:
                        continue
                    preds.append(entry)
            viz_file = viz_dir / f"{stem}.jpg"
            items.append({
                "filename": f.name,
                "image_url": f"/api/dataset/file/{source_name}/{f.name}",
                "predictions": preds,
                "viz_url": f"/api/dataset/file/{viz_dir.name}/{stem}.jpg" if viz_file.exists() else None,
                "prediction_count": len(preds),
            })
        return items

    train_convert = scan_annotations(TRAIN_IMG_DIR, "train", CONVERT_ANN_DIR, CONVERT_VIZ_DIR, is_yolo=True)
    train_real = scan_annotations(TRAIN_IMG_DIR, "train", REAL_ANN_DIR, REAL_VIZ_DIR, has_conf=True)
    train_seg = scan_annotations(TRAIN_IMG_DIR, "train", SEG_ANN_DIR, SEG_VIZ_DIR, is_seg=True)
    val_coco = _build_coco_for_split(VAL_IMG_DIR, "val")
    val_real = scan_annotations(VAL_IMG_DIR, "val", VAL_REAL_ANN_DIR, VAL_REAL_VIZ_DIR, has_conf=True)
    test_real = scan_annotations(TEST_IMG_DIR, "test", TEST_REAL_ANN_DIR, TEST_REAL_VIZ_DIR, has_conf=True)

    test_images = []
    if TEST_IMG_DIR.exists():
        for f in sorted(TEST_IMG_DIR.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                test_images.append({
                    "filename": f.name,
                    "url": f"/api/dataset/file/test/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    train_images = []
    if TRAIN_IMG_DIR.exists():
        for f in sorted(TRAIN_IMG_DIR.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                train_images.append({
                    "filename": f.name,
                    "url": f"/api/dataset/file/train/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    val_images = []
    if VAL_IMG_DIR.exists():
        for f in sorted(VAL_IMG_DIR.iterdir()):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                val_images.append({
                    "filename": f.name,
                    "url": f"/api/dataset/file/val/{f.name}",
                    "size_kb": round(f.stat().st_size / 1024, 1),
                })

    return {
        "raw": raw_images,
        "coco": coco_images,
        "train_convert": train_convert,
        "train_real": train_real,
        "train_seg": train_seg,
        "val_coco": val_coco,
        "val_real": val_real,
        "test_real": test_real,
        "test": test_images,
        "train_plain": train_images,
        "val_plain": val_images,
        "stats": {
            "raw": len(raw_images),
            "train": len(train_images),
            "val": len(val_images),
            "test": len(test_images),
            "coco": len([c for c in coco_images if c["annotation_count"] > 0]),
            "coco_total": len(coco_images),
            "train_convert": len([x for x in train_convert if x["prediction_count"] > 0]),
            "train_real": len([x for x in train_real if x["prediction_count"] > 0]),
            "train_seg": len([x for x in train_seg if x["prediction_count"] > 0]),
            "val_coco": len([x for x in val_coco if x["annotation_count"] > 0]),
            "val_real": len([x for x in val_real if x["prediction_count"] > 0]),
            "test_real": len([x for x in test_real if x["prediction_count"] > 0]),
        },
    }


@router.get("/evaluate")
async def dataset_evaluate(split: str = Query("all", regex="^(train|val|test|all)$")):
    splits = ["train", "val", "test"] if split == "all" else [split]
    device = get_device()

    def _eval(s: str) -> dict:
        model = YOLO(str(MODEL_PATH))
        results = model.val(
            data=str(BASE_DIR / "data.yaml"),
            split=s,
            device=device,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            verbose=False,
        )
        d = results.results_dict
        per_class = []
        if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
            cls_names = model.names if hasattr(model, "names") else {}
            for i, c in enumerate(results.box.ap_class_index):
                name_cls = cls_names.get(int(c), str(c))
                ap = results.box.ap[i] if hasattr(results.box, "ap") and i < len(results.box.ap) else 0
                per_class.append({
                    "class_id": int(c),
                    "name": name_cls,
                    "mAP50": round(ap * 100, 2),
                })
        return {
            "mAP50": round(d.get("metrics/mAP50(B)", 0) * 100, 2),
            "mAP50_95": round(d.get("metrics/mAP50-95(B)", 0) * 100, 2),
            "precision": round(d.get("metrics/precision(B)", 0) * 100, 2),
            "recall": round(d.get("metrics/recall(B)", 0) * 100, 2),
            "per_class": per_class,
        }

    loop = asyncio.get_running_loop()
    metrics = {}
    for s in splits:
        metrics[s] = await loop.run_in_executor(None, _eval, s)

    return {"metrics": metrics}


@router.get("/file/{source:path}/{filename}")
async def dataset_file(source: str, filename: str):
    base_map = {
        "raw": RAW_DIR,
        "train": TRAIN_IMG_DIR,
        "val": VAL_IMG_DIR,
        "test": TEST_IMG_DIR,
        "coco_gt_bbox_img": CONVERT_VIZ_DIR,
        "yolo_train_bbox_img": REAL_VIZ_DIR,
        "yolo_val_bbox_img": VAL_REAL_VIZ_DIR,
        "yolo_test_bbox_img": TEST_REAL_VIZ_DIR,
        "yolo_train_seg_img": SEG_VIZ_DIR,
        "yolo_train_seg_ann": SEG_ANN_DIR,
    }
    dir_path = base_map.get(source)
    if not dir_path or not dir_path.exists():
        raise HTTPException(404, "Source not found")
    file_path = dir_path / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


@router.post("/prepare")
async def dataset_prepare():
    try:
        from app.services.datapreparation_service import run_datapreparation
        return run_datapreparation()
    except Exception as e:
        raise HTTPException(500, f"Data preparation failed: {str(e)}")


@router.post("/pipeline/coco")
async def pipeline_coco():
    try:
        return run_coco_pipeline()
    except Exception as e:
        raise HTTPException(500, f"COCO pipeline failed: {str(e)}")


@router.post("/pipeline/yolo")
async def pipeline_yolo():
    try:
        return run_yolo_pipeline()
    except Exception as e:
        raise HTTPException(500, f"YOLO pipeline failed: {str(e)}")


@router.post("/pipeline/yolo/val")
async def pipeline_yolo_val():
    try:
        return run_yolo_val_pipeline()
    except Exception as e:
        raise HTTPException(500, f"YOLO val pipeline failed: {str(e)}")


@router.post("/pipeline/yolo/seg")
async def pipeline_yolo_seg():
    try:
        return run_yolo_seg_pipeline()
    except Exception as e:
        raise HTTPException(500, f"YOLO seg pipeline failed: {str(e)}")


@router.post("/download")
async def dataset_download():
    try:
        from app.services.datapreparation_service import run_download_only
        return run_download_only()
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


@router.post("/split")
async def dataset_split():
    import hashlib

    if not RAW_DIR.exists():
        raise HTTPException(400, "Raw directory not found")
    all_files = [f for f in sorted(RAW_DIR.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not all_files:
        raise HTTPException(400, "No raw images to split")

    seen_hashes = {}
    files = []
    for f in all_files:
        h = hashlib.md5(f.read_bytes()).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes[h] = f.name
        files.append(f)

    for d in [TRAIN_IMG_DIR, VAL_IMG_DIR, TEST_IMG_DIR, TRAIN_LBL_DIR, VAL_LBL_DIR, TEST_LBL_DIR]:
        if d.exists():
            shutil.rmtree(str(d))
        d.mkdir(parents=True, exist_ok=True)

    random.shuffle(files)
    n = len(files)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)
    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    ann_data = None
    if ANNOTATIONS_FILE.exists():
        ann_data = json.loads(ANNOTATIONS_FILE.read_text())
        img_lookup = {}
        for img in ann_data["images"]:
            key = img["file_name"].replace("/", "_").lower()
            img_lookup[key] = img
        ann_by_img = {}
        for ann in ann_data.get("annotations", []):
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

    copied = {"train": 0, "val": 0, "test": 0}
    for split_name, split_files in [("train", train_files), ("val", val_files), ("test", test_files)]:
        img_dir = {"train": TRAIN_IMG_DIR, "val": VAL_IMG_DIR, "test": TEST_IMG_DIR}[split_name]
        lbl_dir = {"train": TRAIN_LBL_DIR, "val": VAL_LBL_DIR, "test": TEST_LBL_DIR}[split_name]
        for f in split_files:
            shutil.copy2(str(f), str(img_dir / f.name))
            copied[split_name] += 1
            if ann_data:
                key = f.name.lower()
                img_info = img_lookup.get(key)
                if img_info:
                    anns = ann_by_img.get(img_info["id"], [])
                    if anns:
                        with open(lbl_dir / f"{f.stem}.txt", "w") as lf:
                            for ann in anns:
                                cat_map = 0 if ann["category_id"] == 25 else 1
                                x, y, bw, bh = ann["bbox"]
                                x_center = (x + bw / 2) / img_info["width"]
                                y_center = (y + bh / 2) / img_info["height"]
                                w_norm = bw / img_info["width"]
                                h_norm = bh / img_info["height"]
                                lf.write(f"{cat_map} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

    return {"copied": copied, "total_raw": len(all_files), "unique_raw": len(files), "deduped": len(all_files) - len(files)}


@router.post("/pipeline/yolo/test")
async def pipeline_yolo_test():
    try:
        from app.services.yolo_service import run_yolo_test_pipeline
        return run_yolo_test_pipeline()
    except Exception as e:
        raise HTTPException(500, f"YOLO test pipeline failed: {str(e)}")


@router.post("/reset")
async def dataset_reset():
    entries = [d for d in DATASET_DIR.iterdir() if d.is_dir()]
    removed = []
    for d in entries:
        try:
            shutil.rmtree(str(d))
            removed.append(d.name)
        except Exception as e:
            raise HTTPException(500, f"Failed to remove {d.name}: {str(e)}")
    for p in [RAW_DIR, TRAIN_IMG_DIR, TRAIN_LBL_DIR, VAL_IMG_DIR, VAL_LBL_DIR,
              TEST_IMG_DIR, TEST_LBL_DIR, CONVERT_ANN_DIR, CONVERT_VIZ_DIR,
              REAL_ANN_DIR, REAL_VIZ_DIR, VAL_REAL_ANN_DIR, VAL_REAL_VIZ_DIR,
              TEST_REAL_ANN_DIR, TEST_REAL_VIZ_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    return {"removed": removed}


def _parse_annotation_file(ann_dir: Path, filename: str, has_conf: bool = False, is_yolo: bool = False) -> list:
    stem = Path(filename).stem
    ann_file = ann_dir / f"{stem}.txt"
    if not ann_file.exists():
        return []
    predictions = []
    for line in ann_file.read_text().strip().splitlines():
        parts = line.split()
        if is_yolo:
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            entry = {
                "class_id": cls_id,
                "category": "Organik" if cls_id == 0 else "Non-Organik",
                "yolo_bbox": [round(v, 6) for v in [cx, cy, w, h]],
            }
        elif has_conf:
            if len(parts) < 6:
                continue
            cls_id = int(parts[0])
            conf = float(parts[1])
            x1, y1, x2, y2 = map(float, parts[2:6])
            entry = {
                "class_id": cls_id,
                "category": "Organik" if cls_id == 0 else "Non-Organik",
                "bbox": [round(v, 2) for v in [x1, y1, x2, y2]],
                "confidence": round(conf, 4),
            }
        else:
            continue
        predictions.append(entry)
    return predictions


@router.get("/annotation/convert/{filename}")
async def annotation_convert_detail(filename: str):
    predictions = _parse_annotation_file(CONVERT_ANN_DIR, filename, is_yolo=True)
    stem = Path(filename).stem
    viz_path = CONVERT_VIZ_DIR / f"{stem}.jpg"
    return {
        "filename": filename,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "viz_url": f"/api/dataset/file/coco_gt_bbox_img/{stem}.jpg" if viz_path.exists() else None,
    }


@router.get("/annotation/real/{filename}")
async def annotation_real_detail(filename: str):
    predictions = _parse_annotation_file(REAL_ANN_DIR, filename, has_conf=True)
    stem = Path(filename).stem
    viz_path = REAL_VIZ_DIR / f"{stem}.jpg"
    return {
        "filename": filename,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "viz_url": f"/api/dataset/file/yolo_train_bbox_img/{stem}.jpg" if viz_path.exists() else None,
    }


@router.get("/annotation/val/{filename}")
async def annotation_val_detail(filename: str):
    predictions = _parse_annotation_file(VAL_REAL_ANN_DIR, filename, has_conf=True)
    stem = Path(filename).stem
    viz_path = VAL_REAL_VIZ_DIR / f"{stem}.jpg"
    return {
        "filename": filename,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "viz_url": f"/api/dataset/file/yolo_val_bbox_img/{stem}.jpg" if viz_path.exists() else None,
    }


@router.get("/annotation/test/{filename}")
async def annotation_test_detail(filename: str):
    predictions = _parse_annotation_file(TEST_REAL_ANN_DIR, filename, has_conf=True)
    stem = Path(filename).stem
    viz_path = TEST_REAL_VIZ_DIR / f"{stem}.jpg"
    return {
        "filename": filename,
        "predictions": predictions,
        "prediction_count": len(predictions),
        "viz_url": f"/api/dataset/file/yolo_test_bbox_img/{stem}.jpg" if viz_path.exists() else None,
    }
