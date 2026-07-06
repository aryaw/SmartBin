import asyncio
import json
import os
import random
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

import cv2
import numpy as np
from PIL import Image

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import FileResponse
from ultralytics import YOLO

from app.cli.train import train_one
from app.core.config import BASE_DIR, DATASET_PATH, MODEL_PATH, VIZ_DIR
from app.services.kaggle_service import download_and_prepare, _generate_mask, SEED

router = APIRouter(prefix="/api/kaggle", tags=["Kaggle CMS"])

KAGGLE_DIR = BASE_DIR / "dataset" / "kaggle_waste"
TRAIN_RUN_DIR = BASE_DIR / "runs" / "detect" / "api_train"

SUB_TO_MAIN = {}
SUB_TO_SUBTYPE = {}
CATEGORY_FILE = BASE_DIR / "dataset" / "kaggle_waste" / "category_map.json"
_loaded = False

def _load_category_map():
    global SUB_TO_MAIN, SUB_TO_SUBTYPE, _loaded
    if _loaded:
        return
    SUB_TO_MAIN.clear()
    SUB_TO_SUBTYPE.clear()
    cat_path = CATEGORY_FILE
    if cat_path.exists():
        with open(cat_path) as f:
            data = json.load(f)
            for entry in data:
                SUB_TO_MAIN[entry["subcategory"]] = entry["main"]
                SUB_TO_SUBTYPE[entry["subcategory"]] = entry.get("subtype", entry["main"])
    _loaded = True

RECYCLING_ADVICE = {
    "Organik": "Compost bin. Biodegradable waste suitable for composting or eco-enzyme.",
    "Anorganik": "Recycling bin. Sort plastic, paper, glass, metal for Bank Sampah or recycling facility.",
    "Residu": "General trash. Send to TPA (final disposal). Cannot be recycled or composted.",
}

_executor = ThreadPoolExecutor(max_workers=1)
_training_future = None


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


@router.post("/pipeline/run-full")
async def run_full_pipeline(
    epochs: int = Query(int(os.getenv("EPOCHS", "150")), description="Training epochs"),
    batch: int = Query(16, description="Batch size"),
):
    results = {}
    try:
        from app.services.kaggle_service import prepare_from_local
        local_path = DATASET_PATH
        if not local_path.exists():
            raise HTTPException(400, f"Dataset not found at {local_path}")
        dl = prepare_from_local(str(local_path))
        results["download"] = {"total_images": dl["total_images"], "classes": dl["classes"]}

        data_yaml = KAGGLE_DIR / "data.yaml"
        if data_yaml.exists():
            best_path, map50 = train_one(
                pretrained="yolo26m-seg.pt",
                data=str(data_yaml),
                epochs=epochs,
                batch=batch,
                imgsz=640,
                patience=int(os.getenv("PATIENCE", "40")),
                device="cuda:0",
                name="full_pipeline",
                lr0=float(os.getenv("LR0", "0.001")),
                optimizer=os.getenv("OPTIMIZER", "SGD"),
                warmup_epochs=float(os.getenv("WARMUP_EPOCHS", "5")),
                mask_ratio=int(os.getenv("MASK_RATIO", "2")),
            )
            if best_path and Path(best_path).exists():
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(best_path), str(MODEL_PATH))
            results["train"] = {"best_model": str(best_path), "map50": map50}

        from app.services.kaggle_service import run_visualization_pipeline
        try:
            viz_result = run_visualization_pipeline()
            results["visualization"] = {
                "success": viz_result["success"],
                "output_path": viz_result["output_path"],
            }
        except Exception as ve:
            results["visualization"] = {"success": False, "error": str(ve)}

        return {
            "status": "Pipeline complete",
            "results": results,
            "message": "Model trained and visualizations generated. Go to dashboard to detect waste or /visualization to view pipeline images.",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Pipeline failed: {e}")


@router.post("/download")
async def download(source: str = Query("local", description="Dataset source: local (Waste_Classification_Dataset) or kaggle")):
    try:
        if source == "local":
            from app.services.kaggle_service import prepare_from_local
            local_path = DATASET_PATH
            if not local_path.exists():
                raise HTTPException(400, f"Local dataset not found at {local_path}")
            result = prepare_from_local(str(local_path))
            result["source"] = "local"
            return result
        return download_and_prepare()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {e}")


@router.post("/convert")
async def convert():
    if not KAGGLE_DIR.exists():
        raise HTTPException(400, "Dataset not found. Download first.")

    counts = {"skipped_existing": 0, "generated": 0}

    for split in ("train", "val", "test"):
        img_dir = KAGGLE_DIR / split / "images"
        lbl_dir = KAGGLE_DIR / split / "labels"
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.is_dir():
            continue

        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if lbl_path.exists():
                counts["skipped_existing"] += 1
                continue

            try:
                img = Image.open(img_path).convert("RGB")
                poly, _ = _generate_mask(img, randomize=(split == "train"))
                coords = " ".join(f"{v:.6f}" for v in poly)
                lbl_path.write_text(f"0 {coords}\n")
                counts["generated"] += 1
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


@router.get("/train/status")
async def train_status():
    global _training_future
    if _training_future is not None and not _training_future.done():
        return {"running": True, "task": "training"}
    return {"running": False, "task": None}


@router.post("/train")
async def train(
    epochs: int = Query(50, description="Number of epochs"),
    batch: int = Query(16, description="Batch size"),
    imgsz: int = Query(640, description="Image size"),
    lr0: float = Query(0.001, description="Initial learning rate"),
):
    global _training_future
    if _training_future is not None and not _training_future.done():
        raise HTTPException(400, "Training already in progress")

    if not KAGGLE_DIR.exists():
        raise HTTPException(400, "Dataset not found. Download first.")

    data_yaml = KAGGLE_DIR / "data.yaml"
    if not data_yaml.exists():
        raise HTTPException(400, "Data config not found. Run download first.")

    loop = asyncio.get_event_loop()

    def _train():
        best_path, map50 = train_one(
            pretrained="yolo26m-seg.pt",
            data=str(data_yaml),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            patience=30,
            device="cuda:0",
            name="api_train",
            lr0=lr0,
            optimizer="SGD",
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
        _training_future = loop.run_in_executor(_executor, _train)
        result = await _training_future
        return result
    except Exception as e:
        raise HTTPException(500, f"Training failed: {e}")
    finally:
        _training_future = None


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


@router.post("/visualization/run")
async def run_visualization(process: str = Query(None, description="Single process name to regenerate, or all if omitted")):
    from app.services.kaggle_service import run_visualization_pipeline
    try:
        result = run_visualization_pipeline(process_name=process)
        return {
            "success": result["success"],
            "message": "Visualization pipeline complete" if result["success"] else "Visualization pipeline failed",
            "output_path": result["output_path"],
            "details": result.get("results"),
        }
    except Exception as e:
        raise HTTPException(500, f"Visualization failed: {e}")


@router.get("/explore")
async def explore():
    local_path = DATASET_PATH
    if not local_path.exists():
        raise HTTPException(400, f"Dataset not found at {local_path}")

    CLASS_NAMES = []
    CLASS_MAP = {}
    CATEGORY_TO_SUBS = {}
    all_paths = []

    categories = sorted(os.listdir(local_path))
    cid = 0
    for cat in categories:
        inner = local_path / cat / cat
        if not inner.is_dir():
            inner = local_path / cat
            if not inner.is_dir():
                continue
        subs = sorted(os.listdir(inner))
        CATEGORY_TO_SUBS[cat] = subs
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
    class_counts = {}
    class_samples = {}
    for img_path, cid, sub in all_paths:
        class_counts[sub] = class_counts.get(sub, 0) + 1
        if sub not in class_samples:
            class_samples[sub] = f"/api/dataset/file/raw/{Path(img_path).name}"

    distribution = []
    for cid in range(NC):
        name = CLASS_NAMES[cid]
        count = class_counts.get(name, 0)
        distribution.append({
            "class_id": cid,
            "name": name,
            "count": count,
            "sample_url": class_samples.get(name),
        })

    distribution.sort(key=lambda x: x["count"], reverse=True)

    return {
        "num_classes": NC,
        "class_names": CLASS_NAMES,
        "category_to_subs": CATEGORY_TO_SUBS,
        "distribution": distribution,
        "total_images": len(all_paths),
    }


@router.get("/categories")
async def get_categories():
    _load_category_map()
    return {
        "categories": list(RECYCLING_ADVICE.keys()),
        "sub_to_main": SUB_TO_MAIN,
        "recycling_advice": RECYCLING_ADVICE,
    }


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

        result = {
            "box_mAP50": metrics.get("metrics/mAP50(B)", 0),
            "box_mAP50_95": metrics.get("metrics/mAP50-95(B)", 0),
            "box_precision": metrics.get("metrics/precision(B)", 0),
            "box_recall": metrics.get("metrics/recall(B)", 0),
            "mask_mAP50": metrics.get("metrics/mAP50(M)", 0),
            "mask_mAP50_95": metrics.get("metrics/mAP50-95(M)", 0),
            "mask_precision": metrics.get("metrics/precision(M)", 0),
            "mask_recall": metrics.get("metrics/recall(M)", 0),
        }

        if hasattr(val_results, "box") and hasattr(val_results.box, "ap_class_index"):
            cls_names = model.names if hasattr(model, "names") else {}
            box_per_class = {}
            for i, c in enumerate(val_results.box.ap_class_index):
                name_cls = cls_names.get(int(c), str(c))
                ap = val_results.box.ap[i] if hasattr(val_results.box, "ap") and i < len(val_results.box.ap) else 0
                box_per_class[str(int(c))] = {"name": name_cls, "box_ap50": float(ap)}
            result["box_per_class"] = box_per_class

        if hasattr(val_results, "seg") and hasattr(val_results.seg, "ap_class_index"):
            cls_names = model.names if hasattr(model, "names") else {}
            seg_per_class = {}
            for i, c in enumerate(val_results.seg.ap_class_index):
                name_cls = cls_names.get(int(c), str(c))
                ap = val_results.seg.ap[i] if hasattr(val_results.seg, "ap") and i < len(val_results.seg.ap) else 0
                seg_per_class[str(int(c))] = {"name": name_cls, "mask_ap50": float(ap)}
            result["mask_per_class"] = seg_per_class

        return result
    except Exception as e:
        raise HTTPException(500, f"Evaluation failed: {e}")


@router.post("/export")
async def export_model(format: str = Query("onnx", description="Export format: onnx, torchscript, all")):
    best_path = MODEL_PATH
    if not best_path.exists():
        raise HTTPException(400, "No trained model found")

    export_dir = KAGGLE_DIR / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(best_path))
    results = {}

    try:
        if format in ("onnx", "all"):
            onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
            dest = export_dir / "waste_yolo26seg_e2e.onnx"
            if Path(onnx_path).exists():
                shutil.copy(str(onnx_path), str(dest))
                results["onnx"] = {"path": str(dest), "size_mb": round(dest.stat().st_size / 1e6, 1)}
    except Exception as e:
        results["onnx"] = {"error": str(e)}

    try:
        if format in ("torchscript", "all"):
            ts_path = model.export(format="torchscript", imgsz=640)
            dest = export_dir / "waste_yolo26seg.torchscript"
            if Path(ts_path).exists():
                shutil.copy(str(ts_path), str(dest))
                results["torchscript"] = {"path": str(dest), "size_mb": round(dest.stat().st_size / 1e6, 1)}
    except Exception as e:
        results["torchscript"] = {"error": str(e)}

    return {"export_dir": str(export_dir), "formats": results}


@router.post("/inference/batch")
async def inference_batch():
    if not MODEL_PATH.exists():
        raise HTTPException(400, "No trained model found")
    if not KAGGLE_DIR.exists():
        raise HTTPException(400, "Dataset not found")

    test_img_dir = KAGGLE_DIR / "test" / "images"
    if not test_img_dir.is_dir():
        raise HTTPException(400, "Test images not found")

    model = YOLO(str(MODEL_PATH))
    output_dir = KAGGLE_DIR / ".." / "batch_inference"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    class_detections = {}
    images_processed = 0

    img_files = sorted(
        f for f in test_img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in img_files:
        try:
            results = model(str(img_path), retina_masks=True, verbose=False)
            r = results[0]

            annotated = r.plot(masks=True, boxes=True, labels=True, conf=True)
            out_path = output_dir / f"batch_{img_path.name}"
            cv2.imwrite(str(out_path), annotated)

            if r.boxes is not None:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    name = model.names.get(cls_id, str(cls_id))
                    if name not in class_detections:
                        class_detections[name] = {"count": 0, "max_conf": 0, "class_id": cls_id}
                    class_detections[name]["count"] += 1
                    class_detections[name]["max_conf"] = max(class_detections[name]["max_conf"], conf)
                    total += 1

            images_processed += 1
        except Exception:
            continue

    _load_category_map()
    advice_list = []
    for name, det in sorted(class_detections.items(), key=lambda x: x[1]["count"], reverse=True):
        main_cat = SUB_TO_MAIN.get(name, "Unknown")
        subtype = SUB_TO_SUBTYPE.get(name, main_cat)
        advice = RECYCLING_ADVICE.get(subtype, RECYCLING_ADVICE.get(main_cat, ""))
        advice_list.append({
            "class_name": name,
            "class_id": det["class_id"],
            "count": det["count"],
            "max_confidence": round(det["max_conf"], 4),
            "category": main_cat,
            "subtype": subtype,
            "advice": advice,
        })

    return {
        "images_processed": images_processed,
        "total_detections": total,
        "per_class": advice_list,
        "output_dir": str(output_dir),
    }


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
        results = model(tmp_path, retina_masks=True)
        annotated = results[0].plot(masks=True, boxes=True, labels=True, conf=True)

        out_path = tmp_path + "_annotated.jpg"
        cv2.imwrite(out_path, annotated)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                name = model.names.get(cls_id, str(cls_id))
                detections.append({"class": name, "class_id": cls_id, "confidence": round(conf, 4)})

        return FileResponse(
            out_path,
            media_type="image/jpeg",
            filename=f"annotated_{file.filename}",
            headers={"X-Detections": json.dumps(detections)},
        )
    except Exception as e:
        raise HTTPException(500, f"Inference failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/verify")
async def verify():
    if not MODEL_PATH.exists():
        raise HTTPException(400, "No trained model found")

    data_yaml = KAGGLE_DIR / "data.yaml"
    if not data_yaml.exists():
        raise HTTPException(400, "Dataset data.yaml not found")

    model = YOLO(str(MODEL_PATH))
    results = model.val(data=str(data_yaml), split="test", imgsz=640, batch=16)

    _load_category_map()
    per_class = []
    if hasattr(results, "seg") and hasattr(results.seg, "ap_class_index"):
        cls_names = model.names if hasattr(model, "names") else {}
        for i, c in enumerate(results.seg.ap_class_index):
            name = cls_names.get(int(c), str(c))
            ap50 = float(results.seg.ap50[i]) if hasattr(results.seg, "ap50") and i < len(results.seg.ap50) else 0
            ap = float(results.seg.ap[i]) if hasattr(results.seg, "ap") and i < len(results.seg.ap) else 0
            main_cat = SUB_TO_MAIN.get(name, "Unknown")
            per_class.append({
                "class_id": int(c),
                "name": name,
                "category": main_cat,
                "mask_ap50": round(ap50, 4),
                "mask_ap50_95": round(ap, 4),
            })

    per_class.sort(key=lambda x: x["mask_ap50"], reverse=True)

    return {
        "model": "yolo26m-seg",
        "num_classes": len(per_class),
        "summary": {
            "box_mAP50": round(results.box.map50, 4) if hasattr(results, "box") else 0,
            "box_mAP50_95": round(results.box.map, 4) if hasattr(results, "box") else 0,
            "mask_mAP50": round(results.seg.map50, 4) if hasattr(results, "seg") else 0,
            "mask_mAP50_95": round(results.seg.map, 4) if hasattr(results, "seg") else 0,
        },
        "per_class": per_class,
    }
