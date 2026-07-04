import argparse
import os
import shutil
from pathlib import Path

from ultralytics import YOLO

from app.core.config import MODEL_DIR
from app.utils.gpu_utils import get_device


def train_one(pretrained: str, data: str, epochs: int, batch: int, imgsz: int,
              patience: int, device: str, name: str, lr0: float = 0.001,
              lrf: float = 0.01, warmup_epochs: float = 3.0,
              progress_callback=None) -> tuple:
    model = YOLO(pretrained)

    if progress_callback:
        def _on_epoch_end(trainer):
            progress_callback({
                "epoch": trainer.epoch,
                "epochs": trainer.epochs,
                "lr": round(trainer.lr[0] if isinstance(trainer.lr, list) else trainer.lr, 6),
                "loss": round(float(trainer.loss), 4) if hasattr(trainer, "loss") and trainer.loss is not None else None,
            })
        model.add_callback("on_train_epoch_end", _on_epoch_end)

    results = model.train(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        workers=0,
        patience=patience,
        device=device,
        augment=True,
        mosaic=1.0,
        close_mosaic=min(10, epochs // 2),
        hsv_h=0.05,
        hsv_s=0.8,
        hsv_v=0.5,
        scale=0.5,
        translate=0.2,
        degrees=15.0,
        shear=5.0,
        perspective=0.0001,
        flipud=0.2,
        fliplr=0.5,
        erasing=0.4,
        mixup=0.2,
        copy_paste=0.3,
        auto_augment="randaugment",
        lr0=lr0,
        lrf=lrf,
        warmup_epochs=warmup_epochs,
        cos_lr=True,
        label_smoothing=0.05,
        name=name,
        exist_ok=True,
        deterministic=True,
        single_cls=False,
    )

    val = model.val()
    metrics = val.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  mAP@0.5    : {map50 * 100:.1f}%")
    print(f"  mAP@0.5:0.95: {map50_95 * 100:.1f}%")
    print(f"  Precision  : {precision * 100:.1f}%")
    print(f"  Recall     : {recall * 100:.1f}%")
    print(f"  F1-Score   : {f1 * 100:.1f}%")

    if hasattr(val, "box") and hasattr(val.box, "ap_class_index"):
        cls_names = model.names if hasattr(model, "names") else {}
        for i, c in enumerate(val.box.ap_class_index):
            name_cls = cls_names.get(int(c), str(c))
            ap = val.box.ap[i] if hasattr(val.box, "ap") and i < len(val.box.ap) else 0
            print(f"  {name_cls:>20s}: {ap * 100:.1f}%")

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    return best_path, map50


def main():
    parser = argparse.ArgumentParser(description="Train YOLO model for SmartBin")
    parser.add_argument("--model", type=str, default="yolo11m.pt")
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "200")))
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--grid-search", type=int, nargs="*", default=None)
    args = parser.parse_args()

    device = args.device or get_device()

    if args.grid_search:
        best_map50 = 0.0
        best_path = None
        for ep in args.grid_search:
            print(f"\nGrid search: {ep} epochs")
            path, map50 = train_one(
                pretrained=args.model, data=args.data, epochs=ep,
                batch=args.batch, imgsz=args.imgsz, patience=args.patience,
                device=device, lr0=args.lr0, name=f"grid_{ep}ep",
            )
            if map50 > best_map50:
                best_map50 = map50
                best_path = path
        if best_path and best_path.exists():
            dest = MODEL_DIR / "best.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(best_path), str(dest))
            print(f"Best model (mAP@0.5={best_map50 * 100:.1f}%) saved to {dest}")
    else:
        path, _ = train_one(
            pretrained=args.model, data=args.data, epochs=args.epochs,
            batch=args.batch, imgsz=args.imgsz, patience=args.patience,
            device=device, lr0=args.lr0, name="train",
        )
        if path and path.exists():
            dest = MODEL_DIR / "best.pt"
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(path), str(dest))
            print(f"Model saved to {dest}")


if __name__ == "__main__":
    main()
