import argparse
from pathlib import Path

from ultralytics import YOLO

from app.utils.gpu_utils import get_device


def main():
    parser = argparse.ArgumentParser(description="Evaluate YOLO model for SmartBin")
    parser.add_argument("--model", type=str, default="models/best.pt")
    parser.add_argument("--data", type=str, default="data.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--conf-sweep", action="store_true",
                        help="Sweep confidence thresholds 0.1-0.9 to find optimal")
    args = parser.parse_args()

    device = args.device or get_device()
    model = YOLO(args.model)

    results = model.val(
        data=args.data,
        split=args.split,
        device=device,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        verbose=False,
    )

    d = results.results_dict
    cls_names = model.names if hasattr(model, "names") else {}

    print(f"\n{'='*50}")
    print(f"Evaluation Results — split={args.split}")
    print(f"{'='*50}")
    print(f"Model      : {args.model}")
    print(f"Dataset    : {args.data}")
    print(f"mAP@0.5    : {d.get('metrics/mAP50(B)', 0) * 100:.1f}%")
    print(f"mAP@0.5:0.95: {d.get('metrics/mAP50-95(B)', 0) * 100:.1f}%")
    print(f"Precision  : {d.get('metrics/precision(B)', 0) * 100:.1f}%")
    print(f"Recall     : {d.get('metrics/recall(B)', 0) * 100:.1f}%")

    if hasattr(results, "box") and hasattr(results.box, "ap_class_index"):
        print(f"\n  Per-Class mAP@0.5:")
        for i, c in enumerate(results.box.ap_class_index):
            name = cls_names.get(int(c), str(c))
            ap = results.box.ap[i] if hasattr(results.box, "ap") and i < len(results.box.ap) else 0
            print(f"    {name:>20s}: {ap * 100:.1f}%")

    if args.conf_sweep:
        print(f"\n{'='*50}")
        print(f"Confidence Threshold Sweep")
        print(f"{'='*50}")
        for conf in [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            r = model.val(
                data=args.data, split=args.split, device=device,
                imgsz=args.imgsz, conf=conf, iou=args.iou, verbose=False,
            )
            rd = r.results_dict
            p = rd.get("metrics/precision(B)", 0) * 100
            rec = rd.get("metrics/recall(B)", 0) * 100
            m50 = rd.get("metrics/mAP50(B)", 0) * 100
            f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0
            print(f"  conf={conf:.1f}  |  P={p:.1f}%  R={rec:.1f}%  mAP50={m50:.1f}%  F1={f1:.1f}%")


if __name__ == "__main__":
    main()
