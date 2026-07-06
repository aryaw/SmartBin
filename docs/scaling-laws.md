# Scaling Laws for YOLO Waste Detection

## Core Concept

Scaling law = predictable relationship between model size, data, compute, and performance.

Performance (mAP) follows a power-law as you scale:
- Model parameters (bigger architecture)
- Training data size (more images)
- Compute budget (longer training, higher resolution)

## YOLO26 Scaling Family

| Model   | Params  | FLOPs   | Relative mAP gain |
|---------|---------|---------|-------------------|
| YOLO26n | ~5.8M   | ~30 G   | baseline          |
| YOLO26m | ~23.5M  | ~121 G  | +~5-7%            |
| YOLO26l | ~44.2M  | ~230 G  | +~2-3% over m     |
| YOLO26x | ~76.8M  | ~400 G  | +~2-3% over l     |

Each step ~2x params, ~1.5x FLOPs, estimated ~1.3x mAP gain.

## Practical Scaling Steps

| Step | Action | Expected gain |
|------|--------|---------------|
| 1 | Train YOLO26x (vs current 26m) | +2-3% mAP |
| 2 | Add more training data | +5-10% mAP |
| 3 | Longer training + cosine annealing | +1-2% mAP |
| 4 | Hyperparameter search (lr, batch, mosaic) | +2-5% mAP |

## References

- Kaplan et al. "Scaling Laws for Neural Language Models" (OpenAI 2020)
- Ultralytics YOLO documentation
