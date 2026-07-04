import os

import torch

from app.core.config import DEVICE


def init_gpu():
    if not torch.cuda.is_available():
        return

    vram_gb = float(os.getenv("VRAM_LIMIT_GB", "12"))
    total = torch.cuda.get_device_properties(0).total_memory
    fraction = vram_gb * 1024**3 / total
    fraction = min(fraction, 0.95)
    torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cudnn.benchmark = True


def get_device() -> str:
    if torch.cuda.is_available():
        return DEVICE
    return "cpu"


def try_log_gpu() -> dict:
    if not torch.cuda.is_available():
        return {"device": "cpu", "vram_gb": 0}
    return {
        "device": "cuda",
        "vram_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
        "vram_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
    }


def warmup_model(model):
    init_gpu()
    device = get_device()
    model.to(device)
    import numpy as np
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    model(dummy, verbose=False)
