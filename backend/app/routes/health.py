import time
import logging

import torch
from fastapi import APIRouter

from app.schemas.detection import HealthResponse
from app.services.detector import _model, reload_model
from app.core.config import MODEL_PATH

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])

_start_time = time.time()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    cuda_avail = torch.cuda.is_available()
    device = f"cuda:{torch.cuda.current_device()}" if cuda_avail else "cpu"
    vram = round(torch.cuda.memory_allocated() / 1024**3, 2) if cuda_avail else 0
    model_loaded = _model is not None
    model_exists = MODEL_PATH.exists()
    uptime = round(time.time() - _start_time, 1)

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        device=device,
        model_loaded=model_loaded,
        model_exists=model_exists,
        cuda_available=cuda_avail,
        vram_gb=vram,
        uptime_hours=uptime / 3600,
    )


@router.post("/health/reload")
async def reload_model_endpoint():
    try:
        reload_model()
        return {"status": "ok", "message": "Model reloaded"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
