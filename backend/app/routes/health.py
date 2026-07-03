from fastapi import APIRouter
import torch

from app.schemas.detection import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return HealthResponse(status="ok", device=device)
