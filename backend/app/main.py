from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, STATIC_DIR
from app.routes import health, detect, annotation, datasource


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.utils.gpu_utils import init_gpu
    init_gpu()
    from app.services.detector import load_model
    load_model()
    yield


app = FastAPI(title="SmartBin API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(health.router)
app.include_router(detect.router)
app.include_router(annotation.router)
app.include_router(datasource.router)

static_result = STATIC_DIR / "result"
static_result.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
