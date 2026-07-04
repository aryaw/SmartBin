import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, STATIC_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting SmartBin API...")
    from app.utils.gpu_utils import init_gpu
    try:
        init_gpu()
        logger.info("GPU initialized")
    except Exception as e:
        logger.warning(f"GPU init failed, falling back to CPU: {e}")

    from app.services.detector import load_model
    try:
        load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Model load failed: {e}")
    yield
    logger.info("Shutting down SmartBin API...")


app = FastAPI(title="SmartBin API", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception(f"Unhandled error processing {request.method} {request.url.path}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error": str(e)},
        )
    elapsed = time.time() - start
    response.headers["X-Process-Time-Ms"] = str(int(elapsed * 1000))
    return response


from app.routes import health, detect, annotation, datasource

app.include_router(health.router)
app.include_router(detect.router)
app.include_router(annotation.router)
app.include_router(datasource.router)

static_result = STATIC_DIR / "result"
static_result.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
