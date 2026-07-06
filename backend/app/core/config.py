import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

def _resolve_path(name: str, default: str) -> Path:
    val = os.getenv(name, default)
    p = Path(val)
    return p if p.is_absolute() else BASE_DIR / p

DATASET_PATH = BASE_DIR.parent / os.getenv("DATASET_PATH", "backend/dataset")

UPLOAD_DIR = _resolve_path("UPLOAD_DIR", "uploads")
LOG_DIR = _resolve_path("LOG_DIR", "log")
STATIC_DIR = _resolve_path("STATIC_DIR", "static")
RESULT_DIR = _resolve_path("RESULT_DIR", "static/result")
MODEL_DIR = _resolve_path("MODEL_DIR", "models")
MODEL_PATH = _resolve_path("MODEL_PATH", "models/best.pt")

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", str(200 * 1024 * 1024)))

DEVICE = os.getenv("DEVICE", "cuda:0")
CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "0")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "smartbin")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DATABASE_URL = f"postgresql+asyncpg://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

ORGANIC_CATEGORIES = {25}  # only Food waste

def is_organic(category_id: int) -> bool:
    return category_id in ORGANIC_CATEGORIES

ANNOTATIONS_FILE = _resolve_path("ANNOTATIONS_FILE", "../datasource/annotations.json")

VIZ_DIR = BASE_DIR.parent / "waste_datasource" / "visualization"

for d in [UPLOAD_DIR, LOG_DIR, RESULT_DIR, MODEL_DIR, VIZ_DIR]:
    d.mkdir(parents=True, exist_ok=True)
