import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

UPLOAD_DIR = BASE_DIR / "uploads"
LOG_DIR = BASE_DIR / "log"
STATIC_DIR = BASE_DIR / "static"
RESULT_DIR = STATIC_DIR / "result"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best.pt"

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov"}
MAX_FILE_SIZE = 200 * 1024 * 1024

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

ORGANIC_CATEGORIES = {
    13,  14,  15,  16,  17,  18,  19,  20,
    25,  30,  31,  32,  33,  34,  56,
}

def is_organic(category_id: int) -> bool:
    return category_id in ORGANIC_CATEGORIES

ANNOTATIONS_FILE = BASE_DIR.parent / "datasource" / "annotations.json"

for d in [UPLOAD_DIR, LOG_DIR, RESULT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)
