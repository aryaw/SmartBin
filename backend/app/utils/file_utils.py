from pathlib import Path
from datetime import datetime

from fastapi import HTTPException, UploadFile

from app.core.config import ALLOWED_IMAGE_EXT, ALLOWED_VIDEO_EXT, MAX_FILE_SIZE, UPLOAD_DIR


def validate_upload(file: UploadFile):
    ext = Path(file.filename).suffix.lower()
    allowed = ALLOWED_IMAGE_EXT | ALLOWED_VIDEO_EXT
    if ext not in allowed:
        raise HTTPException(400, "Format file tidak didukung. Allowed: JPG, JPEG, PNG, MP4, AVI, MOV")

    if ext in ALLOWED_IMAGE_EXT:
        file.file.seek(0, 2)
        size = file.file.tell()
        file.file.seek(0)
        if size > MAX_FILE_SIZE:
            raise HTTPException(413, "File terlalu besar. Maksimum 200MB")


def get_file_type(ext: str) -> str:
    if ext in ALLOWED_IMAGE_EXT:
        return "image"
    return "video"


def save_upload(file: UploadFile) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ts}_{file.filename}"
    dest = UPLOAD_DIR / filename
    content = file.file.read()
    dest.write_bytes(content)
    return dest


def cleanup_old_files(directory: Path, max_age_hours: int = 1):
    now = datetime.now().timestamp()
    for f in directory.iterdir():
        if f.is_file() and (now - f.stat().st_mtime) > max_age_hours * 3600:
            f.unlink(missing_ok=True)
