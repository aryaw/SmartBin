import time, asyncio
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from app.config import ALLOWED_IMAGE_EXT, RESULT_DIR
from app.schemas.detection import DetectResponse, DetectedObject, Summary
from app.services.detector import detect_image, detect_video
from app.services.log_service import write_detection_log, read_detection_log
from app.utils.file_utils import validate_upload, save_upload, get_file_type, cleanup_old_files, UPLOAD_DIR

router = APIRouter(prefix="/api", tags=["Detection"])


@router.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)):
    validate_upload(file)
    ext = Path(file.filename).suffix.lower()
    file_type = get_file_type(ext)

    upload_path = save_upload(file)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        start = time.time()
        if file_type == "image":
            result = detect_image(upload_path)
        else:
            result = detect_video(upload_path)
        duration_ms = int((time.time() - start) * 1000)
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")
    finally:
        upload_path.unlink(missing_ok=True)

    write_detection_log(
        timestamp=ts,
        filename=file.filename,
        file_type=file_type,
        objects=result["detected_objects"],
        summary=result["summary"],
        duration_ms=duration_ms,
    )

    objects = [
        DetectedObject(**obj) for obj in result["detected_objects"]
    ]
    summary = Summary(**result["summary"])

    return DetectResponse(
        success=True,
        file_type=file_type,
        filename=file.filename,
        frames_processed=result.get("frames_processed"),
        detected_objects=objects,
        summary=summary,
        result_url=result["result_url"],
        recommendation=result["recommendation"],
    )


@router.post("/detect/bulk")
async def detect_bulk(files: list[UploadFile] = File(...)):
    results = []
    errors = []

    async def process_one(file: UploadFile):
        try:
            validate_upload(file)
            ext = Path(file.filename).suffix.lower()
            file_type = get_file_type(ext)
            upload_path = save_upload(file)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                start = time.time()
                if file_type == "image":
                    result = detect_image(upload_path)
                else:
                    result = detect_video(upload_path)
                duration_ms = int((time.time() - start) * 1000)
            except Exception as e:
                return {"filename": file.filename, "error": str(e)}
            finally:
                upload_path.unlink(missing_ok=True)

            write_detection_log(
                timestamp=ts, filename=file.filename, file_type=file_type,
                objects=result["detected_objects"], summary=result["summary"],
                duration_ms=duration_ms,
            )

            return {
                "filename": file.filename,
                "file_type": file_type,
                "success": True,
                "detected_objects": result["detected_objects"],
                "summary": result["summary"],
                "result_url": result["result_url"],
                "recommendation": result["recommendation"],
            }
        except HTTPException as e:
            return {"filename": file.filename, "error": e.detail}
        except Exception as e:
            return {"filename": file.filename, "error": str(e)}

    tasks = [process_one(f) for f in files]
    for r in await asyncio.gather(*tasks):
        if "error" in r:
            errors.append(r)
        else:
            results.append(r)

    return {"results": results, "errors": errors, "total": len(results), "failed": len(errors)}


@router.get("/result/{filename}")
async def get_result(filename: str):
    file_path = RESULT_DIR / filename
    if not file_path.exists():
        raise HTTPException(404, "Result not found")
    from fastapi.responses import FileResponse
    return FileResponse(str(file_path))


@router.get("/log/{timestamp}")
async def get_log(timestamp: str):
    data = read_detection_log(timestamp)
    if data is None:
        raise HTTPException(404, "Log not found")
    return data
