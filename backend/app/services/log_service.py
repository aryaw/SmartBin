import json
from datetime import datetime
from pathlib import Path

from app.core.config import LOG_DIR


def _date_dir() -> Path:
    p = LOG_DIR / datetime.now().strftime("%Y-%m-%d")
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_detection_log(
    timestamp: str,
    filename: str,
    file_type: str,
    objects: list,
    summary: dict,
    duration_ms: int,
):
    log_path = _date_dir()

    log_file = log_path / "detection.log"
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now().isoformat()}] File: {filename}\n")
        f.write(f"[{datetime.now().isoformat()}] Type: {file_type}\n")
        f.write(f"[{datetime.now().isoformat()}] Objects: {summary['total']}\n")
        f.write(f"[{datetime.now().isoformat()}] Organik: {summary['organik']}, Non-Organik: {summary['non_organik']}\n")
        f.write(f"[{datetime.now().isoformat()}] Duration: {duration_ms}ms\n")
        f.write("\n")

    result = {
        "timestamp": timestamp,
        "filename": filename,
        "file_type": file_type,
        "duration_ms": duration_ms,
        "objects": objects,
        "summary": summary,
    }
    result_file = log_path / "result.jsonl"
    with open(result_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    return log_path


def read_detection_log(timestamp: str) -> dict | None:
    for d in sorted(LOG_DIR.iterdir()):
        if not d.is_dir():
            continue
        result_file = d / "result.jsonl"
        if not result_file.exists():
            continue
        for line in result_file.read_text().strip().splitlines():
            entry = json.loads(line)
            if entry.get("timestamp") == timestamp:
                return entry
    return None
