from pydantic import BaseModel


class DetectedObject(BaseModel):
    label: str
    category: str
    confidence: float
    bbox: list[float] | None = None
    count: int | None = None


class Summary(BaseModel):
    organik: int = 0
    non_organik: int = 0
    total: int = 0


class DetectResponse(BaseModel):
    success: bool
    file_type: str
    filename: str
    frames_processed: int | None = None
    detected_objects: list[DetectedObject]
    summary: Summary
    result_url: str
    recommendation: str = "Pisahkan sampah sesuai kategori sebelum dibuang."


class ErrorResponse(BaseModel):
    detail: str


class HealthResponse(BaseModel):
    status: str
    device: str
