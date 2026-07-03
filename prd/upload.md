# Upload Mechanism PRD - SmartBin

---

## Flow

```
User selects file → Client validation (type + size) → Preview
→ Click [Detect] → POST /api/detect (multipart/form-data)
→ Server validation → Save to uploads/ → YOLO inference
→ Save annotated result → Log detection → Return JSON
→ Frontend renders result with bounding boxes + summary
```

## File Validation

- Allowed: `.jpg`, `.jpeg`, `.png`, `.mp4`, `.avi`, `.mov`
- Max size: 200MB
- Client + server validation

## Storage

- Upload: `backend/uploads/{timestamp}_{filename}` (deleted after processing)
- Result: `backend/static/result/{timestamp}_{filename}_annotated.jpg`
- Log: `backend/log/{timestamp}/detection.log`

## Detection Result

Returns JSON with:
- `detected_objects[]`: label, category (Organik/Non-Organik), confidence, bbox [x1,y1,x2,y2]
- `summary`: organik count, non_organik count, total
- `result_url`: path to annotated image/video
- `recommendation`: waste disposal recommendation
