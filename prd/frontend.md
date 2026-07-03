# Frontend PRD - SmartBin

**Stack:** Nuxt 3, Vue 3, TypeScript, Tailwind CSS

---

## 1. Sidebar Navigation (3 groups, 10 items, 10 unique routes)

### Main
| Menu | Route | Description |
|------|-------|-------------|
| Dashboard Report | `/dashboard` | File upload + detect + Prepare Data button |
| Test Upload | `/test` | Quick upload test page |

### Dataset
| Menu | Route | Description |
|------|-------|-------------|
| All Raw Data | `/raw` | Raw image grid + Download + Split + Reset buttons |
| Train Data | `/train-data` | Train image grid + Generate Annotation / Segmentation buttons |
| Test Data | `/test-data` | Test split images grid |
| Evaluation Metrics | `/eval` | Model metrics table + val inference images |

### Annotations
| Menu | Route | Description |
|------|-------|-------------|
| BoundingBox Result | `/boundingbox-result` | COCO bounding box annotations with image zoom |
| Segmentation Result | `/segmentation-result` | Segmentation predictions from COCO dataset |
| Validation Inference Result | `/val-result` | Run val pipeline + show results |
| Test Inference Result | `/test-result` | Run test pipeline + show results |

All grid pages: ZoomModal (click-to-zoom popup) + Pagination component.

---

## 2. Color Scheme

| Token | Hex | Usage |
|-------|-----|-------|
| Primary (sidebar) | `#1F2937` | Sidebar background |
| Secondary | `#3B82F6` | Accent buttons |
| Tertiary | `#2563EB` | Primary action buttons |

---

## 3. Components

| Component | Usage |
|-----------|-------|
| `Pagination.vue` | Page navigation for image grids |
| `ZoomModal.vue` | Teleported fullscreen image popup |
| `BoundingBoxReport.vue` | Detection result table with bbox coords |
| `DetectionResult.vue` | Detection result card with summary + stats |
| `FileUpload.vue` | Drag-drop file upload zone |
| `LoadingOverlay.vue` | Spinner overlay |
| `YoloGrid.vue` | YOLO annotation grid with bbox overlay thumbnails |

---

## 4. Composables

| Composable | Usage |
|------------|-------|
| `useDetection.ts` | API calls for detection endpoints |
| `useFileUpload.ts` | File validation, preview, form-data construction |

---

## 5. API Integration

Base URL: configurable via `.env` (`API_BASE=http://localhost:8000`) or runtime config.
Default: `http://localhost:8080` in dev, `http://backend:8000` in Docker.

---

## 6. Pages Detail

| Page | Key Features |
|------|-------------|
| Dashboard | Gradient header, drag-drop upload, file preview, detection result card, Prepare Data button |
| Raw | Image grid, Pagination, ZoomModal, Download + Split + Reset buttons, status messages |
| Train Data | Image grid, Generate Annotation + Segmentation pipeline buttons |
| Test Data | Grid for test split images |
| Eval | Metrics table (mAP, precision, recall per split), per-class breakdown |
| BoundingBox Result | COCO bbox table with coords, image zoom |
| Segmentation Result | Segmentation polygon table with points, image zoom |
| Val Result | Run val pipeline button, results grid |
| Test Result | Run test pipeline button, results grid |
| Test Upload | Quick file upload for testing detection |
