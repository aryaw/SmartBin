# Frontend PRD - SmartBin

**Stack:** Nuxt 3, Vue 3, TypeScript, Tailwind CSS

---

## 1. Sidebar Navigation

```
Main
└── Dashboard       → /dashboard

Report
├── 1 Training Eval  → /train-eval
├── 2 Validation     → /val-result
├── 3 Test Results   → /test-result
└── 4 Inference      → /inference-export
```

## 2. Pages

| Route | Component | Description |
|-------|-----------|-------------|
| `/dashboard` | dashboard.vue | Upload file → detect → annotated result + Organik/Non-Organik counts + Run Full Pipeline button |
| `/test` | test.vue | Multi-file upload + batch detect + image preview before upload |
| `/result` | result.vue | Detection result detail page |
| `/train-eval` | train-eval.vue | Training evaluation metrics and plots |
| `/val-result` | val-result.vue | Validation results |
| `/test-result` | test-result.vue | Test set evaluation results |
| `/inference-export` | inference-export.vue | Batch inference + model export |
| `/boundingbox-result` | boundingbox-result.vue | Bounding box report |
| `/segmentation-result` | segmentation-result.vue | Segmentation result detail |
| `/kaggle-cms` | kaggle-cms.vue | Kaggle CMS pipeline management |
| `/dataset` | dataset.vue | Dataset browser |
| `/dataset-prep` | dataset-prep.vue | Dataset preparation tools |
| `/eval` | eval.vue | Evaluation overview |
| `/test-data` | test-data.vue | Test data browser |

## 3. Components

| Component | Description |
|-----------|-------------|
| FileUpload.vue | Drag & drop file upload zone |
| DetectionResult.vue | Detection result card (image + summary) |
| BoundingBoxReport.vue | Bounding box table with YOLO/CSV copy |
| ZoomModal.vue | Full-screen image zoom with pan |
| LoadingOverlay.vue | Loading spinner overlay |
| Pagination.vue | Pagination controls |
| Toast.vue | Notification toast |

## 4. Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `NUXT_PUBLIC_API_BASE` | `http://localhost:8000` | Backend API URL |
