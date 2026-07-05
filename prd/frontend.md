# Frontend PRD - SmartBin

**Stack:** Nuxt 3, Vue 3, TypeScript, Tailwind CSS

---

## 1. Sidebar Navigation

```
Main
├── Dashboard        → /dashboard

Pipeline
├── Dataset          → /raw/dataset
├── Preparation      → /raw/preparation
├── Training         → /raw/training
└── Deployment       → /raw/deployment
```

## 2. Pages

### /raw/dataset — Dataset
Two buttons: Load Dataset, Dataset Profiling. Datagrid shows class distribution table.

### /raw/preparation — Preparation
Two buttons: Convert Masks, Visualize Samples. Datagrid shows mask generation stats + sample images.

### /raw/training — Training
Three buttons: Train Model (configurable epochs/batch/lr), Show Curves, Evaluate. Datagrid shows training results + metrics.

### /raw/deployment — Deployment
Four buttons: Upload & Infer, Batch Test, Export Model, Final Verify. Datagrid shows per-class detection counts + advice + verification table.

### /dashboard — Live Detection
File upload → YOLO inference → bounding boxes + labels.

---

## 3. Color Scheme

| Token | Hex | Usage |
|-------|-----|-------|
| Primary (sidebar) | `#1F2937` | Sidebar background |
| Secondary | `#3B82F6` | Accent |
| Tertiary | `#2563EB` | Action buttons |

---

## 4. Components

| Component | Usage |
|-----------|-------|
| Pagination.vue | Image grid pagination |
| ZoomModal.vue | Fullscreen image popup |
| Toast.vue | Error/success notification |
