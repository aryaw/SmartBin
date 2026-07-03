# BAB III: METODE PENELITIAN

## 3.1 Alur Penelitian

Penelitian mengikuti pipeline sistematis empat tahap: (1) pra-pemrosesan data dan anotasi, (2) augmentasi dan pembagian data, (3) pelatihan model YOLO26n, (4) evaluasi dan inferensi.

```
TACO JSON ─→ COCO→YOLO Convert ─→ Split 70/15/15 ─→ Augmentasi Online ─→ YOLO26n Training ─→ Evaluasi
(datasource/)   (annotation_service.py)   (api/dataset/split)   (train.py: mosaic+HSV+geo)   (ultralytics)   (mAP/precision/recall)
```

## 3.2 Dataset dan Anotasi

### 3.2.1 Sumber Data

TACO (Trash Annotations in Context) - citra sampah lingkungan nyata dari `datasource/annotations.json`.

| Karakteristik | Nilai |
|---------------|-------|
| Jumlah citra | 1.500 (official) |
| Jumlah anotasi | 4.784 bounding box |
| Kategori asli | 60 kelas |
| Resolusi | 300×300 - 4000×3000 |

### 3.2.2 Pemetaan Taksonomi ke Dua Kelas

60 kategori TACO dipetakan ke biner:

| Kelas Baru | ID | Kategori TACO Asli |
|-----------|-----|--------------------|
| Organik | 0 | Food waste (category_id 25) |
| Non-Organik | 1 | Plastik, logam, kaca, kertas, tekstil, dll (59 kategori lain) |

Pemetaan diimplementasikan di `backend/app/routes/annotation.py` fungsi `dataset_split` saat menulis label file: `cat_map = 0 if ann["category_id"] == 25 else 1`.

### 3.2.3 Konversi ke Format YOLO

COCO JSON → format YOLO per-file `.txt`:

```
<class_id> <x_center_norm> <y_center_norm> <width_norm> <height_norm>
```

Konversi bounding box:
```python
sx = actual_width / coco_width    # scale factor X
sy = actual_height / coco_height  # scale factor Y
# COCO: [x, y, width, height] → YOLO: [x_center, y_center, width, height]
x_center = (x + width/2) * sx / actual_width   # → [0,1]
y_center = (y + height/2) * sy / actual_height # → [0,1]
w_norm = width * sx / actual_width              # → [0,1]
h_norm = height * sy / actual_height            # → [0,1]
```

### 3.2.4 Stratified Split 70/15/15

Random shuffle → 70% train (1.050), 15% val (225), 15% test (225). Implementasi di `POST /api/dataset/split`:

```python
random.shuffle(files)
n_train = int(n * 0.7); n_val = int(n * 0.15)
train = files[:n_train]; val = files[n_train:n_train+n_val]; test = files[n_train+n_val:]
```

Split endpoint self-cleans train/val/test dirs sebelum re-split. Raw dir preserved.

## 3.3 Augmentasi Data Online

Augmentasi diterapkan online saat training melalui Ultralytics engine:

| Parameter | Nilai | Dampak Komputasi |
|-----------|-------|-----------------|
| mosaic | 1.0 | 4× komputasi per forward |
| close_mosaic | min(10, epochs/2) | Stabilisasi akhir |
| hsv_h / s / v | 0.015 / 0.7 / 0.4 | Color jitter |
| scale | 0.5 | Multi-scale training |
| translate | 0.1 | Shift invariance |
| degrees | 10.0 | Rotasi kecil |
| shear | 2.0 | Affine |
| flipud/fliplr | 0.1 / 0.5 | Mirroring |
| erasing | 0.4 | Cutout |

Mosaic menggabung 4 citra menjadi 1, meningkatkan small object detection dan mengurangi overfitting.

## 3.4 Arsitektur dan Pelatihan Model

### 3.4.1 YOLO26n Architecture

```
Input (640×640×3)
    ↓
Backbone (CSPNet modified)
  ├── Conv-SiLU × N (stem)
  ├── CSPStage × 4 (residual blocks, channel doubling)
  └── SPPF (Spatial Pyramid Pooling Fast)
    ↓
Neck (Concatenation-based FPN)
  ├── Upsample + Concat (P5 → P4 → P3)
  └── Conv × 2 per level
    ↓
Head (Decoupled)
  ├── Classification branch: Conv → Conv → sigmoid
  └── Regression branch: Conv → Conv → bbox + IoU
    ↓
Output: S×S × (4 + 1 + C) per anchor
```

### 3.4.2 Hyperparameter

| Hyperparameter | Nilai | Dasar Pemilihan |
|----------------|-------|-----------------|
| Model | YOLO26n (pretrained COCO) | Transfer learning |
| Image size | 640×640 | YOLO default, 2-class ringan |
| Batch size | 16 | VRAM limit 12GB |
| Optimizer | SGD (momentum=0.937, lr=0.01) | YOLO default |
| Epochs | 100 (default) | Early stopping |
| Patience | 20 epoch | Tanpa perbaikan → stop |
| Device | CUDA (auto) | GPU priority |

### 3.4.3 Loss Function

$$\mathcal{L}_{total} = 7.5 \cdot \mathcal{L}_{CIoU} + 0.5 \cdot \mathcal{L}_{BCE} + 1.5 \cdot \mathcal{L}_{DFL}$$

Bobot box tinggi (7.5) menekankan regresi bounding box yang akurat. Bobot cls rendah (0.5) karena binary classification lebih sederhana. DFL memodelkan distribusi posisi.

### 3.4.4 GPU Memory Management

```python
vram_gb = float(os.getenv("VRAM_LIMIT_GB", "12"))
device = get_device()  # cuda:0
torch.cuda.set_per_process_memory_fraction(vram_gb * 1024**3 / total)
torch.cuda.empty_cache()  # setiap 16 gambar
half = True  # FP16 inference
```

FP16 menggunakan tensor half-precision mengurangi VRAM ~50% dengan minimal akurasi loss.

### 3.4.5 Model Output

Model terbaik disimpan ke `backend/models/best.pt` berdasarkan mAP@0.5 val terbaik.

## 3.5 Evaluasi Model

### 3.5.1 Metrik

- **mAP@0.5:** AP pada IoU threshold 0.5 (standar PASCAL VOC)
- **mAP@0.5:0.95:** Rata-rata AP pada IoU 0.5 hingga 0.95 (standar COCO)
- **Precision:** TP / (TP + FP)
- **Recall:** TP / (TP + FN)
- **Per-class AP:** AP terpisah untuk kelas Organik dan Non-Organik

### 3.5.2 Protokol Evaluasi

Evaluasi dilakukan pada setiap split (train/val/test) melalui YOLO `model.val()`. Sampel per-kelas untuk mengidentifikasi bias.

### 3.5.3 Inference Pipeline

```python
def detect_image(image_path):
    img = cv2.imread(str(image_path))
    results = model(img, device=device, verbose=False)
    # decode boxes, filter by conf_threshold=0.25
    for box in results.boxes:
        cls_id, conf = int(box.cls[0]), float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # draw with cvzone cornerRect + putTextRect
    return {"detected_objects": [...], "summary": {...}}
```

Confidence threshold 0.25. Boxes < threshold ditapis. Visualisasi: green untuk Organik, blue untuk Non-Organik.

## 3.6 Lingkungan Eksperimen

| Komponen | Spesifikasi |
|----------|-------------|
| GPU | NVIDIA CUDA (VRAM limit 12GB) |
| DL Framework | Ultralytics 8.4, PyTorch 2.x |
| Backend | FastAPI 0.115, Uvicorn |
| Frontend | Nuxt.js 3 (minimal) |
| Python | 3.12 |
| CV | OpenCV, cvzone, Supervision |

## 3.7 Implementasi Pipeline (REST API)

Endpoint untuk menjalankan pipeline:

| Endpoint | Fungsi | DS/ML Aspect |
|----------|--------|-------------|
| POST `/api/dataset/download` | Download TACO images → raw/ | Data collection |
| POST `/api/dataset/split` | COCO→YOLO convert + stratified split | Data preprocessing |
| POST `/api/dataset/pipeline/coco` | Generate COCO annotation viz | Data validation |
| POST `/api/dataset/pipeline/yolo` | YOLO inference on train | Model inference |
| POST `/api/dataset/pipeline/yolo/val` | YOLO inference on val | Model validation |
| POST `/api/dataset/pipeline/yolo/test` | YOLO inference on test | Model testing |
| GET `/api/dataset/evaluate` | mAP/precision/recall | Model evaluation |
| POST `/api/detect` | Upload → detect | Model deployment |
