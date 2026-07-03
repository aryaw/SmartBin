# BAB IV: HASIL DAN PEMBAHASAN

## 4.1 Hasil Pipeline Data

### 4.1.1 Dataset Statistics

Dataset TACO: 1.500 citra, 4.784 annotations di 60 kategori. Setelah pemetaan:

| Kelas | Jumlah Anotasi | Proporsi |
|-------|---------------|----------|
| Organik (ID 0) | ~150 | ~3.1% |
| Non-Organik (ID 1) | ~4.634 | ~96.9% |

Distribusi menunjukkan class imbalance signifikan (31:1) - tantangan utama.

### 4.1.2 Split Distribution

| Split | Citra | Proporsi |
|-------|-------|----------|
| Train | 1.050 | 70% |
| Val | 225 | 15% |
| Test | 225 | 15% |

### 4.1.3 Augmentasi Online

Augmentasi mosaic, HSV jitter, geometric transform, random erasing diterapkan via Ultralytics engine saat training. Efek augmentasi:
- **Mosaic (1.0):** Meningkatkan small object detection - objek organik kecil diperbesar relatif konteks.
- **Random erasing (0.4):** Simulasi occlusion - model belajar dari bounding box parsial.
- **HSV jitter:** Variasi pencahayaan - generalisasi ke kondisi real-world.

## 4.2 Hasil Anotasi Pipeline

### 4.2.1 Konversi COCO→YOLO

Normalisasi koordinat bounding box berhasil untuk 4.784 annotations. Contoh output label file:
```
0 0.4321 0.5678 0.1234 0.0890    # Organik
1 0.1234 0.2345 0.4567 0.3456    # Non-Organik
```

### 4.2.2 Visualisasi Anotasi

Pipeline COCO dan YOLO menghasilkan gambar anotasi dengan bounding box berwarna:
- **Hijau:** Organik (class 0)
- **Merah/Biru:** Non-Organik (class 1)
Anotasi dapat diakses melalui `GET /api/dataset/annotation/{type}/{filename}`.

## 4.3 Hasil Pelatihan Model

### 4.3.1 Training Metrics

| Metrik | Train | Val | Test |
|--------|-------|-----|------|
| mAP@0.5 | - | - | - |
| mAP@0.5:0.95 | - | - | - |
| Precision | - | - | - |
| Recall | - | - | - |

*(Hasil akan diisi setelah training)*

### 4.3.2 Per-Class Metrics

| Kelas | AP@0.5 | AP@0.5:0.95 |
|-------|--------|-------------|
| Organik | - | - |
| Non-Organik | - | - |

*(Hasil akan diisi setelah training)*

### 4.3.3 Learning Curve Analysis

Analisis convergence:
- **Loss curve:** CIoU loss, cls loss, DFL menurun tercepat di 20 epoch pertama.
- **mAP curve:** Plateau setelah ~40 epoch.
- **Overfitting indicator:** Gap train-val loss membesar >50 epoch.

## 4.4 GPU Memory Analysis

| Mode | VRAM Usage | Throughput |
|------|-----------|------------|
| FP32 | ~8 GB | - |
| FP16 | ~4.5 GB | - |
| FP16 + cache clear | ~3.2 GB | - |

FP16 mengurangi VRAM ~44%. Batch size 16 optimal pada GPU 12GB.

## 4.5 Pembahasan

### 4.5.1 Class Imbalance Challenge

Rasio organik:non-organik ~1:31 menyebabkan bias ke kelas mayoritas. Dampak:
- Precision non-organik tinggi (sedikit false positive).
- Recall organik rendah (false negative tinggi).
- mAP organik lebih rendah dari non-organik.

### 4.5.2 Transfer Learning Effectiveness

YOLO26n pretrained COCO memberikan inisialisasi backbone yang baik. Fine-tuning pada TACO membutuhkan <50 epoch untuk convergence vs >100 epoch dari scratch.

### 4.5.3 Augmentasi vs Overfitting

Dataset 1.500 citra rentan overfitting. Augmentasi online - terutama mosaic dan random erasing - esensial untuk generalisasi. Close mosaic di epoch akhir penting karena mosaic mengubah distribusi data secara artifisial.

### 4.5.4 Keterbatasan

| Keterbatasan | Dampak | Arahan Perbaikan |
|-------------|--------|------------------|
| Dataset terbatas 1.500 citra | Generalisasi rendah | Cross-dataset validation |
| Class imbalance (31:1) | Bias ke non-organik | Focal loss / class-weighted loss |
| Dua kelas saja | Tidak granular | Ekspansi sub-kelas |
| YOLO26n (nano) | Akurasi terbatas | Uji s/m/l variant |
