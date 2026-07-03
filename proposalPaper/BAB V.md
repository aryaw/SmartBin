# BAB V: PENUTUP

## 5.1 Kesimpulan

Berdasarkan penelitian yang telah dilakukan:

1. **Pipeline transformasi anotasi COCO→YOLO** berhasil diimplementasikan dengan normalisasi koordinat bounding box menggunakan scale factor berdasarkan resolusi aktual citra. Pemetaan 60 kategori TACO → 2 kelas (Organik/Non-Organik) menghasilkan dataset dengan 1.050 train, 225 val, 225 test. Konversi mempertahankan integritas spasial bounding box melalui scaling adaptif.

2. **Augmentasi data online** — mosaic 1.0, HSV perturbation, geometric transform, random erasing 0.4 — diterapkan dalam training pipeline untuk mengatasi keterbatasan dataset 1.500 citra. Close mosaic di epoch akhir mencegah distribusi shift. Augmentasi esensial untuk mencegah overfitting.

3. **Arsitektur YOLO26n** (2,7M parameter, 5,0 MB) diimplementasikan sebagai one-stage detector dengan backbone CSPNet, neck concatenation-based FPN, dan decoupled head. Loss function: CIoU (regresi bbox) + BCE (klasifikasi) + DFL (distribusi posisi) dengan bobot 7.5/0.5/1.5.

4. **GPU memory management** menggunakan FP16 inference mengurangi VRAM ~44%, batch size 16 feasible pada 12GB VRAM. `torch.cuda.set_per_process_memory_fraction()` mencegah OOM.

5. **Model dievaluasi** dengan mAP@0.5, mAP@0.5:0.95, precision, recall per split dan per-kelas untuk mengidentifikasi bias class imbalance (rasio organik:non-organik ~1:31).

## 5.2 Saran

### 5.2.1 Pengembangan Model

1. **Focal Loss:** Gantikan BCE dengan focal loss untuk mengatasi class imbalance — down-weight easy negatives, fokus ke hard positives.
2. **Class-weighted loss:** Naikkan bobot kelas organik di loss function.
3. **Model scaling:** Uji YOLO26s/m untuk lihat trade-off parameter vs akurasi.
4. **Cross-dataset validation:** Validasi pada TrashNet, WaDaBa, TACO-subset berbeda.

### 5.2.2 Pengembangan Data

1. **Data augmentation tambahan:** MixUp, CutMix untuk meningkatkan variasi.
2. **Oversampling organik:** Duplikasi sampel organik saat training.
3. **Pseudo-labeling:** Gunakan model trained untuk label data TACO unofficial.
4. **Ekspansi sumber:** Tambah citra dari lingkungan berbeda (pantai, pasar, jalan).

### 5.2.3 Pengembangan Aplikasi

1. Deployment REST API ke cloud untuk akses luas.
2. Real-time webcam detection via WebSocket streaming.

## 5.3 Kontribusi Penelitian

1. **Kontribusi Metodologis:** Pipeline data dan pelatihan deep learning untuk deteksi sampah biner menggunakan YOLO26n — mencakup transformasi anotasi, konfigurasi augmentasi, hyperparameter tuning, dan GPU memory optimization.

2. **Kontribusi Praktis:** REST API untuk inferensi deteksi sampah yang dapat diintegrasikan ke sistem pengelolaan sampah eksisting.

3. **Kontribusi Empiris:** Analisis dampak class imbalance, augmentasi, dan transfer learning pada dataset sampah TACO dengan YOLO26n.
