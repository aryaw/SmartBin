# BAB V: PENUTUP

## 5.1 Kesimpulan

1. **Integrasi dua dataset** berhasil dilakukan: TACO (1.500 citra dengan anotasi COCO, 60 subkategori) dan Waste Classification Dataset lokal (2.939 citra, 18 subkategori) dipetakan ke 2 kelas (Organik/Non-Organik) dan digabung menjadi 3.973 citra.

2. **Pipeline pseudo-polygon mask generation** diimplementasikan dengan edge detection Otsu (66.4%) dan fallback geometris (33.6%). Dataset dikonversi ke format YOLO-seg dengan stratified split 70/15/15.

3. **Model YOLOv26m-seg** mencapai Box mAP@0.5 80.4% dan Mask mAP@0.5 49.7%. Kelas Non-Organik (Box mAP@0.5 83.6%) lebih baik dari Organik (77.2%) karena jumlah data lebih banyak.

4. **Aplikasi web** menyediakan deteksi langsung melalui upload gambar/video dengan output annotated image + summary + rekomendasi pembuangan.

## 5.2 Saran

### 5.2.1 Pengembangan Data

1. Kumpulkan lebih banyak data Organik untuk menyeimbangkan rasio kelas (target 1:1).
2. Anotasi mask manual pada subset data untuk meningkatkan kualitas segmentasi.

### 5.2.2 Pengembangan Model

1. Implementasi class-weighted loss untuk mengatasi class imbalance.
2. Fine-tuning pada YOLO26l atau YOLO26x untuk peningkatan akurasi.
3. Cross-dataset validation pada dataset sampah lain.

### 5.2.3 Pengembangan Aplikasi

1. Real-time webcam detection via WebSocket.
2. Deployment REST API ke production.

## 5.3 Kontribusi Penelitian

1. **Kontribusi Data:** Integrasi dua sumber dataset (TACO + Waste Classification Dataset) dengan mapping ke 2 kelas sesuai Pergub Bali No.47/2019.

2. **Kontribusi Metodologis:** Pipeline end-to-end untuk konversi dataset klasifikasi ke instance segmentation dengan pseudo-polygon mask.

3. **Kontribusi Praktis:** Aplikasi web untuk deteksi sampah Organik/Non-Organik dengan upload gambar/video.
