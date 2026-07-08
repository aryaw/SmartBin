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

**Daftar Referensi:**

[1] Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. SMC*, 9(1), 62-66.
[2] Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. *CVGIP*, 30(1), 32-46.
[3] Douglas, D.H. & Peucker, T.K. (1973). Algorithms for the reduction of points. *Cartographica*, 10(2), 112-122.
[4] Bradski, G. & Kaehler, A. (2008). *Learning OpenCV*. O'Reilly Media.
[5] Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
[6] Soille, P. (2003). *Morphological Image Analysis* (2nd ed.). Springer.
[7] OpenCV (2024). OpenCV 4.13.0 Documentation. https://docs.opencv.org/4.13.0/
[8] Ultralytics (2023). YOLOv8 Documentation. https://docs.ultralytics.com/
[9] IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*. IEEE Std 754-2019.
[10] Shorten, C. & Khoshgoftaar, T.M. (2019). A survey on image data augmentation. *J. Big Data*, 6(1), 60.
[14] Bochkovskiy, A. et al. (2020). YOLOv4. *arXiv:2004.10934*.
[15] Zhang, H. et al. (2018). mixup. *Proc. ICLR*.
[16] Ghiasi, G. et al. (2021). Simple copy-paste. *Proc. CVPR*.
[17] Redmon, J. et al. (2016). You only look once. *Proc. CVPR*, 779-788.
[29] Proença, P.F. & Simões, P. (2020). TACO. *arXiv:2003.06975*.
[30] Kaggle (2020). Waste Classification Dataset. https://www.kaggle.com/datasets/phenomsg/waste-classification
[31] Pergub Bali No.47/2019. Pengelolaan Sampah Berbasis Sumber.
[32] DLHK Bali (2023). Data Produksi Sampah Harian Provinsi Bali.
