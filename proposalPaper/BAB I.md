# BAB I: PENDAHULUAN

## 1.1 Latar Belakang

Permasalahan sampah telah mencapai tingkat mengkhawatirkan, khususnya di Bali sebagai destinasi wisata global. Bali menghasilkan ~1.340 ton sampah per hari (DLHK Bali), dengan 60% berasal dari sektor pariwisata. Komposisi sampah didominasi organik (60%) dan plastik (30%). Tantangan utama adalah pemilahan yang masih manual, tidak efisien, dan rawan error.

Pemerintah Bali melalui Pergub No.47/2019 menetapkan standar klasifikasi sampah menjadi 3 kategori: Organik, Anorganik (recyclable), dan Residu (landfill). Sistem ini bertujuan meningkatkan efektivitas daur ulang dan mengurangi beban Tempat Pemrosesan Akhir (TPA). Penelitian ini mengadopsi standar tersebut dengan klasifikasi 2 kelas utama (Organik/Non-Organik), di mana Non-Organik selanjutnya dipetakan ke subkelas Anorganik dan Residu.

Perkembangan deep learning dalam computer vision membuka paradigma baru otomatisasi deteksi dan klasifikasi objek. Instance segmentation memberikan informasi detail piksel-level untuk setiap objek. Arsitektur YOLO (You Only Look Once) sebagai one-stage detector telah berevolusi hingga generasi YOLOv26 yang mendukung instance segmentation.

Penelitian ini mengembangkan pipeline instance segmentation untuk klasifikasi sampah ke dalam 2 kelas utama (Organik/Non-Organik) menggunakan dua sumber dataset: TACO (Trash Annotations in Context, 1.500 citra dengan anotasi COCO, 60 subkategori) dan Waste Classification Dataset lokal (2.939 citra, 18 subkategori). Kedua sumber dipetakan ke kelas biner dan digabung menjadi 3.973 citra. Pipeline end-to-end mencakup pseudo-polygon mask generation, pelatihan YOLOv26m-seg, dan aplikasi web untuk inferensi.

Pendekatan 2 kelas dipilih karena kesederhanaan validasinya. Masyarakat umum dapat membedakan sampah organik (sisa makanan, daun, limbah dapur) dari non-organik (plastik, kertas, logam) tanpa keahlian khusus.

## 1.2 Rumusan Masalah

1. Bagaimana mengintegrasikan dua sumber dataset (TACO dan Waste Classification Dataset) ke dalam pipeline segmentasi dengan klasifikasi 2 kelas?
2. Bagaimana mengkonversi dataset klasifikasi ke format instance segmentation dengan pseudo-polygon mask?
3. Bagaimana performa segmentasi YOLOv26m-seg pada dataset gabungan?

## 1.3 Tujuan Penelitian

1. Mengintegrasikan dataset TACO dan Waste Classification Dataset untuk klasifikasi sampah Organik/Non-Organik.
2. Mengembangkan pipeline konversi dataset ke format YOLO-seg dengan pseudo-polygon mask.
3. Melatih dan mengevaluasi model YOLOv26m-seg untuk segmentasi 2 kelas sampah.

## 1.4 Manfaat Penelitian

Manfaat Akademis: Memberikan referensi implementasi instance segmentation untuk klasifikasi sampah 2 kelas dengan integrasi multi-dataset.

Manfaat Praktis: Menyediakan sistem deteksi sampah Organik/Non-Organik lengkap dengan aplikasi web untuk upload dan inferensi langsung.

Manfaat Lingkungan: Mendukung efektivitas pemilahan sampah melalui segmentasi otomatis berbasis computer vision.

## 1.5 Batasan Penelitian

1. Dataset menggunakan gabungan TACO (1.500 citra dengan anotasi COCO) dan Waste Classification Dataset lokal (2.939 citra) yang dipetakan ke 2 kelas: Organik dan Non-Organik.
2. Masker segmentasi adalah pseudo-polygon (generated, bukan annotated manual).
3. Arsitektur model terbatas pada YOLOv26m-seg (medium variant, 23.5M parameter).

## 1.6 Sistematika Penulisan

| Bab | Isi |
|-----|-----|
| Bab I Pendahuluan | Latar belakang, rumusan masalah, tujuan, manfaat, batasan |
| Bab II Tinjauan Pustaka | CNN, YOLO26 arsitektur, loss functions, augmentasi, metrik evaluasi |
| Bab III Metode Penelitian | Pipeline data, pseudo-mask generation, konfigurasi YOLOv26m-seg, lingkungan eksperimen |
| Bab IV Hasil dan Pembahasan | Dataset statistics, hasil training, per-class AP, pembahasan |
| Bab V Penutup | Kesimpulan, saran, kontribusi penelitian |

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
