# BAB I: PENDAHULUAN

## 1.1 Latar Belakang

Permasalahan sampah global telah mencapai tingkat mengkhawatirkan. Produksi sampah global mencapai 2,01 miliar ton per tahun dan diproyeksikan meningkat 70% pada 2050 (Kaza et al., 2018). Di Indonesia, timbulan sampah nasional mencapai 175.000 ton per hari. Tantangan utama adalah pemilahan yang masih manual, tidak efisien, dan rawan error.

Perkembangan deep learning dalam computer vision membuka paradigma baru otomatisasi deteksi dan klasifikasi objek. Arsitektur Convolutional Neural Network (CNN) mengekstraksi fitur visual secara hierarkis melalui konvolusi, aktivasi non-linear, dan pooling (LeCun et al., 2015). Dalam deteksi objek, pendekatan one-stage detector seperti YOLO (You Only Look Once) memperlakukan deteksi sebagai masalah regresi spasial - memprediksi bounding box dan probabilitas kelas dalam satu forward pass (Redmon et al., 2016). Pendekatan ini memungkinkan inferensi real-time tanpa region proposal network.

Evolusi arsitektur YOLO menunjukkan peningkatan dalam efisiensi parameter dan kualitas fitur. YOLO26 sebagai generasi terbaru memperkenalkan peningkatan backbone (CSPNet termodifikasi dengan residual shortcut), neck (concatenation-based Feature Pyramid Network untuk fusi multi-skala), dan decoupled head (classification branch dan regression branch terpisah). Varian YOLO26n memiliki 2,7M parameter (5,0 MB) dengan mAP 40,1% pada COCO - keseimbangan optimal antara akurasi dan kecepatan.

Penelitian deteksi sampah berbasis YOLO telah menunjukkan hasil menjanjikan pada dataset spesifik. Namun, masih terdapat celah pada integrasi pipeline deep learning end-to-end - dari transformasi anotasi COCO ke YOLO, augmentasi data terstruktur, manajemen GPU memory, hingga evaluasi metrik komprehensif.

Berdasarkan celah tersebut, penelitian ini mengusulkan pipeline deep learning end-to-end untuk deteksi sampah menggunakan YOLO26n dengan fokus pada aspek data science dan model optimization.

## 1.2 Rumusan Masalah

1. Bagaimana merancang pipeline transformasi anotasi COCO ke format YOLO - meliputi normalisasi koordinat bounding box, pemetaan taksonomi multi-kelas ke biner, dan stratified split?
2. Bagaimana arsitektur YOLO26n (backbone CSPNet, neck FPN, decoupled head) bekerja pada dataset sampah dengan class imbalance?
3. Bagaimana konfigurasi augmentasi data dan hyperparameter (batch size, learning rate, loss weights) memengaruhi metrik mAP@0.5, mAP@0.5:0.95, precision, recall?
4. Bagaimana manajemen GPU memory (FP16 inference, VRAM capping, cache clearing) memungkinkan training pada GPU 12GB?

## 1.3 Tujuan Penelitian

1. Mengembangkan pipeline konversi anotasi dan augmentasi data untuk deteksi sampah menggunakan YOLO26n.
2. Mengimplementasikan dan mengevaluasi pelatihan model dengan konfigurasi hyperparameter terstruktur.
3. Menganalisis performa model melalui metrik deteksi objek komprehensif.

## 1.4 Manfaat Penelitian

Manfaat Akademis: Memberikan referensi implementasi deteksi objek untuk klasifikasi sampah menggunakan arsitektur YOLO26, mencakup pipeline data, konfigurasi training, dan metrik evaluasi.

Manfaat Praktis: Menyediakan REST API untuk inferensi yang dapat diintegrasikan ke sistem pengelolaan sampah.

Manfaat Lingkungan: Mendukung efektivitas pemilahan sampah melalui otomatisasi deteksi berbasis computer vision.

## 1.5 Batasan Penelitian

1. Klasifikasi dibatasi pada dua kelas: organik dan non-organik (binary classification).
2. Dataset menggunakan 1.500 citra dari TACO official dataset (60 kategori asli ditransformasi).
3. Arsitektur model terbatas pada YOLO26n (varian nano, 2,7M parameter).
4. Evaluasi dilakukan pada dataset TACO tanpa pengujian cross-dataset.
5. Aplikasi web sebagai antarmuka minimal, bukan fokus utama penelitian.

## 1.6 Sistematika Penulisan

Penelitian terdiri dari lima bab. Bab I menguraikan latar belakang, rumusan masalah, tujuan, dan batasan. Bab II membahas landasan teori CNN, arsitektur YOLO26, loss functions (CIoU, BCE, DFL), augmentasi data, dan metrik evaluasi (mAP, precision, recall). Bab III menjelaskan pipeline data, arsitektur model, metodologi pelatihan, dan protokol evaluasi. Bab IV menyajikan hasil eksperimen dan pembahasan. Bab V berisi kesimpulan dan saran.
