# ABSTRAK

**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

Peningkatan volume sampah global mendorong kebutuhan solusi klasifikasi otomatis berbasis deep learning. Penelitian ini mengembangkan pipeline instance segmentation end-to-end menggunakan arsitektur YOLOv26m-seg untuk mengklasifikasikan sampah ke dalam 2 kelas utama: Organik dan Non-Organik sesuai standar Pemerintah Bali Pergub No.47/2019 [31]. Dataset menggunakan dua sumber: TACO [29] (Trash Annotations in Context, 1.500 citra dengan anotasi COCO, 60 subkategori) dan Waste Classification Dataset [30] lokal (2.939 citra, 18 subkategori). Kedua sumber dipetakan ke 2 kelas dan digabung menjadi 3.973 citra (684 Organik, 3.289 Non-Organik). Karena dataset merupakan klasifikasi (tanpa mask), pipeline menghasilkan pseudo-polygon mask menggunakan edge detection berbasis Otsu thresholding [1] dan fallback geometris. Arsitektur YOLOv26m-seg [8] menggunakan fitur MuSGD Optimizer, Multi-Scale Proto Modules, dan NMS-Free End-to-End. Pelatihan dikonfigurasi dengan 80 epoch, batch 16, image size 640. Evaluasi menggunakan metrik box dan mask mAP@0.5. Model mencapai Box mAP@0.5 80.4%, Mask mAP@0.5 49.7%. Aplikasi web menyediakan deteksi langsung melalui upload gambar/video dengan rekomendasi pembuangan sesuai klasifikasi.

**Kata kunci:** instance segmentation, YOLOv26, waste classification, TACO dataset, deep learning, Organik Non-Organik

# ABSTRACT

**Waste Detection and Classification Using YOLOv26m-seg for Instance Segmentation (2 Classes: Organik/Non-Organik)**

Global waste volume growth demands automated classification solutions based on deep learning. This research develops an end-to-end instance segmentation pipeline using YOLOv26m-seg architecture to classify waste into 2 main classes: Organik (Organic) and Non-Organik (Non-Organic) aligned with Bali regulation Pergub No.47/2019. The dataset combines two sources: TACO (Trash Annotations in Context, 1,500 images with COCO annotations, 60 subcategories) and a local Waste Classification Dataset (2,939 images, 18 subcategories). Both are mapped to 2 classes and merged into 3,973 images (684 Organic, 3,289 Non-Organic). Since the dataset is classification-only (no masks), the pipeline generates pseudo-polygon masks using Otsu thresholding edge detection and geometric fallbacks. YOLOv26m-seg architecture leverages MuSGD Optimizer, Multi-Scale Proto Modules, and NMS-Free End-to-End. Training configured with 80 epochs, batch 16, image size 640. Evaluation uses box and mask mAP@0.5. Model achieves Box mAP@0.5 80.4%, Mask mAP@0.5 49.7%. A web application provides direct detection via image/video upload with disposal recommendations.

**Keywords:** instance segmentation, YOLOv26, waste classification, TACO dataset, deep learning, Organic Non-Organic

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
