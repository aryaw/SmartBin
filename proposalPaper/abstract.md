# ABSTRAK

**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

Peningkatan volume sampah global mendorong kebutuhan solusi klasifikasi otomatis berbasis deep learning. Penelitian ini mengembangkan pipeline instance segmentation end-to-end menggunakan arsitektur YOLOv26m-seg untuk mengklasifikasikan sampah ke dalam 2 kelas utama: Organik dan Non-Organik sesuai standar Pemerintah Bali Pergub No.47/2019. Dataset menggunakan dua sumber: TACO (Trash Annotations in Context, 1.500 citra dengan anotasi COCO, 60 subkategori) dan Waste Classification Dataset lokal (2.939 citra, 18 subkategori). Kedua sumber dipetakan ke 2 kelas dan digabung menjadi 3.973 citra (684 Organik, 3.289 Non-Organik). Karena dataset merupakan klasifikasi (tanpa mask), pipeline menghasilkan pseudo-polygon mask menggunakan edge detection berbasis Otsu thresholding dan fallback geometris. Arsitektur YOLOv26m-seg menggunakan fitur MuSGD Optimizer, Multi-Scale Proto Modules, dan NMS-Free End-to-End. Pelatihan dikonfigurasi dengan 80 epoch, batch 16, image size 640. Evaluasi menggunakan metrik box dan mask mAP@0.5. Model mencapai Box mAP@0.5 80.4%, Mask mAP@0.5 49.7%. Aplikasi web menyediakan deteksi langsung melalui upload gambar/video dengan rekomendasi pembuangan sesuai klasifikasi.

**Kata kunci:** instance segmentation, YOLOv26, waste classification, TACO dataset, deep learning, Organik Non-Organik

# ABSTRACT

**Waste Detection and Classification Using YOLOv26m-seg for Instance Segmentation (2 Classes: Organik/Non-Organik)**

Global waste volume growth demands automated classification solutions based on deep learning. This research develops an end-to-end instance segmentation pipeline using YOLOv26m-seg architecture to classify waste into 2 main classes: Organik (Organic) and Non-Organik (Non-Organic) aligned with Bali regulation Pergub No.47/2019. The dataset combines two sources: TACO (Trash Annotations in Context, 1,500 images with COCO annotations, 60 subcategories) and a local Waste Classification Dataset (2,939 images, 18 subcategories). Both are mapped to 2 classes and merged into 3,973 images (684 Organic, 3,289 Non-Organic). Since the dataset is classification-only (no masks), the pipeline generates pseudo-polygon masks using Otsu thresholding edge detection and geometric fallbacks. YOLOv26m-seg architecture leverages MuSGD Optimizer, Multi-Scale Proto Modules, and NMS-Free End-to-End. Training configured with 80 epochs, batch 16, image size 640. Evaluation uses box and mask mAP@0.5. Model achieves Box mAP@0.5 80.4%, Mask mAP@0.5 49.7%. A web application provides direct detection via image/video upload with disposal recommendations.

**Keywords:** instance segmentation, YOLOv26, waste classification, TACO dataset, deep learning, Organic Non-Organic
