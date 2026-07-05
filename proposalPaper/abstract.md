# ABSTRAK

**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation dengan 18 Subkategori**

Peningkatan volume sampah global mendorong kebutuhan solusi klasifikasi otomatis berbasis deep learning. Penelitian ini mengembangkan pipeline instance segmentation end-to-end menggunakan arsitektur YOLOv26m-seg untuk mengklasifikasikan sampah ke dalam 18 subkategori dari 4 kategori utama: Hazardous, Non-Recyclable, Organic, dan Recyclable. Dataset phenomsg/waste-classification dengan ~2.917 citra digunakan sebagai sumber data. Karena dataset merupakan klasifikasi (tanpa mask), pipeline menghasilkan pseudo-polygon mask menggunakan edge detection berbasis Otsu thresholding (~83%) dan fallback geometris (ellipse/rounded rectangle, ~17%). Arsitektur YOLOv26m-seg menggunakan fitur MuSGD Optimizer (hybrid SGD + Muon), Semantic Segmentation Loss, Multi-Scale Proto Modules, dan NMS-Free End-to-End. Pelatihan dikonfigurasi dengan 120 epoch, batch size 16, image size 640, optimizer SGD, dan early stopping patience 20. Augmentasi mencakup mosaic, mixup, copy-paste, dan transformasi geometrik. Evaluasi menggunakan metrik mask mAP@0.5, mAP@0.5:0.95, precision, recall, serta analisis per-subkategori. Aplikasi web menyediakan antarmuka pipeline 12 langkah dari eksplorasi dataset hingga verifikasi akhir.

**Kata kunci:** instance segmentation, YOLOv26, waste classification, deep learning, CNN, polygon mask

# ABSTRACT

**Waste Detection and Classification Using YOLOv26m-seg for Instance Segmentation with 18 Subcategories**

Global waste volume growth demands automated classification solutions based on deep learning. This research develops an end-to-end instance segmentation pipeline using YOLOv26m-seg architecture to classify waste into 18 subcategories across 4 main categories: Hazardous, Non-Recyclable, Organic, and Recyclable. The phenomsg/waste-classification dataset with ~2,917 images serves as data source. Since the dataset is classification-only (no masks), the pipeline generates pseudo-polygon masks using Otsu thresholding edge detection (~83%) and geometric fallbacks (ellipse/rounded rectangle, ~17%). YOLOv26m-seg architecture leverages MuSGD Optimizer (SGD + Muon hybrid), Semantic Segmentation Loss, Multi-Scale Proto Modules, and NMS-Free End-to-End. Training is configured with 120 epochs, batch size 16, image size 640, SGD optimizer, and early stopping patience 20. Augmentation includes mosaic, mixup, copy-paste, and geometric transforms. Evaluation uses mask mAP@0.5, mAP@0.5:0.95, precision, recall, and per-subcategory analysis. A web application provides a 12-step pipeline interface from dataset exploration to final verification.

**Keywords:** instance segmentation, YOLOv26, waste classification, deep learning, CNN, polygon mask
