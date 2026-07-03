# ABSTRAK

**Deteksi dan Klasifikasi Sampah Organik dan Non-Organik Menggunakan Arsitektur YOLO26**

**Kelompok:**
- I Putu Gede Arya Wiratama — 252011032
- I Wayan Kardana — 252011033
- Ni Nyoman Yuli Purnamadewi — 252011030

Peningkatan volume sampah global mendorong kebutuhan solusi klasifikasi otomatis berbasis deep learning. Convolutional Neural Network (CNN) telah terbukti efektif mengekstraksi fitur visual secara hierarkis untuk deteksi objek. Penelitian ini mengembangkan pipeline deep learning end-to-end menggunakan arsitektur YOLO26n — generasi terbaru one-stage detector — untuk klasifikasi sampah organik dan non-organik. Dataset TACO (Trash Annotations in Context) dengan 1.500 citra dan 60 kategori ditransformasi menjadi dua kelas melalui pemetaan taksonomi. Arsitektur YOLO26n menggunakan backbone CSPNet termodifikasi dengan neck concatenation-based Feature Pyramid Network dan decoupled head untuk classification + regression branch secara paralel. Pipeline mencakup konversi anotasi COCO ke format YOLO dengan normalisasi koordinat bounding box, augmentasi citra online (mosaic=1.0, HSV perturbation, geometric transformasi, random erasing=0.4), dan stratified split 70/15/15. Pelatihan dikonfigurasi dengan batch size 16, image size 640, optimizer SGD, loss function CIoU + BCE + DFL dengan bobot 7.5/0.5/1.5, dan early stopping patience 20. GPU memory management diterapkan dengan FP16 inference dan VRAM cap 12GB via `torch.cuda.set_per_process_memory_fraction()`. Evaluasi menggunakan metrik mAP@0.5, mAP@0.5:0.95, precision, recall, serta analisis per-kelas. Aplikasi web dibangun sebagai antarmuka minimal untuk menjalankan inferensi melalui REST API.

**Kata kunci:** object detection, YOLO26, deep learning, CNN, bounding box regression, CIoU loss, TACO dataset, klasifikasi sampah

---

# ABSTRACT

**Organic and Non-Organic Waste Detection Using YOLO26 Architecture: A Deep Learning Pipeline from Annotation to Model Evaluation**

Global waste volume growth demands automated classification solutions based on deep learning. Convolutional Neural Networks have proven effective for hierarchical visual feature extraction in object detection. This research develops an end-to-end deep learning pipeline using the YOLO26n architecture — the latest one-stage detector generation — for organic and non-organic waste classification. The TACO dataset with 1,500 images and 60 categories was transformed into two classes through taxonomy mapping. The YOLO26n architecture employs a modified CSPNet backbone with concatenation-based Feature Pyramid Network neck and a decoupled head for parallel classification and regression branches. The pipeline includes COCO-to-YOLO annotation conversion with bounding box coordinate normalization, online image augmentation (mosaic=1.0, HSV perturbation, geometric transforms, random erasing=0.4), and stratified 70/15/15 split. Training is configured with batch size 16, image size 640, SGD optimizer, CIoU + BCE + DFL loss function with 7.5/0.5/1.5 weights, and early stopping patience 20. GPU memory management applies FP16 inference and 12GB VRAM cap via `torch.cuda.set_per_process_memory_fraction()`. Evaluation uses mAP@0.5, mAP@0.5:0.95, precision, recall, and per-class analysis. A minimal web interface provides inference access through REST API.

**Keywords:** object detection, YOLO26, deep learning, CNN, bounding box regression, CIoU loss, TACO dataset, waste classification
