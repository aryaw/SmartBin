# BAB III: METODE PENELITIAN

## 3.1 Alur Penelitian

Penelitian mengikuti pipeline:

1. Dataset Preparation: Integrasi dua sumber dataset → mapping ke 2 kelas → flat folder structure
2. Pseudo-Mask Generation: Edge detection + fallback geometris
3. Stratified Split: Train/val/test 70/15/15
4. Training: YOLOv26m-seg dengan pseudo-mask labels
5. Evaluasi: Box & Mask mAP, precision, recall
6. Deployment: Model disalin ke backend untuk inferensi via web

---

## 3.2 Dataset

### 3.2.1 Sumber Data

Dua sumber dataset digunakan:

| Sumber | Jumlah Citra | Subkategori | Tipe Anotasi | Referensi |
|--------|-------------|-------------|--------------|-----------|
| TACO (Trash Annotations in Context) | 1.500 | 60 | COCO format (segmentation polygon, bbox) | [29] |
| Waste Classification Dataset (lokal) | 2.939 | 18 | Klasifikasi per folder | [30] |

### 3.2.2 Mapping ke 2 Kelas

TACO categories dipetakan menggunakan `ORGANIC_CATEGORIES` dari config:
- **Organik (kelas 0):** hanya Food waste (category_id=25)
- **Non-Organik (kelas 1):** 59 kategori lainnya (aluminium, baterai, botol plastik, gelas, kaleng, kardus, kertas, dll)

Waste Classification Dataset dipetakan:
- **Organik:** coffee_tea_bags, egg_shells, food_scraps, kitchen_waste, yard_trimmings
- **Non-Organik:** e-waste, cans, glass, paper, plastic_bottles, batteries, paints, pesticides, ceramic, diapers, plastics_bags, sanitary_napkin, styrofoam

Non-Organik selanjutnya dipetakan ke subkelas Pergub No.47/2019:
- **Anorganik (recyclable):** e-waste, cans, glass, paper, plastic_bottles
- **Residu (landfill):** batteries, paints, pesticides, ceramic, diapers, plastics_bags, sanitary_napkin, styrofoam

### 3.2.3 Dataset Final

| Kelas | Jumlah Citra |
|-------|-------------|
| Organik | 684 |
| Non-Organik | 3.289 |
| **Total** | **3.973** |

### 3.2.4 Stratified Split 70/15/15

```
Train: ~2.781 images (70%)
Val:    ~596 images (15%)
Test:   ~596 images (15%)
```

---

## 3.3 Pseudo-Polygon Mask Generation

Karena dataset adalah klasifikasi (tanpa anotasi mask), pipeline mengenerate polygon mask:

### 3.3.1 Edge Detection (strategi utama)

1. Convert RGB → Grayscale
2. Gaussian Blur (5x5)
3. Otsu thresholding → binary mask
4. Invert if mean > 127
5. Morphological close (5x5, 2 iter) + open (1 iter)
6. Find contours, ambil largest
7. Skip if area < 20% total image
8. Approximate polygon (epsilon = 0.01 * arcLength)
9. Sample to 24 points, normalize ke [0, 1]

### 3.3.2 Fallback Geometris

Jika edge detection gagal:
- 60%: Elliptical polygon (20 titik)
- 40%: Rounded rectangle polygon (20 titik)

### 3.3.3 Format Label YOLO-seg

```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

Semua koordinat dinormalisasi ke [0, 1].

---

## 3.4 Arsitektur YOLOv26m-seg

| Fitur | Status |
|-------|--------|
| MuSGD Optimizer | Active (SGD + Muon hybrid) |
| Multi-Scale Proto Modules | Active |
| NMS-Free End-to-End | Active |
| Anchor-Free Detection | Active |

### 3.4.1 Hyperparameter

| Hyperparameter | Nilai |
|----------------|-------|
| Model | yolo26m-seg.pt |
| Image size | 640 |
| Batch size | 16 |
| Epochs | 80 (patience=40) |
| Optimizer | SGD (MuSGD) |
| Learning rate | 0.001 |
| Box loss weight | 7.5 |
| Cls loss weight | 0.5 |
| Mask ratio | 2 |

### 3.4.2 Augmentasi

| Augmentasi | Nilai | Referensi |
|------------|-------|-----------|
| Mosaic | 1.0 | [14][8] |
| Mixup | 0.2 | [15] |
| Copy-paste | 0.15 | [16] |
| HSV jitter | default | [17][10] |
| Flip LR | 50% | [10] |

---

## 3.5 Evaluasi

### 3.5.1 Metrik Utama

- Box mAP@0.5 dan mAP@0.5:0.95
- Mask mAP@0.5 dan mAP@0.5:0.95
- Per-class Box & Mask AP@50
- Precision dan Recall

### 3.5.2 Recycling Advice

| Kategori | Subkelas | Advice |
|----------|----------|--------|
| Organik | - | Buang ke Tempat Sampah Organik |
| Non-Organik | Anorganik | Buang ke Tempat Sampah Non-Organik |

---

## 3.6 Lingkungan Eksperimen

| Komponen | Spesifikasi |
|----------|-------------|
| CPU | AMD Ryzen 7 8700F (8 core, 16 thread) |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) |
| CUDA | 13.0, Driver 595.71.05 |
| DL Framework | Ultralytics 8.4.84, PyTorch 2.12.1 |
| Backend | FastAPI, Uvicorn |
| Frontend | Nuxt.js 3 (Vue 3) |
| Python | 3.12 |
| OS | Ubuntu 25.10 |
| Training time | ~2.5 jam (80 epoch) |

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
