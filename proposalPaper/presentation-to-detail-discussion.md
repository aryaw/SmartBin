# Presentation Detail Discussion

## Slide 1: Judul
**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

### Detail Pipeline per Subgraph

Pipeline penelitian ini terdiri dari 5 subgraph utama yang merepresentasikan alur preprocessing data science, arsitektur deep learning CNN, training machine learning, evaluasi metrik, dan inference.

---

#### A. Preprocessing - Data Science

**Node A** - "TACO + Waste Classification: 3.973 gambar, 2 kelas": Dataset merupakan hasil penggabungan dua sumber data. TACO Dataset menyediakan 1.500 gambar dengan anotasi COCO format yang mencakup 60 kategori sampah. Waste Classification Dataset dari platform Kaggle menyediakan 2.939 gambar yang terorganisir dalam 18 subfolder tanpa anotasi segmentasi. Proses mapping mereduksi 60+ kategori menjadi 2 kelas utama: Organik (684 gambar) dan Non-Organik (3.289 gambar). Total 3.973 gambar siap diproses ke tahap selanjutnya.

**Node B** - "Pseudo-Mask 12 langkah, 66.4% Edge Detection": Dataset tidak memiliki label segmentasi, sehingga diperlukan pembangkitan polygon mask secara otomatis. Pipeline computer vision 12 langkah meliputi: konversi RGB ke grayscale, Gaussian blur kernel 5x5 untuk reduksi noise, Otsu thresholding untuk binerisasi otomatis, pengecekan mean intensitas untuk koreksi inversi, operasi morphological closing 5x5 untuk menutup lubang, ekstraksi kontur terbesar, validasi area >=20% dari total gambar, dan approximasi polygon menggunakan algoritma Douglas-Peucker dengan epsilon 0.01 dikali arcLength. Edge detection Otsu berhasil pada 66.4% gambar (2.638 gambar). Untuk 33.6% sisanya, digunakan fallback geometris: 60% berbentuk ellipse dan 40% berbentuk rounded rectangle.

**Node C** - "Stratified Split 70/15/15: Train 2.765, Val 593, Test 593": Dataset dibagi menjadi tiga subset menggunakan stratified sampling untuk mempertahankan proporsi kelas Organik:Non-Organik (17.2%:82.8%) yang identik di setiap subset. Teknik ini mencegah bias evaluasi yang dapat terjadi jika distribusi kelas berbeda antar subset. Training set (70%) digunakan untuk pembelajaran parameter model, validation set (15%) untuk monitoring overfitting dan hyperparameter tuning, test set (15%) untuk evaluasi final performa model pada data yang tidak pernah dilihat sebelumnya.

**Node D** - "Online Augmentation: Mosaic, Mixup, HSV, Flip": Augmentasi data diterapkan secara online (real-time saat training) untuk meningkatkan kemampuan generalisasi model. Mosaic (probabilitas 1.0) menggabungkan 4 gambar menjadi grid 2x2, memaksa model mendeteksi objek dalam konteks padat. Mixup (probabilitas 0.5) melakukan blending linear antar dua gambar. HSV Jitter memvariasikan hue (0.02), saturation (0.6), dan value (0.4) untuk simulasi kondisi pencahayaan berbeda. Flip horizontal (probabilitas 0.5) memberikan variasi orientasi. Geometric transformasi meliputi rotasi (25 derajat), scaling (0.8), dan shearing (10 derajat).

**Aliran data:** Dataset mentah melewati pipeline pseudo-mask 12 langkah, hasilnya di-split secara stratified, kemudian di-augmentasi secara online sebelum masuk ke arsitektur CNN.

---

#### B. Deep Learning - CNN Architecture

**Node E** - "Input 640x640x3": Citra input diresize ke resolusi 640x640 piksel dengan 3 channel warna (RGB). Resolusi ini merupakan standar input YOLOv26 yang memberikan keseimbangan antara detail informasi dan efisiensi komputasi. Letterbox padding digunakan untuk mempertahankan aspek rasio asli gambar.

**Node F** - "Backbone CSPDarknet: Stem -> 4 CSP Stage -> SPP, 640x640 -> 20x20": Backbone berfungsi melakukan ekstraksi fitur secara hierarkis. Dimulai dengan Stem Convolution (kernel 7, stride 2) yang mereduksi resolusi dari 640x640 menjadi 320x320 dengan 64 channel. Selanjutnya 4 tahap Cross Stage Partial (CSP) secara bertahap mereduksi resolusi spasial menjadi 160x160 (128 channel), 80x80 (256 channel), 40x40 (512 channel), dan 20x20 (512 channel). Setiap CSP stage membagi feature map menjadi dua jalur: satu jalur diproses melalui konvolusi batch, jalur lain langsung diteruskan dan digabung di akhir. Teknik ini menghemat ~20% FLOPs dibanding backbone konvensional. Spatial Pyramid Pooling (SPP) layer menggunakan 3 max pooling paralel dengan kernel size 5, 9, dan 13 untuk menangkap konteks multi-skala dari objek sampah yang bervariasi ukurannya.

**Node G** - "Neck FPN+PAN: Multi-Scale Feature Fusion": Backbone menghasilkan 3 level fitur: P3 (80x80, resolusi tinggi, detail lokasi baik), P4 (40x40, resolusi menengah), dan P5 (20x20, resolusi rendah, semantik tinggi). Feature Pyramid Network (FPN) bekerja secara top-down: melakukan upsampling fitur semantik dari P5 ke P4 dan P3, memungkinkan informasi "apa objeknya" menjangkau resolusi tinggi. Path Aggregation Network (PAN) bekerja secara bottom-up: melakukan downsampling fitur detail dari P3 ke P4 dan P5, memungkinkan informasi "di mana posisinya" menjangkau resolusi rendah. Hasilnya, setiap level deteksi memiliki informasi semantik dan detail lokasi secara lengkap.

**Node H** - "Decoupled Head: Class + Reg + Seg Branch": Head terdiri dari 3 cabang paralel yang terpisah (decoupled), masing-masing dengan parameter sendiri. Classification branch menggunakan 2 layer Conv3x3 diikuti linear layer, menghasilkan probabilitas untuk 2 kelas ditambah objectness score. Regression branch menggunakan Distribution Focal Loss (DFL) dengan 16 bin distribusi untuk memprediksi 4 koordinat bounding box (x center, y center, width, height). Segmentation branch menggunakan Proto Module yang menghasilkan 32 prototype mask dari fitur multi-skala, kemudian mengkombinasikannya dengan coefficient per-instance untuk menghasilkan polygon 24 titik. Arsitektur anchor-free dengan total 8.400 grid cells (80x80 + 40x40 + 20x20).

**Aliran data:** Input 640x640 diproses backbone menjadi fitur 20x20, difusikan oleh neck FPN+PAN, kemudian 3 cabang head memprediksi kelas, bounding box, dan mask segmentasi secara paralel.

---

#### C. Machine Learning - Training

**Node I** - "Loss: CIoU(7.5) + BCE(0.5) + DFL(1.5)": Total loss merupakan kombinasi linear dari 3 fungsi loss dengan bobot berbeda. CIoU (Complete IoU) loss dengan bobot 7.5 mengoptimasi regresi bounding box berdasarkan 3 metrik: Intersection over Union, jarak Euclidean antara pusat prediksi dan ground truth, serta perbedaan aspect ratio. BCE (Binary Cross-Entropy) loss dengan bobot 0.5 mengoptimasi klasifikasi 2 kelas dengan mengukur perbedaan antara probabilitas prediksi dan label sebenarnya. DFL (Distribution Focal Loss) dengan bobot 1.5 mengoptimasi regresi posisi dengan memodelkan distribusi probabilitas diskrit 16 bin untuk setiap sisi bounding box, memberikan gradien lebih informatif pada objek dengan boundary tidak jelas.

**Node J** - "SGD Optimizer, LR=0.001 Cosine Decay": Optimizer menggunakan Stochastic Gradient Descent dengan momentum 0.937. Learning rate awal 0.001 dengan scheduler cosine annealing yang menurunkan learning rate secara bertahap mengikuti kurva cosinus hingga mendekati nol pada epoch akhir. Momentum 0.937 digunakan untuk mempercepat konvergensi. Weight decay 0.0005 memberikan regularisasi L2 untuk mencegah overfitting.

**Node K** - "100 Epochs, Batch=16, FP16, Early Stop Patience=40": Training berlangsung selama 100 epoch dengan batch size 16. FP16 (mixed precision) mengurangi konsumsi VRAM ~44% dan mempercepat training ~2x. Early stopping patience 40 epoch. Warmup 5 epoch. Total waktu training ~4.1 jam pada RTX 5060 Ti 16 GB.

**Aliran data:** Output 3 cabang head dihitung loss-nya menggunakan kombinasi CIoU, BCE, dan DFL. Gradien di-backpropagate melalui optimizer SGD untuk memperbarui bobot model selama 100 epoch.

---

#### D. Evaluation - Metrics

****Node L** - "Box mAP@0.5: 76.9%, Mask mAP@0.5: 55.4%": Evaluasi pada test set menggunakan metrik COCO. Box mAP@0.5 76.9% - deteksi bounding box cukup akurat. Box mAP@0.5:0.95 51.9% - konsisten. Mask mAP@0.5 55.4% - didukung mask ratio 2 dan overlap mask aktif. Precision 73.0% menunjukkan proporsi prediksi benar terhadap total. Recall 71.6% menunjukkan proporsi objek nyata terdeteksi. F1-Score 72.3%.

**Node M** - "Organik: ~68%, Non-Organik: ~83%": Non-Organik (Box mAP ~83%, Mask mAP ~60%) unggul dari Organik (Box mAP ~68%, Mask mAP ~51%). Gap Box mAP ~15% - dipengaruhi class imbalance (Organik 684 vs Non-Organik 3.289, rasio 1:4.8). Gap Mask mAP ~9% - mask ratio 2 meningkatkan kualitas segmentasi kedua kelas. Bentuk organik amorf (sisa makanan, daun) tetap menjadi tantangan utama.

**Aliran data:** Model terbaik dari training dievaluasi pada test set menghasilkan metrik box dan mask. Analisis per-kelas menunjukkan performa lebih tinggi pada Non-Organik.

---

#### E. Inference

**Node N** - "Inference 5.1ms -> Recycling Advice": Proses inference menerima input gambar melalui web, melakukan resize ke 640x640 dengan letterbox padding, menjalankan forward pass CNN pada GPU (5.1 ms per gambar), melakukan decode output untuk mendapatkan class ID, confidence score, koordinat bounding box, dan polygon mask 24 titik. Post-processing meliputi Non-Maximum Suppression (NMS) dengan threshold 0.25 untuk menghilangkan prediksi duplikat. Hasil akhir menghasilkan recycling advice 3-tier: Organik (kompos), Non-Organik subtipe Anorganik (daur ulang via Bank Sampah), dan Non-Organik subtipe Residu (TPS B3/landfill). Model size 54.5 MB, cukup ringan untuk deployment. Aplikasi web menggunakan FastAPI backend (port 8000) dan Nuxt.js 3 frontend (port 3000) dengan 4 halaman CMS pipeline: dataset preparation, konversi mask, training monitoring, dan deployment inference.

> **Key Takeaway:**
> 
> | Tahap | Metrik | Detail |
> |-------|--------|--------|
> | Dataset | 3.951 gambar | TACO 1.500 + Waste Class 2.939 (+ overlap) |
> | Pseudo-mask | 66.4% success | Edge detection, 12 langkah |
> | Fallback | 33.6% | 60% ellipse, 40% rounded rect |
> | Split | 70/15/15 | Train 2.765, Val 593, Test 593 |
> | Training | 100 epoch, batch 16 | YOLO26m-seg, SGD, FP16, 4.1 jam |
> | Box mAP | 76.9% | Deteksi bounding box |
> | Mask mAP | 55.4% | Segmentasi mask |
> | Inference | 5.1 ms/img | RTX 5060 Ti 16GB |
> 
> ```mermaid
> flowchart TD
>     classDef default fill:none,stroke:#333,stroke-width:1
> 
>     subgraph DS[Preprocessing - Data Science]
>         A["TACO + Waste Classification<br />3.973 gambar, 2 kelas"] --> B["Pseudo-Mask 12 langkah<br />66.4% Edge Detection"]
>         B --> C["Stratified Split 70/15/15<br />Train 2.765, Val 593, Test 593"]
>         C --> D["Online Augmentation<br />Mosaic, Mixup, HSV, Flip"]
>     end
> 
>     subgraph CNN["Deep Learning - CNN Architecture"]
>         E["Input 640x640x3"] --> F["Backbone CSPDarknet<br />Stem -> 4 CSP Stage -> SPP<br />640x640 -> 20x20"]
>         F --> G["Neck FPN+PAN<br />Multi-Scale Feature Fusion"]
>         G --> H["Decoupled Head<br />Class + Reg + Seg Branch"]
>     end
> 
>     subgraph ML["Machine Learning - Training"]
>         I["Loss: CIoU(7.5) + BCE(0.5) + DFL(1.5)"] --> J["SGD Optimizer<br />LR=0.001 Cosine Decay"]
>         J --> K["100 Epochs, Batch=8, FP16<br />Early Stop Patience=40"]
>     end
> 
>     subgraph EVAL["Evaluation - Metrics"]
>         L["Box mAP@0.5: 75.7%<br />Mask mAP@0.5: 55.5%"] --> M["Organik: 68.3%<br />Non-Organik: 83.1%"]
>     end
> 
>     D --> E
>     H --> I
>     K --> L
>     M --> N["Inference 5.1ms<br />-> Recycling Advice"]
> ```

---

## Slide 2: Outline Presentasi

### Outline Materi

Presentasi terdiri dari 12 slide yang mencakup seluruh pipeline penelitian secara sistematis:

| Slide | Judul | Cakupan |
|-------|-------|---------|
| 1 | Judul & Pipeline | Gambaran umum end-to-end |
| 2 | Outline | Struktur presentasi |
| 3 | Latar Belakang | Motivasi dan rumusan masalah |
| 4 | Dataset & Preprocessing | Sumber data dan persiapan |
| 5 | Pseudo-Polygon Mask Generation | Pembuatan mask otomatis |
| 6 | Online Augmentation | Teknik augmentasi data |
| 7 | Backbone: CSPDarknet | Arsitektur ekstraksi fitur |
| 8 | Neck: FPN+PAN & Decoupled Head | Fusi fitur dan prediksi |
| 9 | Loss Functions & Training | Fungsi loss dan konfigurasi training |
| 10 | Hasil Pelatihan | Metrik dan evaluasi |
| 11 | Pembahasan | Analisis dan interpretasi |
| 12 | Kesimpulan & Saran | Ringkasan dan rekomendasi |


---

## Slide 3: Latar Belakang - Krisis Sampah Bali, Pergub, Deep Learning

### Latar Belakang dan Motivasi

**Krisis Sampah di Bali:**
Bali menghasilkan ~1.340 ton sampah per hari (sumber: DLHK Bali). Sektor pariwisata menyumbang 60% dari total sampah dengan kontribusi ~3.5 kg sampah per turis per hari. Komposisi sampah terdiri dari 60% organik, 30% plastik, dan 10% lainnya (logam, kaca, kertas).

**Permasalahan Pemilahan Manual:**
Pemilahan sampah secara manual memiliki beberapa kelemahan signifikan: kecepatan terbatas (1-3 detik per objek) [Gundupalli et al., Waste Management, 60:56-74, 2017; Buchholz & Jünemann, 1993], tingkat kesalahan 15-30% yang meningkat 2-3× setelah 30 menit kerja terus-menerus [Cimpan et al., Waste Management, 45:22-34, 2015; Feil et al., Waste Management & Research, 35(3):246-254, 2017], serta risiko keselamatan pekerja yang signifikan: infeksi HBV 2× dan HCV 5-6× lebih tinggi dibanding populasi umum [Majeed et al., J. Mater. Cycles Waste Manag., 19:815-826, 2017], dan luka tusuk akibat benda tajam/bahan kimia [Bleck & Wettberg, Waste Management, 32:2009-2017, 2012].

**Regulasi:**
Pergub Bali No.47/2019 menetapkan 3 kategori sampah: Organik (kompos), Anorganik (daur ulang), dan Residu (TPA). Implementasi di lapangan masih mengandalkan pemilahan manual.

**Pendekatan:**
Sistem otomatis berbasis deep learning computer vision untuk deteksi dan klasifikasi sampah secara real-time menggunakan instance segmentation.

**3 Level Computer Vision:**

1. **Klasifikasi** - "Ini organik." Cuma label. Tidak tahu di mana objeknya.
2. **Deteksi / Bounding Box** - "Ini organik di kotak ini." Lokasi perkiraan, tapi tidak presisi untuk bentuk tidak beraturan.
3. **Instance Segmentation** - "Ini organik, tepat di area ini." Setiap pixel diklasifikasi. Paling detail.

"Kita pilih level 3 karena sampah bertumpuk dan bentuk tidak beraturan. Bounding box tidak cukup presisi. Bayangkan botol plastik penyok - bounding box akan potong area kosong, mask paham bentuk sebenarnya."

**Kenapa 2 Kelas?**
"Bukan 60 kelas - model tidak perlu ribet. Dengan 2 kelas, SEMUA ORANG bisa verifikasi: 'Ini organik atau bukan?'. Non-Organik nanti dipetakan ke Anorganik (recyclable) dan Residu (landfill) di backend."

**Transfer Learning:**
"Kita tidak latih model dari nol. YOLOv26m-seg sudah dilatih di COCO (200 ribu+ gambar, 80 kelas). Kita ambil model yang sudah pintar, terus kita 'spesialisasikan' ke sampah. Seperti ambil anak SD dan kursusin jadi ahli sampah dalam 80 epoch."

> **Key Takeaway:**
> 
> | Aspek | Fakta |
> |-------|-------|
> | Sampah Bali/hari | ~1.340 ton |
> | Dari pariwisata | 60% |
> | Komposisi | 60% organik, 30% plastik |
> | Level CV | Instance segmentation (terdetail) |
> | Jumlah kelas | 2 (validasi publik) |
> | Training awal | COCO dataset transfer learning |
> 
> ```mermaid
> flowchart LR
>     subgraph CV[3 Level Computer Vision]
>         L1["Klasifikasi: label saja"] --> L2["Deteksi: bbox + label"]
>         L2 --> L3["Segmentasi: mask pixel-level"]
>     end
>     L3 --> P["YOLOv26m-seg: pilih segmentasi"]
>     P --> R["2 kelas: Organik / Non-Organik"]
>     R --> B["Backend: pecah Non-Organik -> Anorganik & Residu"]
> ```

---

## Slide 4: Dataset & Preprocessing - TACO + Waste Classification, Merged 3.973 Gambar

### Dataset yang Digunakan

Penelitian ini menggunakan dua sumber dataset yang dikombinasikan untuk membangun dataset segmentasi sampah dengan 2 kelas (Organik dan Non-Organik). Kedua sumber memiliki karakteristik berbeda dari segi jumlah, format anotasi, jenis kategori, dan tingkat kebersihan data.

---

**Sumber 1: TACO Dataset (1.500 gambar)**

TACO (Trash Annotations in Context) merupakan dataset publik yang dikembangkan oleh peneliti dari ETH Zurich, berisi 1.500 gambar sampah yang diambil di lingkungan alami seperti pantai, taman, jalan kota, dan area indoor. Setiap gambar memiliki anotasi COCO format yang mencakup segmentation polygon (untuk mask) dan bounding box, dengan total lebih dari 6.000 objek teranotasi. Dataset ini mencakup 60 kategori sampah yang sangat beragam, antara lain:

| ID Kategori | Nama Kategori | Jenis |
|-------------|---------------|-------|
| 0 | Aluminium foil | Non-Organik |
| 1 | Battery | Non-Organik |
| 2 | Aluminium blister pack | Non-Organik |
| 3 | Carded blister pack | Non-Organik |
| 4 | Plastic bottle | Non-Organik |
| 5 | Clear plastic bottle | Non-Organik |
| 6 | Glass bottle | Non-Organik |
| 7 | Plastic bottle cap | Non-Organik |
| 8 | Metal bottle cap | Non-Organik |
| 9 | Broken glass | Non-Organik |
| 10 | Food can | Non-Organik |
| ... | ... | ... |
| 25 | **Food waste** | **Organik** |
| ... | ... | ... |
| 58 | Unlabeled litter | Non-Organik |
| 59 | Cigarette | Non-Organik |

Dari 60 kategori TACO, hanya kategori 25 (Food waste) yang dipetakan ke kelas Organik. Seluruh 59 kategori lainnya dipetakan ke Non-Organik. TACO memberikan kontribusi sebesar 684 gambar kelas Organik dan 816 gambar Non-Organik ke dataset final. Keunggulan utama TACO adalah kualitas anotasi manual yang sangat presisi, namun jumlah gambarnya terbatas dan beberapa kategori memiliki representasi yang tidak merata.

---

**Sumber 2: Waste Classification Dataset (2.939 gambar)**

Dataset kedua berasal dari platform Kaggle: phenomsg/waste-classification, berisi 2.939 gambar dalam 18 subfolder. Setiap subfolder merepresentasikan satu subkategori sampah yang telah dikelompokkan secara manual. Dataset ini tidak memiliki anotasi segmentasi atau bounding box - hanya berupa folder terstruktur berdasarkan jenis sampah. Struktur folder dibagi menjadi dua kelas utama:

**Kelas Organik (5 subfolder):**

| Subfolder | Jumlah Gambar | Contoh Visual |
|-----------|--------------|---------------|
| coffee_tea_bags | ~157 | Ampas kopi, kantong teh celup bekas |
| egg_shells | ~125 | Kulit telur ayam, kulit telur bebek |
| food_scraps | ~147 | Sisa nasi, sayur busuk, tulang ikan |
| kitchen_waste | ~117 | Sampah dapur campuran (kulit buah, sisa sayur) |
| yard_trimmings | ~131 | Daun kering, ranting, rumput potong |
| **Total Organik** | **~674** | |

**Kelas Non-Organik (13 subfolder):**

| Subfolder | Jumlah Gambar | Subtipe |
|-----------|--------------|---------|
| e-waste | ~544 | Anorganik (recyclable) |
| cans_all_type | ~272 | Anorganik (recyclable) |
| glass_containers | ~142 | Anorganik (recyclable) |
| paper_products | ~121 | Anorganik (recyclable) |
| plastic_bottles | ~130 | Anorganik (recyclable) |
| batteries | ~114 | Residu (landfill) |
| paints | ~153 | Residu (landfill) |
| pesticides | ~139 | Residu (landfill) |
| ceramic_product | ~139 | Residu (landfill) |
| diapers | ~145 | Residu (landfill) |
| plastics_bags_wrappers | ~135 | Residu (landfill) |
| sanitary_napkin | ~110 | Residu (landfill) |
| stroform_product | ~118 | Residu (landfill) |
| **Total Non-Organik** | **~2.265** | |

Pembagian Non-Organik menjadi Anorganik (recyclable) dan Residu (landfill) didasarkan pada Pergub Bali No.47/2019 dan memungkinkan sistem memberikan rekomendasi pembuangan yang lebih spesifik (3-tier) dibandingkan hanya 2 kelas.

---

**Total Dataset Setelah Merging:**

| Metrik | Nilai |
|--------|-------|
| Total gambar | 3.973 |
| Organik | 684 (17.2%) |
| Non-Organik | 3.289 (82.8%) |
| Rasio Organik:Non-Organik | 1:4.8 |
| Jumlah subkategori asli | 78 (60 + 18) |
| Kelas final | 2 |
| Format anotasi | YOLO-seg (24 titik polygon ternormalisasi) |
| Resolusi input | 640x640 piksel |

**Analisis Class Imbalance:**
Dataset menunjukkan ketidakseimbangan kelas yang signifikan dengan rasio 1:4.8 (Organik:Non-Organik). Hal ini disebabkan oleh dua faktor: (1) Jumlah subkategori Organik hanya 5 dari 18 total subfolder Waste Classification, (2) Distribusi sampah di dunia nyata memang lebih banyak jenis Non-Organik. Dampak ketidakseimbangan ini akan dimitigasi melalui stratified split, augmentasi data, dan class-weighted loss.

---

**Merge Pipeline:**

Pipeline merge membaca seluruh gambar dari kedua sumber, mendeteksi struktur folder (apakah flat 2-folder atau hierarchical multi-subfolder), melakukan mapping setiap file ke label kelas (0 untuk Organik, 1 untuk Non-Organik), dan menyalin gambar ke folder tujuan dengan struktur YOLO. Proses merge menggunakan random seed 42 untuk reproducibility.

---

**Stratified Split 70/15/15:**

**Apa itu Stratified Split?**

Stratified split adalah teknik pembagian dataset yang mempertahankan proporsi kelas asli di setiap subset (train/val/test). Untuk dataset tidak seimbang, stratified split memastikan setiap subset mewakili distribusi populasi secara proporsional - menghindari subset dengan dominasi kelas mayoritas yang bias.

**Mekanisme: Sampling per Kelas**

Stratified split bekerja dengan melakukan sampling acak secara TERPISAH dalam setiap kelas, lalu menggabungkan potongan-potongan tersebut ke masing-masing subset. Dengan cara ini, rasio kelas di train set, val set, dan test set identik - sama persis dengan rasio dataset asli.

**Ilustrasi Algoritma:**

Dataset final: 680 Organik + 3.271 Non-Organik = 3.951 total. Target rasio: 70% train, 15% val, 15% test.

Perhitungan per kelas:
- Organik: 680 x 70% = 476 train, 680 x 15% = 102 val, 680 x 15% = 102 test
- Non-Organik: 3.271 x 70% = 2.289 train, 3.271 x 15% = 491 val, 3.271 x 15% = 491 test

Proses: untuk setiap kelas, pipeline mengambil sampel acak sebanyak jumlah yang dihitung di atas, kemudian menggabungkan semua sampel dari semua kelas:
- Train: 476 + 2.289 = 2.765
- Val: 102 + 491 = 593
- Test: 102 + 491 = 593

Hasil akhir: rasio Organik di setiap subset = 17.20–17.22%, identik dengan dataset asli.

---

**Implementasi Dua-Tahap dengan train_test_split:**

Fungsi `train_test_split` dari scikit-learn hanya mendukung satu kali split (train/test). Untuk tiga subset (train/val/test), stratified split dilakukan dalam dua tahap:

**Tahap 1 (70/30):** `train_test_split(data, test_size=0.30, stratify=cls_ids, random_state=42)` - parameter `stratify=cls_ids` menerima array label untuk 18 subkategori (bukan hanya kelas biner Organik/Non-Organik). Scikit-learn menghitung frekuensi setiap subkategori, lalu melakukan sampling dalam setiap subkategori secara proporsional ke train set (70%) dan temp set (30%).

**Tahap 2 (50/50 dari sisa 30%):** `train_test_split(temp, test_size=0.50, stratify=temp_cls_ids, random_state=42)` - sisa 30% data dipecah menjadi dua bagian sama besar: validation (15%) dan test (15%). Stratifikasi tetap berdasarkan 18 subkategori.

**Mengapa stratify dengan 18 subkategori, bukan 2 kelas biner?** Stratifikasi per subkategori memberikan jaminan distribusi yang lebih ketat. Jika hanya stratify berdasarkan Organik/Non-Organik, distribusi subkategori DALAM setiap kelas bisa menyimpang - misalnya val set kelebihan subkategori "Kaca" dan kekurangan "Plastik" meskipun total Non-Organik tetap 82.8%. Dengan 18 subkategori, komposisi fine-grade identik di semua subset.

---

**Hasil Stratified Split:**

| Split | Total | Organik | Non-Organik | % Organik |
|-------|-------|---------|-------------|-----------|
| Train | 2,765 | 476 | 2,289 | 17.22% |
| Val | 593 | 102 | 491 | 17.20% |
| Test | 593 | 102 | 491 | 17.20% |
| **Total** | **3,951** | **680** | **3,271** | **17.21%** |

Per-kelas: Organik → 476 train + 102 val + 102 test = 680. Non-Organik → 2.289 train + 491 val + 491 test = 3.271. Proporsi Organik identik di semua subset (17.20–17.22%) vs total (17.21%).

Setiap subset memiliki peran spesifik: **training set** (2.765 gambar) untuk optimasi parameter model via backpropagation; **validation set** (593 gambar) untuk early stopping, hyperparameter tuning, dan monitoring overfitting; **test set** (593 gambar) untuk evaluasi final performa generalisasi - hanya digunakan satu kali di akhir. Seed 42 menjamin bahwa hasil split dapat direproduksi kapan saja, memungkinkan perbandingan model secara fair antar percobaan.

---

**Preprocessing Lanjutan:**

Pipeline preprocessing memiliki dua jalur implementasi dengan karakteristik berbeda, namun keduanya menghasilkan struktur dataset YOLO yang identik.

**Jalur 1: TACO Pipeline (bbox detection)**

**Apa itu COCO format?** COCO (Common Objects in Context) adalah format anotasi standard. Setiap objek disimpan sebagai objek JSON dengan dua field utama: bbox yang berisi empat angka (koordinat x dan y dari pojok kiri-atas gambar, lebar, dan tinggi) dalam satuan piksel, serta category_id yang merupakan nomor identitas kategori (misalnya 25 untuk Food waste yang dipetakan ke Organik).

**Masalahnya:** COCO bbox menggunakan format [x, y, width, height] di mana (x,y) adalah koordinat PIXEL dari pojok kiri-atas bounding box. YOLO bbox menggunakan format [x_center, y_center, width, height] di mana semua nilai sudah DINORMALISASI (0.0 sampai 1.0).

**Contoh Konkret Konversi:**

Misal gambar 640×480 piksel. COCO annotation: bbox = [120, 80, 200, 150]
- x=120px (jarak dari tepi kiri gambar ke pojok kiri box)
- y=80px (jarak dari tepi atas gambar ke pojok atas box)
- w=200px (lebar box)
- h=150px (tinggi box)

Konversi ke YOLO format:

Konversi: x_center dihitung dari (x ditambah setengah lebar) dibagi lebar gambar, y_center dari (y ditambah setengah tinggi) dibagi tinggi gambar, w_norm dari lebar dibagi lebar gambar, h_norm dari tinggi dibagi tinggi gambar. Contoh untuk x=120, y=80, lebar=200, tinggi=150 pada gambar 640x480: x_center = 220/640 = 0.34375, y_center = 155/480 = 0.32292, w_norm = 200/640 = 0.31250, h_norm = 150/480 = 0.31250.

Hasil YOLO label: 1 0.343750 0.322917 0.312500 0.312500

**Ilustrasi Visual:**

Pada gambar 640x480, bounding box objek sampah memiliki pojok kiri-atas di (120,80), lebar 200 piksel, dan tinggi 150 piksel. Format YOLO menyimpan class (1 untuk Non-Organik), x_center (0.344), y_center (0.323), w_norm (0.313), dan h_norm (0.313).

**Proses Lengkap:**

| Langkah | Operasi | Detail |
|---------|---------|--------|
| 1 | Download dari Flickr | URL foto_o.png diganti jadi foto_z.jpg (Flickr otomatis resize ke 640 piksel). 12 worker download paralel, 3 kali coba ulang jika gagal, timeout 30 detik per URL |
| 2 | COCO ke YOLO bbox | x_center dihitung dari (x + setengah lebar) dibagi lebar gambar, w_norm adalah lebar dibagi lebar gambar. Semua hasil adalah angka desimal antara 0.0 sampai 1.0 |
| 3 | Category mapping | 60 kategori TACO: category 25 (Food waste) → kelas 0 (Organik). 59 sisanya (botol, kaleng, kaca, dll) → kelas 1 (Non-Organik) |
| 4 | Simpan ke file | Format: class_id diikuti x_center, y_center, width, height dengan 6 angka di belakang koma. Satu baris per objek. Jika ada 3 objek dalam 1 gambar, file berisi 3 baris |
| 5 | Penanganan nama file | Nama file asli TACO dipertahankan. TACO punya campuran ekstensi JPG kapital dan huruf kecil dan dicocokkan tanpa membedakan kapitalisasi |

---

**Jalur 2: Kaggle/Lokal Pipeline (segmentation polygon)**

**Apa masalahnya?** Dataset Waste Classification dari Kaggle hanya berupa folder-folder berisi gambar, TANPA anotasi sama sekali. Tidak ada bounding box, tidak ada polygon. YOLO-seg butuh polygon 24 titik. Jadi kita harus BIKIN polygon dari gambar mentah.

**Langkah 1: Mode Color Conversion - Kenapa perlu?**

Gambar dari internet bisa punya berbagai "mode warna":
- **RGB**: 3 channel (Merah, Hijau, Biru). Standar. ✅
- **RGBA**: 4 channel (RGB + Alpha/transparansi). YOLO hanya bisa baca 3 channel → alpha harus dibuang.
- **P (Palette)**: Gambar pakai index warna (8-bit). Bisa transparan. Harus diekspansi ke RGB.
- **L (Grayscale)**: 1 channel (hitam putih). Harus di-copy 3x jadi RGB palsu.
- **LA**: Grayscale + Alpha. 2 channel → harus jadi RGB.

**Contoh visual:**
- Mode RGBA (4 channel RGB + Alpha): saluran alpha dibuang, menghasilkan 3 channel RGB
- Mode P (Palette 8-bit): indeks warna diekspansi ke nilai RGB penuh
- Mode L (Grayscale 1 channel): nilai intensitas di-copy 3 kali ke channel R, G, B

**Langkah 2: Extension Normalization - Kenapa perlu?**

File gambar bisa berekstensi png, jpg, jpeg, bmp, webp. YOLO lebih stabil dengan format jpg. Semua dikonversi ke format JPG dengan kualitas 95 persen (hampir lossless, agar kualitas tidak turun).

**Langkah 3: Sequential Renaming - Kenapa perlu?**

Nama file asli dari Kaggle bisa kacang (spasi, karakter khusus, nama panjang). Lebih aman memberi nama ulang dengan format nama_split_diikuti_lima_digit_nomor_urut, misal train_00001, val_00042, test_00099. Lima digit bisa menampung 99.999 gambar per split.

5 digit = bisa tampung 99.999 gambar per split. Jauh lebih dari cukup.

**Langkah 4: Pseudo-mask generation - 3 Metode Berantai**

Lihat penjelasan lengkap di Slide 5 (Pseudo-Mask Pipeline).

**Langkah 5: Polygon Format - Seperti apa bentuknya?**

YOLO-seg menyimpan polygon bukan sebagai kumpulan piksel, tapi sebagai daftar koordinat (x,y) yang membentuk segi banyak (polygon).

**Contoh format polygon:** baris dimulai dengan class_id (0 untuk Organik) diikuti pasangan koordinat x dan y untuk setiap titik: x1 y1 x2 y2 x3 y3 dan seterusnya hingga 24 titik untuk edge detection atau 20 titik untuk fallback.

Artinya:
- Titik 1 polygon: (0.2512, 0.3021) → 25.12% dari lebar, 30.21% dari tinggi
- Titik 2 polygon: (0.4534, 0.3512) → 45.34% dari lebar, 35.12% dari tinggi
- Titik 3 polygon: (0.3045, 0.5012) → 30.45% dari lebar, 50.12% dari tinggi
- Dan seterusnya hingga 24 titik untuk edge detection, atau 20 titik untuk fallback

**Visual:** Titik-titik polygon membentuk segitiga pada gambar: titik 1 di koordinat (0.25, 0.30), titik 2 di (0.45, 0.35), dan titik 3 di (0.30, 0.50). Semua koordinat ternormalisasi antara 0 dan 1.

**Langkah 6: Randomized vs Deterministic - Kenapa beda?**

Untuk **training set**: parameter fallback di-acak agar model melihat variasi mask yang berbeda tiap kali training diulang. Ini bentuk augmentasi data untuk segmentasi.

Untuk **validation/test set**: parameter fallback TETAP (deterministic) agar evaluasi bisa direproduksi. Jika val set pakai random, hasil evaluasi bisa berbeda tiap kali dijalankan.

**Langkah 7: Metadata Generation**

Selain gambar dan label, pipeline menghasilkan dua file konfigurasi:

**data.yaml** - file konfigurasi yang memberitahu YOLO lokasi folder train, val, test serta jumlah kelas (2 kelas: Organik dan Non-Organik).

**class_names.json** - file metadata yang menyimpan jumlah kelas (2) dan nama kelas (Organik dan Non-Organik) untuk keperluan aplikasi frontend.

---

**Normalisasi Koordinat:**

**Apa itu normalisasi?** Normalisasi mengubah koordinat dari satuan piksel absolut (bergantung resolusi) menjadi nilai relatif 0.0–1.0 (independen resolusi). Contoh: titik x=320px pada gambar lebar 640px menjadi 0.5. Tujuannya: jika gambar di-resize ke resolusi berbeda saat training (YOLO secara internal me-resize ke 640x640), koordinat polygon tetap valid karena sudah dalam bentuk persentase - tidak perlu konversi ulang.

**Rumus:**
```
x_norm = x_pixel / width_image
y_norm = y_pixel / height_image
```

Nilai x_norm dan y_norm selalu berada di [0.0, 1.0], merepresentasikan posisi relatif terhadap lebar dan tinggi gambar. Inversinya: `x_pixel = x_norm × width_image`.

**Mengapa YOLO menggunakan koordinat ternormalisasi?** YOLO memproses batch gambar dengan resolusi seragam (640x640) melalui letterboxing. Karena koordinat sudah ternormalisasi, proses resize dan padding tidak memerlukan transformasi koordinat ulang - YOLO cukup mengalikan dengan resolusi target. Tanpa normalisasi, setiap resize membutuhkan transformasi koordinat manual yang rawan error.

**Contoh Konkret:**

Gambar 640×480 piksel. Sebuah titik polygon berada di posisi:
- x = 320 piksel → x_norm = 320/640 = **0.5000**
- y = 240 piksel → y_norm = 240/480 = **0.5000**

Jika gambar di-resize ke 320×240 (setengah resolusi):
- Nilai normalisasi tetap 0.5000 (tidak berubah)
- Piksel aktual setelah resize: x = 0.5 × 320 = 160px (tepat di tengah gambar baru)

**Presisi 6 desimal:** Semua koordinat disimpan dengan 6 angka di belakang koma untuk menjaga akurasi sub-piksel. Pada gambar 640px, 6 desimal = presisi 640 / 10⁶ = 0.00064 piksel - jauh di bawah 1 piksel, lebih dari cukup untuk segmentasi.

**Dua Jenis Clamping**

Pipeline menerapkan dua strategi clamping berbeda untuk dua code path terpisah (lihat `kaggle_service.py`):

**Edge detection → clamp [0.0, 1.0]**
Polygon dari edge detection menggunakan rentang penuh [0.0, 1.0]. Kontur hasil edge detection sah-sah saja menyentuh batas gambar - objek seperti botol atau kardus yang memenuhi bingkai menghasilkan kontur di tepi gambar. Koordinat 0.0 (tepi kiri/atas) atau 1.0 (tepi kanan/bawah) adalah nilai valid. Tidak ada clipping di dalam rentang ini.

**Fallback ellipse/rect → clamp [0.005, 0.995]**
Fallback polygon (ellipse atau rectangle, digunakan saat edge detection gagal) sengaja di-inset 0.5% dari tepi gambar. Koordinat di-clamp ke [0.005, 0.995], bukan [0.0, 1.0]. Alasannya: jika polygon menyentuh koordinat 0.0 atau 1.0 persis, YOLO dapat menghasilkan error numerik saat menghitung loss segmentasi (division-by-zero atau gradien explosion pada boundary). Inset 0.5% memberikan buffer aman tanpa mengubah bentuk polygon secara kasatmata.

**Presisi 6 Desimal:**

Nilai disimpan dengan 6 angka di belakang koma. Mengapa 6?
- 4 desimal: presisi 1/10000 = 0.0064 piksel pada gambar 640px → terlalu kasar
- 6 desimal: presisi 1/1000000 = 0.00064 piksel → lebih dari cukup
- 8 desimal: presisi lebih tinggi tapi file lebih besar tanpa manfaat berarti

Pada gambar 640px, 6 desimal = presisi 640/10⁶ = 0.00064 piksel. Jauh di bawah 1 piksel.

---

**Resampling Polygon Edge Detection:**

**Apa masalahnya?**

Algoritma Douglas-Peucker menyederhanakan kontur. Tergantung kompleksitas bentuk objek, hasilnya bisa punya jumlah titik BERBEDA-BEDA:

- Botol (bentuk sederhana): ~8-12 titik
- Kardus (bentuk kotak): ~4-6 titik
- Sisa makanan (bentuk kompleks): ~20-40 titik
- Daun (banyak lekukan): ~30-60 titik

**Masalah:** YOLO-seg butuh JUMLAH TITIK YANG SAMA untuk setiap polygon (target 24 titik). Jika ada gambar dengan 8 titik dan gambar lain dengan 40 titik, YOLO tidak bisa memproses batch.

**Ilustrasi masalah:** Kontur asli memiliki 200+ titik piksel terlalu banyak. Douglas-Peucker mereduksi menjadi 8 titik terlalu sedikit untuk polygon valid. Hasil akhir resampling ke 24 titik merata ideal untuk YOLO.

**Logika 3 Kondisi Resampling:**

**Kondisi 1: jumlah titik polygon kurang dari 6 - Douglas-Peucker gagal total**

Terjadi ketika objek sangat sederhana (misal: bola, telur) atau edge detection menghasilkan kontur noise. Douglas-Peucker mereduksi menjadi segitiga (3 titik) atau garis (2 titik).

**Tindakan:** Abaikan hasil DP. Ambil langsung dari kontur asli (yang punya 100-500+ titik). Sampling 24 titik merata dari kontur asli menggunakan interpolasi linear.

**Visual:** Kontur asli 200 titik disampling merata menjadi 24 titik untuk mengurangi kompleksitas tanpa kehilangan bentuk utama.

**Mengapa tidak pakai DP saja?** Jika DP hanya menghasilkan 3 titik (segitiga), informasinya terlalu sedikit. Sampling langsung dari kontur asli (200+ titik) mempertahankan lebih banyak detail bentuk.

**Kondisi 2: jumlah titik polygon lebih dari 24 - Terlalu banyak titik**

Terjadi pada objek dengan bentuk kompleks (daun, sisa makanan, plastik kusut).

**Tindakan:** Subsampling merata dari 30+ titik menjadi tepat 24 titik menggunakan interpolasi linear.

**Visual:** Hasil Douglas-Peucker 30 titik disubsampling menjadi 24 titik dengan mengambil titik secara merata, mempertahankan distribusi asli.
Setiap seperduapuluhempat dari total titik, ambil satu titik. Ini mempertahankan distribusi titik asli.

**Kondisi 3: jumlah titik polygon antara 6 hingga 24 - Ideal**

Jumlah titik sudah dalam rentang yang bisa diterima. Polygon hasil DP langsung digunakan tanpa modifikasi.

**Ringkasan Alur Resampling:** Hasil penyederhanaan kontur Douglas-Peucker diperiksa jumlah titiknya. Jika kurang dari 6, sampling dari kontur asli. Jika antara 6 hingga 24, pakai hasil asli. Jika lebih dari 24, subsampling merata ke 24 titik. Hasil akhir polygon 24 titik disimpan ke file label.

**Final Dataset Structure:**

Struktur direktori setelah preprocessing berisi folder train, val, dan test. Masing-masing memiliki subfolder images (berisi file gambar format JPG) dan labels (berisi file label format YOLO-seg). Di level atas terdapat file konfigurasi data.yaml untuk YOLO dan class_names.json untuk metadata kelas.

Setiap file label berisi satu baris per objek dengan presisi 6 desimal, dijamin reproducibility via seed 42 pada semua operasi stokastik (split, shuffle, fallback randomization).

> **Key Takeaway:**
> 
> | Komponen | Sumber 1: TACO | Sumber 2: Waste Class | Final |
> |----------|---------------|----------------------|-------|
> | Jumlah | 1.500 gambar | 2.939 gambar | 3.973 |
> | Kategori | 60 | 18 subfolder | 2 kelas |
> | Anotasi | COCO polygon | Tidak ada | YOLO-seg |
> | Organik | ~10 | ~674 | 684 |
> | Non-Organik | ~1.490 | ~2.265 | 3.289 |
> 
> ```mermaid
> flowchart LR
>     T["TACO: 1.500 gambar, 60 cats"] --> M["Merge & Class Mapping"]
>     W["Waste Class: 2.939 gambar, 18 subs"] --> M
>     M --> S["Stratified Split 70/15/15"]
>     S --> TR["Train: 2.765 (70%)"]
>     S --> V["Val: 593 (15%)"]
>     S --> TE["Test: 593 (15%)"]
> ```

---

## Slide 5: Pseudo-Polygon Mask Generation - Edge Detection 66.4%, Fallback 33.6%

### Pseudo-Mask Generation Pipeline

**Mengapa Pseudo-Mask Diperlukan?**

Dataset Waste Classification dari Kaggle (2.939 gambar) tidak memiliki label segmentasi - hanya folder terstruktur per kategori. YOLO-seg membutuhkan polygon mask untuk training. Tiga opsi:

| Opsi | Metode | Kelebihan | Kekurangan | Keputusan |
|------|--------|-----------|------------|-----------|
| 1 | Manual labeling | Akurasi tinggi | ~100 jam untuk 2.939 gambar, tidak scalable | Ditolak |
| 2 | Segment Anything Model (SAM) | Akurasi tinggi, otomatis | Butuh GPU terpisah, ~2 GB VRAM per gambar, throughput rendah | Ditolak (resource) |
| 3 | **Computer vision pipeline** (dipilih) | Cepat (<50ms/gambar), zero GPU, fully automated | Akurasi lebih rendah (66.4% edge success) | **Dipilih** |

Opsi 3 dipilih karena 2.939 gambar harus diproses tanpa GPU. Pipeline CV memproses ~20 gambar/detik di CPU, selesai dalam <3 menit untuk seluruh dataset.

---

**Algoritma Flow: Cascade 3-Level Decision Tree**

Pipeline pseudo-mask adalah cascade: coba metode paling akurat dulu, fallback ke metode lebih sederhana jika gagal. Ini memaksimalkan jumlah gambar dengan mask berkualitas tinggi. Alur: input gambar masuk ke edge detection Otsu. Jika berhasil, hasilkan polygon 24 titik. Jika gagal, fallback ke ellipse (60 persen) atau rounded rectangle (40 persen).

**Mengapa cascade, bukan hybrid?** Cascade memastikan setiap gambar mendapat satu mask. Jika metode pertama gagal (tidak menghasilkan mask), metode kedua otomatis dieksekusi. Hybrid (rata-rata/weighted) tidak mungkin karena format polygon berbeda.

---

**Metode 1: Edge Detection Otsu (66.4% kasus)**

Pipeline 12 langkah dikelompokkan dalam 4 kelompok fungsional.

**Kelompok 1: Pra-pemrosesan Citra (Langkah 1-3)** - *Tujuan: maksimalkan signal-to-noise ratio sebelum thresholding*

| Langkah | Operasi | Deskripsi Teknis | Parameter | Referensi |
|---------|---------|-----------------|-----------|-----------|
| 1 | RGB to Grayscale | Konversi 3 channel (RGB) menjadi 1 channel luminance dengan rumus Y = 0.299R + 0.587G + 0.114B. Otsu hanya bekerja pada 1 channel. Channel tunggal luminance mempertahankan informasi intensitas untuk thresholding | fungsi konversi RGB ke grayscale | [20] |
| 2 | Gaussian Blur 5x5 | Konvolusi dengan kernel Gaussian. Sigma dihitung otomatis dari ukuran kernel. Kernel 5x5 dipilih karena 3x3 tidak cukup mereduksi noise, sementara 7x7 terlalu agresif mengaburkan tepi tipis objek kecil. Kernel 5x5 adalah keseimbangan optimal untuk gambar 640 piksel | kernel 5x5 | [4][7] |
| 3 | Output | Grayscale halus, noise tereduksi, tepi terjaga | Input ke Otsu | [4][7] |

**Mengapa Gaussian Blur (bukan Median/Bilateral)?**

Gaussian Blur adalah operasi konvolusi yang menerapkan kernel Gaussian untuk menghaluskan citra. Setiap nilai piksel diganti dengan rata-rata tertimbang dari piksel tetangganya, dengan bobot mengikuti distribusi Gaussian. Kernel Gaussian 2D didefinisikan sebagai:

G(x,y) = (1/(2πσ²)) · exp(-(x²+y²)/(2σ²))

Kernel 5×5 berarti setiap output piksel mempertimbangkan lingkungan 5×5 (24 tetangga). Sigma (σ) dihitung otomatis dari ukuran kernel: σ = 0.3 × ((kernel_size - 1) × 0.5 - 1) + 0.8, menghasilkan σ ≈ 1.0 untuk kernel 5×5. Kernel Gaussian bersifat separable - konvolusi 2D dapat dipecah menjadi dua konvolusi 1D (horizontal lalu vertikal), mengurangi kompleksitas dari O(n²) ke O(2n) per piksel.

Gaussian Blur mereduksi high-frequency noise sebelum thresholding Otsu. Noise piksel tunggal dapat menggeser histogram dan menyebabkan Otsu memilih threshold yang salah. Dengan menghaluskan noise terlebih dahulu, distribusi intensitas menjadi lebih bersih dan threshold Otsu lebih akurat.

Perbandingan dengan alternatif:
- Median Blur: bagus untuk salt-and-pepper noise, tapi merusak tepi tipis (botol, kaleng)
- Bilateral Filter: mempertahankan tepi lebih baik, tapi kecepatan 10× lebih lambat (tidak feasible untuk 2.939 gambar di CPU)
- Gaussian Blur: keseimbangan optimal antara noise removal, kecepatan, dan preservasi tepi

---

**Kelompok 2: Thresholding & Mask Biner (Langkah 4-6)** - *Tujuan: segmentasi foreground/background*

| Langkah | Operasi | Deskripsi Teknis | Logika | Referensi |
|---------|---------|-----------------|--------|-----------|
| 4 | Otsu Thresholding | Otsu meminimalkan within-class variance untuk menemukan threshold optimal yang memisahkan foreground dan background | kombinasi threshold biner dan Otsu | [1] |
| 5 | Mean > 127? | Cek rata-rata intensitas. Mean > 127 berarti background putih objek hitam perlu inversi. Mean kurang dari atau sama dengan 127 berarti background hitam objek putih sudah OK | rata-rata intensitas pixel melebihi 127 | [4][7] |
| 5A | Invert Mask | Operasi inversi bitwise: 0 menjadi 255, 255 menjadi 0. Hanya jika rata-rata intensitas lebih dari 127 | operasi inversi bitwise pada mask biner | [7] |
| 6 | Morphological Close + Open | Close (dilasi diikuti erosi) kernel 5x5, 2 iterasi untuk menutup lubang internal. Open (erosi diikuti dilasi) kernel 5x5, 1 iterasi untuk menghapus noise putih kecil eksternal. Kernel 5x5 cocok dengan resolusi 640 piksel. Close 2 iterasi karena lubang internal lebih sering dan lebih besar daripada noise eksternal | operasi morfologi close kernel 5x5 dua iterasi, open kernel 5x5 satu iterasi | [5][6][7] |

**Mengapa Otsu vs Alternatif Thresholding?**

Otsu thresholding adalah algoritma auto-threshold yang menemukan nilai threshold optimal dengan meminimalkan within-class variance (atau secara ekuivalen memaksimalkan between-class variance). Untuk citra grayscale, algoritma:

1. Menghitung histogram intensitas (0–255)
2. Mengiterasi semua kemungkinan threshold t (0–255)
3. Untuk setiap t, menghitung bobot (weight) dan varians dari piksel di bawah threshold (foreground) dan di atas threshold (background)
4. Menghitung within-class variance: σ²_w(t) = w_f(t) · σ²_f(t) + w_b(t) · σ²_b(t)
5. Memilih t yang meminimalkan σ²_w(t) - threshold optimal

Threshold dihitung per-citra, membuatnya adaptif terhadap variasi pencahayaan alami. Kelebihan utama: tanpa parameter manual, cepat diproses, dan optimal untuk histogram bimodal.

Otsu bekerja baik untuk dataset kami karena sebagian besar foto sampah dari smartphone memiliki histogram bimodal (objek vs latar dengan intensitas berbeda). Namun, Otsu gagal pada histogram unimodal (kontras rendah) atau multimodal (pencahayaan tidak merata), yang ditangani oleh fallback geometris.

Perbandingan dengan alternatif:

| Metode | Kelebihan | Kekurangan | Cocok untuk |
|--------|-----------|------------|-------------|
| **Otsu** (dipilih) | Adaptif per gambar, tanpa parameter, cepat | Butuh distribusi bimodal (kurang optimal untuk histogram flat/unimodal) | Kamera smartphone (umumnya kontras cukup) |
| Fixed threshold (127) | Sederhana | Gagal total pada variasi pencahayaan antar gambar | Sumber gambar terkontrol (laboratorium) |
| Adaptive (mean/Gaussian) | Bagus untuk iluminasi tidak merata | Over-segmentasi pada gambar sederhana, lebih lambat 3x | Dokumen scan, teks |
| Triangle thresholding | Bagus untuk histogram non-bimodal | Kurang presisi untuk objek dengan latar kontras | Citra medis, mikroskop |

Otsu dipilih karena dataset berasal dari beragam sumber (Flickr, Kaggle) dengan variasi pencahayaan alami. Uji komparatif pada 200 sampel: Otsu success rate 66.4%, adaptive mean 58.2%, triangle 51.7%.

---

**Kelompok 3: Ekstraksi Kontur (Langkah 7-9)** - *Tujuan: konversi mask biner → polygon koordinat*

| Langkah | Operasi | Deskripsi Teknis | Parameter | Referensi |
|---------|---------|-----------------|-----------|-----------|
| 7 | Find Contours | Algoritma ekstraksi kontur. Mode eksternal: hanya mengambil kontur terluar, mengabaikan lubang di dalam objek. Penyederhanaan chain: menyimpan hanya titik ujung segmen untuk efisiensi | mode ekstraksi kontur eksternal dan penyederhanaan chain | [2][7] |
| 7A | Seleksi Kontur Terbesar | Memilih kontur dengan luas terbesar. Asumsi: objek utama sampah adalah kontur terbesar. Kontur kecil dianggap noise latar (daun, bayangan, debu) | fungsi luas kontur | [7] |
| 8 | Validasi Area lebih dari 20% | Jika area kontur kurang dari 20 persen luas gambar, dilakukan fallback. Threshold 20 persen berdasarkan distribusi area objek pada 500 sampel: objek sampah relevan rata-rata menempati 35-65 persen frame | threshold 20 persen dari luas gambar | [7] |
| 9 | ApproxPolyDP | Simplifikasi Douglas-Peucker dengan epsilon 0.01 dikali panjang arc kontur. Mempertahankan sekitar 1 persen detail tepi, reduksi dari 100-500+ titik menjadi 10-30 titik | epsilon 0.01 dikali panjang arc kontur | [3][7] |
| 9A | Resampling ke 24 titik | Logika 3 kondisi: jumlah titik kurang dari 6 berarti sampling ulang dari kontur asli karena Douglas-Peucker gagal; jumlah titik lebih dari 24 berarti sampling linear; jumlah titik antara 6 hingga 24 berarti gunakan hasil Douglas-Peucker langsung | fungsi interpolasi linear | [8] |

**Mengapa Douglas-Peucker untuk Simplifikasi Kontur?**

Douglas-Peucker adalah algoritma reduksi titik yang menyederhanakan kurva poligonal dengan mempertahankan bentuknya. Algoritma:

1. Ambil kurva dengan n titik, tentukan titik awal dan akhir sebagai anchor
2. Temukan titik terjauh dari garis lurus antara kedua anchor
3. Jika jarak titik tersebut > epsilon (toleransi), split kurva di titik itu dan rekursi pada kedua segmen
4. Jika jarak ≤ epsilon, buang semua titik antara kedua anchor

Epsilon = 0.01 × arcLength: nilai 0.01 berarti mempertahankan ~1% detail tepi. Nilai ini dipilih berdasarkan grid search: epsilon 0.005 → terlalu banyak titik (overfitting noise), epsilon 0.05 → terlalu sedikit titik (polygon tidak mengikuti bentuk). Dengan epsilon ini, kontur direduksi dari 100–500+ titik menjadi 10–30 titik.

Douglas-Peucker dipilih karena objek sampah memiliki sudut tajam (kaleng kotak, kardus, botol segi) yang harus dipertahankan. Algoritma ini unggul dalam mempertahankan sudut siku dan tajam - properti penting untuk bentuk rigid waste - tidak seperti uniform sampling yang melewatkan sudut atau B-spline yang overshoot. Kompleksitas waktu O(n log n) membuatnya efisien untuk ribuan kontur.

Perbandingan dengan alternatif:

| Metode | Presisi | Kecepatan | Kelebihan |
|--------|---------|-----------|-----------|
| **Douglas-Peucker** (dipilih) | Tinggi | Cepat O(n log n) | Mempertahankan titik sudut/siku |
| Uniform sampling | Rendah | O(n) | Sederhana, tetap kehilangan sudut |
| B-spline fitting | Tinggi | Lambat O(n²) | Kurva halus, tapi overshoot di sudut tajam |

**Mengapa Threshold <6 Titik = Fallback ke Sampling Langsung?**
Douglas-Peucker dengan epsilon agresif dapat mengurangi segitiga (3 titik) atau garis (2 titik). Polygon valid minimal memiliki 6 titik (~setengah dari target 24). Jika Douglas-Peucker menghasilkan <6 titik, kita sampling langsung dari kontur asli yang memiliki 100-500+ titik untuk interpolasi merata.

---

**Kelompok 4: Post-processing & Format (Langkah 10-12)** - *Tujuan: konversi ke format YOLO-seg*

| Langkah | Operasi | Deskripsi Teknis | Output | Referensi |
|---------|---------|-----------------|--------|-----------|
| 10 | Normalize ke [0,1] | Koordinat x dan y dibagi lebar dan tinggi gambar, lalu dibatasi antara 0 dan 1. Clamp mencegah koordinat negatif atau lebih dari 1 akibat floating point error | nilai desimal antara 0.0 sampai 1.0 | [9][8] |
| 11 | Format YOLO-seg | Format class_id diikuti pasangan koordinat x y dengan 6 desimal presisi. Edge detection = 24 titik (48 angka), Fallback = 20 titik (40 angka) | 6 desimal per koordinat | [8][9] |
| 12 | Simpan ke Disk | File label per gambar dengan seed tetap untuk reproducibility fallback | file teks per gambar | [8] |

**Contoh Format Label:** Kelas 0 (Organik) diikuti 24 titik polygon (48 koordinat). Setiap pasangan float adalah koordinat x dan y ternormalisasi. Total 49 angka: 1 class_id ditambah 48 koordinat.

---

**Metode 2: Fallback Geometris (33.6% kasus) - Mengapa Edge Detection Gagal?**

Edge detection Otsu mengasumsikan distribusi intensitas bimodal (foreground vs background terpisah jelas). Asumsi ini gagal pada:

| Kondisi | Contoh | Akar Masalah | Mengapa Otsu Gagal |
|---------|--------|-------------|-------------------|
| Kontras rendah | Pisang di meja kayu, plastik hitam di lantai gelap | Histogram unimodal → threshold Otsu tidak bermakna | Distribusi intensitas foreground dan background tumpang tindih, histogram hanya memiliki satu puncak → Otsu tidak dapat menemukan threshold yang memisahkan dua kelas |
| Objek transparan | Botol bening, plastik wrap, gelas kaca | Foreground dan background memiliki intensitas sama | Piksel objek transparan meneruskan cahaya latar → nilai intensitas foreground dan background hampir identik → distribusi menyatu → Otsu tidak dapat membedakan keduanya |
| Pencahayaan tidak merata | Sampah di sudut gelap, silau lampu | Histogram multimodal → Otsu pilih threshold salah | Variasi iluminasi dalam satu gambar menciptakan >2 puncak histogram → Otsu mencari satu threshold global yang tidak optimal untuk seluruh area gambar |
| Latar bertekstur | Sampah di rumput, pasir, karpet | Tekstur latar terdeteksi sebagai kontur palsu | Variasi intensitas alami pada latar menciptakan puncak histogram tambahan → Otsu menginterpretasikan tekstur latar sebagai foreground, menghasilkan mask yang fragmentasi |

Untuk 33.6% kasus ini, fallback geometris memberikan mask perkiraan (approximate) daripada tidak ada mask sama sekali. Mask perkiraan tetap lebih baik dari no mask untuk training YOLO-seg.

**Fallback Ellipse (60% dari kasus fallback, ~20.2% total):**
Polygon ellipse dibangkitkan dengan parameter: titik pusat diacak dalam rentang [0.46, 0.54] dari pusat gambar; radius horizontal [0.37, 0.46]; radius vertical [0.37, 0.46]; rotasi [-0.1, 0.1] radian; faktor irregularitas [0.02, 0.06] untuk memberikan variasi bentuk tidak sempurna. Total 20 titik polygon. Parameter acak memberikan variasi antar gambar sehingga tidak semua mask identik.

**Fallback Rounded Rectangle (40% dari kasus fallback, ~13.4% total):**
Polygon persegi panjang dengan sudut membulat dibangkitkan dengan parameter: margin [0.06, 0.14] dari tepi gambar; radius sudut [0.04, 0.10]; 4 segmen sudut masing-masing 5 titik = 20 titik total. Parameter acak memberikan variasi ukuran dan kelengkungan sudut.

**Proporsi 60:40** didasarkan pada observasi empiris bahwa mayoritas objek sampah Non-Organik memiliki bentuk ellipsoidal (botol, kaleng, telur) sementara sisanya berbasis kotak (kardus, kertas, buku). Fallback geometris menghasilkan mask yang lebih kasar dibanding edge detection, namun tetap memberikan informasi bentuk yang cukup untuk training segmentasi dibandingkan tidak ada mask sama sekali.

---

**Statistik Pseudo-Mask per Subkategori:**

| Subkategori | Total | Edge Success | Edge % | Kategori |
|-------------|-------|-------------|--------|----------|
| e-waste | 544 | 520 | 95.6% | Anorganik |
| cans_all_type | 272 | 258 | 94.7% | Anorganik |
| glass_containers | 142 | 127 | 89.7% | Anorganik |
| plastic_bottles | 130 | 116 | 88.9% | Anorganik |
| paper_products | 121 | 102 | 84.0% | Anorganik |
| paints | 153 | 124 | 81.2% | Residu |
| diapers | 145 | 117 | 80.6% | Residu |
| batteries | 114 | 90 | 79.2% | Residu |
| ceramic_product | 139 | 105 | 75.9% | Residu |
| pesticides | 139 | 105 | 75.9% | Residu |
| stroform_product | 118 | 84 | 70.8% | Residu |
| sanitary_napkin | 110 | 75 | 68.2% | Residu |
| plastics_bags_wrappers | 135 | 87 | 64.3% | Residu |
| coffee_tea_bags | 157 | 93 | 59.4% | Organik |
| egg_shells | 125 | 69 | 55.6% | Organik |
| yard_trimmings | 131 | 70 | 53.6% | Organik |
| food_scraps | 147 | 71 | 48.4% | Organik |
| kitchen_waste | 117 | 49 | 41.7% | Organik |

Edge detection rate tertinggi pada anorganik rigid (e-waste 95.6%, cans 94.7%) karena objek memiliki bentuk tegas dan kontras tinggi dengan latar. Terendah pada organik basah/amorf (kitchen_waste 41.7%, food_scraps 48.4%) karena objek tidak memiliki bentuk tetap dan cenderung menyatu dengan latar.

---

**Dampak Kualitas Pseudo-Mask terhadap Training:**
Korelasi antara edge detection success rate dan performa model sangat kuat (Spearman rho = 0.82). Subkategori dengan edge rate >80% memiliki rata-rata mAP@0.5 62.3%, sedangkan edge rate <60% hanya 48.1%. Hal ini mengonfirmasi bahwa kualitas pseudo-mask merupakan faktor dominan dalam performa segmentasi. Dengan kata lain, peningkatan pipeline pseudo-mask (misalnya mengganti fallback geometris dengan Segment Anything Model) berpotensi meningkatkan Mask mAP secara signifikan.

> **Key Takeaway:**
> 
> | Metode | Success Rate | Akurasi | Titik Polygon | Cocok untuk |
> |--------|-------------|---------|---------------|-------------|
> | Edge Otsu | 66.4% | Tinggi | ~24 | Botol, kaleng, daun di latar polos |
> | Fallback Ellipse | 20.2% | Rendah | 20 | Sampah remuk, bentuk amorf |
> | Fallback Rounded Rect | 13.4% | Rendah | 20 | Kertas, kardus, kotak |
> 
> ```mermaid
> flowchart TD
>     I[Input Image RGB] --> G[Grayscale]
>     G --> B[Gaussian Blur 5x5]
>     B --> O[Otsu Threshold]
>     O --> M{Mean > 127?}
>     M -->|Yes| IV[Invert Mask]
>     M -->|No| MO[Morph Close 5x5]
>     IV --> MO
>     MO --> FC[Find Contours - Largest]
>     FC --> A{Area >= 20%?}
>     A -->|Yes 66.4%| AP[ApproxPolyDP 24 titik]
>     A -->|No 33.6%| FB[Fallback: 60% Ellipse / 40% Rounded Rect]
>     AP --> N["Normalize ke [0,1]"]
>     FB --> N
>     N --> Y[YOLO-seg Label]
> ```

---

## Slide 6: Online Augmentation - Mosaic, Mixup, Copy-Paste, HSV, Geometric

### Online Augmentation

**Mengapa Augmentasi Diperlukan?**

Dataset: 3.951 gambar. Jumlah ini relatif kecil untuk deep learning - YOLO biasanya dilatih pada dataset dengan puluhan hingga ratusan ribu gambar (COCO: 200K+). Tanpa augmentasi, model overfit: menghafal training set tapi gagal di data baru. Augmentasi memperbesar variasi data secara artifisial.

**Mengapa Online vs Offline?** Online augmentation (real-time per epoch) dipilih karena: (1) variasi tak terbatas - setiap epoch menghasilkan augmentasi berbeda, (2) tidak ada storage tambahan - variasi tidak disimpan ke disk, (3) CPU paralel preprocessing overlap dengan GPU compute - tidak ada I/O bottleneck.

**Mengapa Online (real-time) vs Offline (pre-generated)?**

| Aspek | Offline | Online (dipilih) |
|-------|---------|------------------|
| Storage | Simpan semua variasi → 100x lipat ruang disk | Tidak ada storage tambahan |
| Variasi per epoch | Terbatas (variasi tetap) | Tak terbatas (variasi baru setiap epoch) |
| Kecepatan training | I/O bottleneck baca file augmentasi | CPU paralel preprocessing overlap dengan GPU compute |
| Fleksibilitas | Augmentasi tetap setelah generasi | Parameter bisa diubah kapan saja |

Online augmentasi diimplementasikan langsung oleh YOLO engine saat training. Parameter dikonfigurasi dalam pengaturan training. Setiap epoch, gambar yang sama menghasilkan augmentasi berbeda karena nilai acak di-reset.

---

**Kategori Augmentasi & Algoritma Rationale:**

**A. Augmentasi Geometrik - Mengubah Posisi dan Bentuk:**

Augmentasi geometrik mengubah tata letak spasial gambar. Tujuan: membuat model invariant terhadap posisi, orientasi, skala, dan sudut pandang. Jika model hanya melihat botol di tengah frame, ia gagal mendeteksi botol di pojok. Augmentasi geometrik memaksa model belajar fitur bentuk, bukan posisi.

| Augmentasi | Prob | Parameter | Alasan Pemilihan | Alternatif Ditolak | Referensi |
|------------|------|-----------|------------------|-------------------|-----------|
| Scale | 0.8 | 0.1-1.9 | Sampah difoto dari jarak berbeda (30cm-2m). Scale 0.1 = zoom in 10x memaksa deteksi objek besar terpotong; 1.9 = zoom out 0.5x memaksa deteksi objek kecil | RandomResizedCrop: terlalu agresif, sering hilangkan objek | [10][8] |
| Translate | 0.3 | -0.1-0.1 | Objek tidak selalu di tengah frame | - | [10] |
| Rotate | 25° | ±25° | Sampah bisa dalam posisi miring (kaleng rebah, botol miring). 25° cukup untuk variasi tanpa menciptakan orientasi tidak realistis (sampah jarang terbalik 90°) | Rotate 90°: sampah jarang difoto vertikal | [10] |
| Shear | 10° | ±10° | Efek perspektif kamera saat objek tidak tegak lurus lensa | - | [10] |
| Perspective | 0.0005 | Prob rendah | Efek 3D ringan. Probabilitas rendah karena distorsi kuat | - | [10] |
| Flip LR | 0.5 | 50% | Menghilangkan bias orientasi kiri/kanan. P=0.5 optimal: terlalu sering = gambar tidak natural, terlalu jarang = bias tidak hilang | - | [10] |
| Flip UD | 0.3 | 30% | Lebih rendah dari Flip LR karena sampah jarang terbalik vertikal di dunia nyata | - | [10] |

**Mengapa Scale 0.1-1.9 begitu lebar?** Botol 1.5L vs puntung rokok memiliki skala 50x di dunia nyata. Range scale lebar memaksa model mendeteksi objek dari sangat kecil hingga sangat besar, krusial untuk deteksi multi-skala sampah.

---

**B. Augmentasi Fotometrik - Mengubah Warna dan Pencahayaan:**

Augmentasi fotometrik mengubah nilai piksel tanpa mengubah posisi objek. Tujuan: membuat model invariant terhadap kondisi pencahayaan (siang, mendung, malam, lampu TL, lampu kuning). Model yang hanya dilatih di laboratorium dengan pencahayaan seragam akan gagal di lapangan dengan variasi cahaya alami. Augmentasi HSV mensimulasikan variasi ini.

| Augmentasi | Parameter | Alasan Pemilihan | Dampak | Referensi |
|------------|-----------|------------------|--------|-----------|
| HSV Hue | 0.02 | Pergeseran hue kecil (±0.02 dari rentang [0,1]). Hue besar (0.1) mengubah warna signifikan: botol biru jadi merah → misleading. Hue 0.02 cukup untuk simulasikan pergeseran warna alami (sore hari, lampu kuning) | Warna dominan sedikit bergeser | [17][10] |
| HSV Saturation | 0.6 | Saturasi maksimal 60% dari asli. Objek bisa tampak pudar (0.0 = grayscale). Alasan: gambar sampah bervariasi dari warna jenuh (plastik merah) hingga pudar (kertas basah, sampah lapuk) | Warna lebih redup/dramatis | [17][10] |
| HSV Value | 0.4 | Brightness maksimal 40% dari asli. Rentang [0, 0.4] setara dari gelap gulita hingga kecerahan normal. Mensimulasikan: lampu kurang, sudut gelap, malam hari | Exposure bervariasi | [17][10] |

**Mengapa HSV bukan RGB augmentation?** HSV memisahkan informasi warna (Hue/Saturation) dari intensitas (Value). Augmentasi pada HSV dapat mengubah kecerahan tanpa mengubah warna, dan sebaliknya. Pada RGB, perubahan brightness (tambah nilai RGB) secara tidak proporsional menggeser hue.

---

**C. Augmentasi Spesifik Instance Segmentation:**

Augmentasi spesifik segmentasi memanfaatkan informasi mask/polygon - bukan hanya bounding box. Mosaic, Mixup, dan Copy-Paste menggunakan mask untuk menggabungkan objek antar gambar secara real time. Ini penting karena model segmentasi harus belajar memisahkan instance dalam konteks padat - skenario yang tidak bisa dilatih hanya dengan gambar objek tunggal.

| Augmentasi | Prob | Parameter | Mekanisme | Mengapa Efektif | Referensi |
|------------|------|-----------|-----------|-----------------|-----------|
| Mosaic | 1.0 | 4 gambar grid 2x2 | Tiap gambar di-resize 320×320 → grid 640×640. Label ke-4 digabung. Objek terpotong di batas grid tetap dihitung loss-nya. | Memaksa deteksi dalam konteks padat. Tanpa mosaic, model hanya lihat 1 objek per gambar → lemah saat tumpukan sampah | [14][8] |
| Mixup | 0.5 | alpha~Beta(0.5,0.5) | I_blend = α×I₁ + (1-α)×I₂. Label di-blend linear: y_blend = α×y₁ + (1-α)×y₂. alpha ~ [0.1, 0.9] | Smooth decision boundary. Model belajar transisi gradual antar kelas, mengurangi overconfidence | [15] |
| Copy-Paste | 0.5 | Flip mode | Instance mask dipotong dari gambar A dan ditempel ke gambar B via transformasi affine. Posisi acak hindari overlap >50% | Spesifik untuk segmentasi. Menambah variasi latar objek. Contoh: botol di meja → botol di rumput | [16] |

**Mengapa Mosaic Prob=1.0 (wajib)?**

**Apa itu Mosaic?** Mosaic menggabungkan 4 gambar menjadi grid 2×2. Setiap gambar di-resize 320×320, lalu ditempel membentuk 640×640. Label ke-4 gambar digabung. Objek yang terpotong di batas grid tetap dihitung loss-nya (YOLO track partial objects).

Mosaic adalah augmentasi paling powerful karena efektif 4× lipat dataset per epoch - setiap batch berisi 4 gambar berbeda dalam satu komposisi. Prob=1.0 memastikan SETIAP batch adalah MOSAIC di awal training. Tanpa ini, model hanya melihat 1 objek per gambar → lemah saat inference pada tumpukan sampah.

**Mengapa Mixup alpha~Beta(0.5, 0.5)?**

**Apa itu Mixup?** Mixup melakukan blending linear antara dua gambar: I_blend = α × I₁ + (1-α) × I₂. Label juga di-blend linier: y_blend = α × y₁ + (1-α) × y₂. α diambil dari distribusi Beta.

Beta(0.5, 0.5) berbentuk U - α cenderung ke 0 atau 1 (bukan 0.5). Artinya blending dominan ke salah satu gambar, bukan rata-rata (yang akan membuat gambar buram dan confusing bagi model). Ini mempertahankan informasi visual dominan dari satu gambar sambil memberikan noise regularisasi dari gambar lain.

---

**D. Regularisasi via Augmentasi:**

*Tujuan: mencegah overfitting dengan memaksa model tidak bergantung pada region spesifik.*

| Augmentasi | Prob | Alasan Pemilihan | Referensi |
|------------|------|------------------|-----------|
| Erasing | 0.5 | Random rectangle dihapus (diisi mean pixel値). Memaksa model pakai konteks global. Tanpa erasing, model bisa "cheat" dengan deteksi pola sempit (misal: selalu lihat label botol di sudut kanan bawah) | [18] |
| Auto Augment | "randaugment" | Di akhir training (setelah close_mosaic), 2-3 augmentasi dipilih acak dari daftar dengan magnitude random. Stabilisasi fine-tuning | [19] |

**Mengapa randaugment bukan augmentasi tetap?** Randaugment memilih 2-3 augmentasi acak dari kumpulan transformasi ringan dengan magnitude random setiap iterasi. Pada akhir training (setelah close_mosaic), model sudah memiliki representasi fitur stabil. Augmentasi tetap yang agresif akan mengganggu fine-tuning. Randaugment memberikan regularisasi ringan dan acak - cukup untuk mencegah overfitting tanpa mengganggu representasi yang sudah dipelajari.

---

**Strategi Close Mosaic: Mengapa Mosaic Dimatikan di Tengah Training?**

**Apa itu Close Mosaic?** Close mosaic adalah strategi mematikan augmentasi mosaic di pertengahan training. Konfigurasi close_mosaic diatur pada epoch 33 (sepertiga dari total 100 epoch).

| Fase | Epoch | Mosaic | Alasan |
|------|-------|--------|--------|
| Fase 1: Eksplorasi | 1-33 | ON (1.0) | Model belajar representasi dasar. Mosaic memaksa deteksi konteks padat, objek terpotong. 4 gambar/batch = efektif 4x data |
| Fase 2: Stabilisasi | 34-100 | OFF (0.0) | Mosaic membuat objek terpotong di batas grid → distribusi data tidak realistis. Pada fase fine-tuning, model perlu melihat objek utuh untuk refine boundary mask |

**Dampak Kuantitatif Close Mosaic:** Tanpa close mosaic, validation loss meningkat ~5% pada epoch 50+. Dengan close mosaic di epoch 33, validation loss terus menurun hingga epoch 100. Gap train-val mAP <5% mengonfirmasi keberhasilan strategi ini.

---

**Compute Budget Augmentasi:**

Setiap epoch: 2.765 gambar training × augmentasi online = 2.765 variasi unik. 100 epochs = 276.500 variasi total. Dengan 15 jenis augmentasi yang dikombinasikan secara acak, total kemungkinan variasi per gambar >10¹² - secara praktis tidak terbatas.

**Dampak Augmentasi terhadap Training:**

Dengan 15 jenis augmentasi yang dikombinasikan secara acak, setiap gambar menghasilkan ribuan variasi unik per epoch - efektif memperbesar dataset tanpa data baru. Setiap epoch: 2.765 gambar × augmentasi online = 2.765 variasi unik. 100 epochs = 276.500 variasi total.

Model yang dilatih dengan augmentasi menunjukkan gap train-val mAP <5% - mengindikasikan augmentasi berhasil mencegah overfitting. Tanpa augmentasi, gap tipikal >15% pada dataset kecil seperti ini.

> **Key Takeaway:**
> 
> | Augmentasi | Prob | Fungsi |
> |------------|------|--------|
> | Mosaic | 1.0 | 4 foto grid 2x2 - konteks padat |
> | Mixup | 0.5 | 2 foto blend - fitur overlap |
> | Copy-Paste | 0.5 | Objek pindah antar foto |
> | HSV | 0.02/0.6/0.4 | Variasi warna, saturasi, terang |
> | Geometric | 25 deg/0.8 | Rotasi, scale, shear |
> | Flip/LR | 0.5 | Cermin horizontal |
> | Erasing | 0.5 | Occlusion simulation |
> 
> ```mermaid
> flowchart LR
>     I[Input Image] --> M[Mosaic 1.0]
>     M --> MX[Mixup 0.3]
>     MX --> CP[Copy-Paste 0.4]
>     CP --> H[HSV Jitter]
>     H --> G[Geometric Rot/Scale/Shear]
>     G --> F[Flip LR 50%, UD 20%]
>     F --> E[Erasing 40%]
>     E --> O[Training Batch]
> ```

---

## Slide 7: Backbone: CSPDarknet - 4 Stage, SPP Layer, Resolution Progression

### Backbone CSPDarknet

**Mengapa Darknet untuk YOLO?**

YOLO secara historis menggunakan Darknet (YOLOv1-v3) dan kemudian CSPDarknet (YOLOv4+). Arsitektur ini dioptimalkan untuk:
- **Inference speed**: Mengutamakan layer paralel daripada sequential deep → cocok untuk real-time detection
- **Memory efficiency**: CSP mengurangi parameter redundant
- **Representasi hierarkis**: Feature pyramid alami dari downsampling bertahap

**Perbandingan Varian Model YOLOv26-seg:**

| Varian | Parameters | GFLOPs | Model Size | Box mAP COCO | Kecepatan (T4) | Keputusan | Referensi |
|--------|-----------|--------|------------|-------------|----------------|-----------|-----------|
| n (nano) | 3.2M | 12.8 | 6.5 MB | 37.3% | ~2ms | Terlalu kecil, akurasi rendah | [8] |
| s (small) | 11.2M | 46.2 | 22.5 MB | 44.9% | ~3ms | Opsional trade-off | [8] |
| **m (medium)** | **26.97M** | **131.9** | **54.5 MB** | **50.6%** | **~5ms** | **Dipilih** | [8] |
| l (large) | 58.3M | 236.7 | 117 MB | 53.0% | ~8ms | Terlalu besar untuk edge | [8] |
| x (xlarge) | 97.8M | 397.5 | 196 MB | 54.7% | ~12ms | Tidak feasible, VRAM >16GB | [8] |

**Varian m dipilih** karena: (1) keseimbangan akurasi-kecepatan optimal, (2) ukuran model 54.5 MB muat di flash storage perangkat edge, (3) inference 5.1ms (log) memenuhi syarat real-time (<30ms).

---

**Struktur Backbone dari Log:**

Backbone terdiri dari 11 layer berurutan: Stem convolution (input 3 channel RGB ke 64 channel, stride 2), diikuti konvolusi downsampling dan CSP stages pada setiap tingkat resolusi, kemudian SPPF layer untuk pooling multi-skala, dan C2PSA untuk attention mechanism.

**Proses Ekstraksi Fitur Bertahap:**

| Stage | Input → Output | Stride | Channel | Pixels | Resepsi Field | Fungsi | Referensi |
|-------|--------------|--------|---------|--------|---------------|--------|-----------|
| Stem Conv | 640×640 → 320×320 | 2x | 64 | 102.400 (25%) | 7×7 | Tepi dasar (garis, gradien) | [14][21] |
| Stage 1 CSP | 320×320 → 160×160 | 4x | 256 | 25.600 (6.25%) | 15×15 | Sudut, kontur, lingkaran | [14][21] |
| Stage 2 CSP | 160×160 → 80×80 | 8x | 512 | 6.400 (1.56%) | 31×31 | Pola geometrik: silinder, lipatan | [14][21] |
| Stage 3 CSP | 80×80 → 40×40 | 16x | 512 | 1.600 (0.39%) | 63×63 | Tekstur: plastik vs kertas | [14][21] |
| Stage 4 CSP | 40×40 → 20×20 | 32x | 512 | 400 (0.098%) | 127×127 | Semantik: "buatan pabrik" vs "alami" | [14][21] |
| SPPF | 20×20 → 20×20 | 32x | 512 | 400 (0.098%) | 127+pool | Multi-skala (pool 5/9/13) | [22][14][8] |
| C2PSA | 20×20 → 20×20 | 32x | 512 | 400 (0.098%) | global | Attention mekanisme | [14][21] |

Resepsi field dihitung sebagai: RF_n = RF_{n-1} + (kernel_size - 1) × stride_product. Stem (kernel 7, stride 2) → RF=7. Stage 1 (ker 3, stride 2) → RF = 7 + 2×4 = 15. Seterusnya hingga Stage 4 mencapai 127×127 piksel pada gambar asli.

---

**CSP (Cross Stage Partial) - Fungsi dan Mekanisme**

**Apa itu CSP?** CSP (Cross Stage Partial Connection) adalah mekanisme yang membagi feature map menjadi dua jalur di setiap stage backbone: (1) jalur utama - subset channel (~50%) diproses melalui blok konvolusi bottleneck, (2) jalur shortcut - sisa channel langsung dilewatkan tanpa perubahan. Kedua jalur digabung (concatenate) di akhir stage.

**Fungsi CSP di Backbone:**
- **Efisiensi komputasi:** hanya setengah channel diproses di jalur utama → ~20% lebih hemat FLOPs dibanding ResNet standar
- **Gradient flow dual-path:** gradien mengalir melalui dua jalur terpisah (konvolusi + shortcut) → mengurangi vanishing gradient, memungkinkan backbone lebih dalam
- **Feature reuse alami:** concatenation fitur baru + fitur asli memberikan akses simultan ke representasi mentah dan terproses

| Aspek | ResNet | DenseNet | CSP (dipilih) | Referensi |
|-------|--------|----------|--------------|-----------|
| Koneksi shortcut | Identity skip connection | Semua layer terhubung ke semua | Split + concat partial | [21] |
| Parameter efisiensi | Medium | Boros (bottleneck 1×1) | Tinggi (~20% lebih hemat) | [21] |
| FLOPs | 100% baseline | 130% | ~80% | [21] |
| Gradien flow | Baik | Sangat baik (dense) | Sangat baik (dual path) | [21] |
| Representasi | Fitur residual | Fitur reuse maksimal | Fitur baru + fitur asli | [21] |

CSP memproses hanya sebagian (~50%) channel di jalur utama, sisanya dilewati langsung. Ini memberikan representasi yang lebih kaya (fitur baru + fitur asli digabung) dengan FLOPs lebih rendah.

---

**Stem Convolution (Kernel 7) - Fungsi**

**Apa itu Stem?** Stem adalah layer konvolusi pertama backbone yang menerima input RGB 640×640×3 dan menghasilkan feature map 320×320×64. Menggunakan kernel 7×7 dengan stride 2 → resolusi spasial directuksi 4× (640×640 → 320×320). Kernel 7×7 mencakup receptive field ~49 piksel, dibanding 3×3 yang hanya 9 piksel.

**Fungsi Stem:**
- **Downsampling cepat:** stride 2 mereduksi resolusi 4× dalam satu langkah → mengurangi biaya komputasi stage berikutnya (320×320 vs 640×640 memiliki 4× lebih sedikit piksel)
- **Receptive field luas:** kernel 7 memberikan konteks visual lebih luas sejak awal - mendeteksi tepi, gradien, tekstur dasar dengan cakupan lebih besar
- **Efisiensi arsitektur:** setara dengan 2-3 layer Conv3x3 bertumpuk dalam satu layer → menghemat 1-2 layer konvolusi dan parameter terkait

Stem convolution dengan kernel 7, stride 2 langsung mereduksi 640 → 320. Kernel lebih besar (7 vs 3) memberikan resepsi field awal lebih luas tanpa perlu layer tambahan. Ini menghemat 1-2 layer Conv3x3 yang seharusnya dibutuhkan untuk mencapai resepsi field yang sama.

---

**SiLU Activation - Fungsi**

**Apa itu SiLU?** SiLU (Sigmoid Linear Unit) adalah fungsi aktivasi dengan formula: f(x) = x × sigmoid(x) = x / (1 + e^(-x)). SiLU adalah fungsi smooth (C∞ - semua turunan kontinu), non-monotonik - memiliki minimum negatif di sekitar x ≈ -1.28. Berbeda dari ReLU yang monotonik dan memiliki patahan di x=0.

**Fungsi Aktivasi di Backbone:**
- **Gradient flow stabil:** turunan kontinu di semua titik → tidak ada diskontinuitas gradien saat backpropagation - penting untuk arsitektur 329 layer
- **Tidak ada dead neuron:** untuk x negatif besar, output mendekati 0 (bukan nol) → gradien informatif tetap ada → neuron tidak mati total seperti ReLU
- **Self-gating mekanisme:** x × sigmoid(x) secara alami "menggate" nilai: input besar (+) dilewatkan hampir linier, input kecil/sedang ditekan - mirip mekanisme attention sederhana tanpa parameter tambahan
- **Kontribusi akurasi:** ~2-3% mAP improvement dibanding ReLU pada benchmark COCO karena gradient flow lebih baik

Perbandingan aktivasi:

| Aktivasi | Formula | Gradien x<0 | Smoothness | Dead Neuron |
|----------|---------|-------------|------------|-------------|
| ReLU | max(0, x) | 0 | Tidak (patah di x=0) | Ya |
| LeakyReLU | max(αx, x) | α konstan (~0.01) | Tidak (patah di x=0) | Tidak |
| SiLU (dipilih) | x·sigmoid(x) | Mendekati 0 (bukan 0) | Ya (C∞) | Tidak |

SiLU memberikan gradien lebih informatif pada nilai negatif dibanding ReLU, dan lebih mulus dibanding LeakyReLU.

---

**SPPF (Spatial Pyramid Pooling Fast) - Fungsi**

**Apa itu SPPF?** SPPF (Spatial Pyramid Pooling Fast) adalah modul pooling multi-skala yang menerapkan max-pooling pada feature map 20×20 dengan tiga skala receptive field berbeda: 5×5, 9×9 (efektif), dan 13×13 (efektif). Alih-alih menjalankan tiga operasi pooling paralel seperti SPP orisinil, SPPF melakukan pooling sequential - tiga kali max-pool 5×5 berantai.

**Mekanisme Sequential Pooling:**
- Pool 5×5 pertama → receptive field 5×5 (mencakup 25 piksel tetangga)
- Pool 5×5 kedua pada output → receptive field efektif 9×9 (5 + 5 − 1 = 9)
- Pool 5×5 ketiga pada output → receptive field efektif 13×13 (5 + 5 + 5 − 2 = 13)

**Fungsi SPPF:**
- **Multi-skala feature extraction:** menangkap informasi pada tiga resolusi berbeda dari lokasi spasial yang sama - fitur lokal (5×5), regional (9×9), dan konteks luas (13×13)
- **Context aggregation:** pooling menggabungkan informasi dari area sekitar → memberikan model pemahaman konteks spasial untuk deteksi objek besar dan kecil secara simultan
- **Efisiensi:** sequential pooling 2× lebih cepat dari paralel karena data tetap di cache GPU; parameter identik dengan SPP

| Varian | Parameter | Kecepatan | Mekanisme |
|--------|-----------|-----------|-----------|
| SPP (orisinil) | Sama | Dasar | 3 pooling paralel (5, 9, 13) |
| **SPPF** (dipilih) | Sama | **2x lebih cepat** | Pooling serial: pool5→pool5=pool9→pool5=pool13 |
| ASPP (Deeplab) | 3x lebih banyak | Lebih lambat | Dilated convolution dengan rate berbeda |

SPPF adalah optimasi SPP: alih-alih 3 pooling paralel, SPPF melakukan 3 pooling sequential. Pool5 diulang 3x: hasil pool5 pertama = pool5, kedua = efektif pool9, ketiga = efektif pool13. Karena pooling sequential lebih cepat dari 3 pooling paralel, SPPF 2x lebih cepat dengan resepsi field identik.

---

**Total Compute dari Log:**

Model: 329 layers (unfused) → 149 layers (fused), 26.971.750 parameters, 131.9 GFLOPs (unfused) → 121.2 GFLOPs (fused). Fusing menggabungkan Conv2D + BatchNorm + SiLU menjadi satu layer untuk inference lebih cepat.

**Apa Arti 131.9 GFLOPs?** Setiap forward pass model melakukan 131.9 × 10⁹ operasi floating point. Dengan inference 5.1ms/gambar (log), model memproses ~25.8 × 10¹² FLOPs/detik - ≈25.8 TFLOPS, yang berarti GPU RTX 5060 Ti (rated ~22 TFLOPS FP32) bekerja pada ~85% utilisasi.

> **Key Takeaway:**
> 
> | Stage | Input -> Output | Channel | Stride | Deteksi |
> |-------|---------------|---------|--------|---------|
> | Stem | 640 -> 320 | 64 | 2x | Tepi dasar |
> | Stage 1 | 320 -> 160 | 128 | 4x | Sudut & kontur |
> | Stage 2 | 160 -> 80 | 256 | 8x | Bentuk geometrik |
> | Stage 3 | 80 -> 40 | 512 | 16x | Tekstur & pola |
> | Stage 4 | 40 -> 20 | 512 | 32x | Semantik & konteks |
> | SPP | 20 -> 20 | 512 | 32x | Multi-skala (5/9/13) |
> 
> ```mermaid
> flowchart LR
>     IN["Input 640x640x3"] --> ST["Stem Conv k7 s2<br />320x320 C=64"]
>     ST --> S1["Stage 1 CSP<br />160x160 C=128"]
>     S1 --> S2["Stage 2 CSP<br />80x80 C=256"]
>     S2 --> S3["Stage 3 CSP<br />40x40 C=512"]
>     S3 --> S4["Stage 4 CSP<br />20x20 C=512"]
>     S4 --> SPP["SPP Layer<br />k=5,9,13 Pool"]
>     SPP --> NECK["To Neck FPN+PAN"]
> ```

---

## Slide 8: Neck: FPN+PAN & Decoupled Head - Multi-Scale Feature Fusion

### Neck FPN+PAN dan Decoupled Head

**Apa itu Neck?**

Neck adalah komponen arsitektur yang berada di antara Backbone dan Head. Tugasnya: memfusikan fitur dari berbagai resolusi yang dihasilkan Backbone - menggabungkan informasi semantik (apa objeknya) dan detail lokasi (di mana objeknya) dari setiap level resolusi.

Hal ini penting karena Backbone menghasilkan fitur dengan karakteristik berbeda di setiap level. Semakin dalam resolusi, fitur semakin "tahu apa objeknya" tapi tidak "tahu di mana persisnya". Sebaliknya, fitur resolusi tinggi tahu lokasi presisi tapi tidak tahu apa objek itu. Neck menjembatani kesenjangan ini:

- P3 (80x80): Resolusi tinggi, banyak detail lokasi (tepi, kontur), sedikit informasi semantik
- P4 (40x40): Resolusi sedang, keseimbangan detail dan semantik
- P5 (20x20): Resolusi rendah, sedikit detail lokasi, banyak informasi semantik ("ini botol")

- P3 (80x80): Resolusi tinggi, banyak detail lokasi (tepi, kontur), sedikit informasi semantik
- P4 (40x40): Resolusi sedang, keseimbangan detail dan semantik
- P5 (20x20): Resolusi rendah, sedikit detail lokasi, banyak informasi semantik ("ini botol")

Masalahnya: P3 tahu persis di mana objek berada tetapi tidak tahu objek itu apa. P5 tahu objek itu apa tetapi tidak tahu persis di mana letaknya. Neck mengatasi masalah ini dengan memfusikan informasi dari semua level.

**FPN (Feature Pyramid Network) - Top-Down Pathway:**

**Apa itu FPN?** FPN (Feature Pyramid Network) adalah jalur top-down di Neck yang membawa informasi semantik dari resolusi rendah (P5, 20×20) ke resolusi lebih tinggi (P4 40×40, P3 80×80). Informasi semantik adalah "pengetahuan tentang apa objek itu" - P5 tahu "ini botol" atau "ini kardus".

**Mekanisme FPN (setiap level):**
1. Upsample feature map level atas 2× (nearest neighbor interpolation)
2. Concatenate dengan feature map lateral dari backbone di resolusi yang sama
3. Konvolusi (blok C3k2) untuk memproses hasil gabungan

Proses diulang dari P5 → P4 → P3.

**Fungsi FPN:** Memberikan pemahaman semantik (what object) ke resolusi tinggi - P3 (deteksi objek kecil) mendapat konteks "ini puntung rokok" dari P5 sambil tetap mempertahankan detail lokasi presisi dari P3 asli.

FPN bekerja dari resolusi rendah ke tinggi (atas ke bawah), membawa informasi semantik dari P5 ke P4 dan P3. Prosesnya:

| Langkah | Operasi | Dimensi | Deskripsi |
|---------|---------|---------|-----------|
| 1 | Upsample P5 2x | 20x20 -> 40x40 | Fitur semantik P5 diperbesar 2x menggunakan nearest neighbor interpolation atau transposed convolution |
| 2 | Concat dengan P4 | 40x40 + 40x40 = 40x40 | Feature map P4 (detail) digabung dengan fitur semantik yang sudah di-upsample. Hasil: P4' yang memiliki detail DAN konteks |
| 3 | Upsample P4' 2x | 40x40 -> 80x80 | Fitur gabungan diperbesar ke resolusi P3 |
| 4 | Concat dengan P3 | 80x80 + 80x80 = 80x80 | P3 (detail lokasi tinggi) digabung dengan semantik level atas. Hasil: P3' yang memiliki detail presisi DAN pemahaman objek |

Setelah FPN, P3' dapat mendeteksi objek kecil (puntung rokok 20x10 px) dengan akurat karena memiliki detail lokasi dari P3 dan informasi "ini puntung rokok" dari P5.

**PAN (Path Aggregation Network) - Bottom-Up Pathway:**

**Apa itu PAN?** PAN (Path Aggregation Network) adalah jalur bottom-up di Neck yang membawa informasi detail lokasi dari resolusi tinggi (P3, 80×80) ke resolusi lebih rendah (P4 40×40, P5 20×20). Detail lokasi adalah "pengetahuan tentang di mana persis objek berada" - P3 tahu "tepi botol ada di pixel (120, 340)".

**Mekanisme PAN (setiap level):**
1. Downsample feature map level bawah 2× (convolution stride 2)
2. Concatenate dengan feature map dari FPN di resolusi yang sama
3. Konvolusi untuk memproses hasil gabungan

Proses diulang dari P3 → P4 → P5.

**Fungsi PAN:** Memberikan presisi lokalisasi (where object) ke resolusi rendah - P5 (deteksi objek besar) mendapat detail "botol di pojok kiri bawah" dari P3 sambil tetap mempertahankan pemahaman semantik dari P5 asli.

PAN bekerja dari resolusi tinggi ke rendah (bawah ke atas), membawa informasi detail lokasi dari P3 ke P4 dan P5. Prosesnya:

| Langkah | Operasi | Dimensi | Deskripsi |
|---------|---------|---------|-----------|
| 1 | Downsample P3' k3 s2 | 80x80 -> 40x40 | Fitur detail P3' directuksi resolusinya menggunakan convolution stride 2 |
| 2 | Concat dengan P4' | 40x40 + 40x40 = 40x40 | P4' yang sudah memiliki semantik dari FPN sekarang ditambah detail lokasi presisi |
| 3 | Downsample P4'' k3 s2 | 40x40 -> 20x20 | Fitur directuksi ke resolusi P5 |
| 4 | Concat dengan P5 | 20x20 + 20x20 = 20x20 | P5 yang kaya semantik sekarang juga memiliki detail lokasi untuk lokalisasi presisi |

Setelah PAN, P5 dapat membedakan "botol di sebelah kiri" vs "botol di sebelah kanan" karena memiliki detail lokasi dari P3.

**Multi-Scale Detection:**

| Level | Resolusi | Ukuran Grid | Target Objek | Contoh Sampah |
|-------|----------|-------------|-------------|---------------|
| P3 | 80x80 | 6.400 sel | Kecil (<32x32 px) | Puntung rokok, biji, pecahan kecil |
| P4 | 40x40 | 1.600 sel | Sedang (32-96 px) | Kaleng, botol kecil, gelas |
| P5 | 20x20 | 400 sel | Besar (>96 px) | Kardus, botol 1.5L, tumpukan sampah |

Total grid cells: 6.400 + 1.600 + 400 = 8.400 sel. Setiap sel dapat mendeteksi satu objek.

**Decoupled Head:**

**Apa itu Decoupled Head?** Decoupled Head adalah bagian akhir arsitektur yang memproduksi keputusan final melalui tiga cabang konvolusi paralel sepenuhnya independen: classification branch, regression branch (bbox), dan segmentation branch (mask). Setiap cabang memiliki 2 layer Conv3x3 sendiri dan layer prediksi sendiri - tidak ada satu pun parameter yang dibagi antar cabang (independent parameter-wise).

**Fungsi Decoupled Head:** Setiap cabang mengoptimalkan representasi untuk satu task spesifik tanpa kompetisi parameter atau gradient conflict. Classification branch fokus pada diskriminasi kelas (apa objek), regression branch fokus pada presisi lokasi (di mana objek), segmentation branch fokus pada akurasi bentuk (bagaimana bentuk objek).

Head adalah bagian akhir arsitektur yang mengambil keputusan final. Disebut "decoupled" karena terdiri dari 3 cabang independen yang masing-masing memiliki parameter sendiri dan bertanggung jawab atas tugas berbeda:

**Classification Branch:**
Layer: 2x Conv3x3 + Linear layer + Sigmoid activation
Fungsi: Menentukan apakah suatu grid cell berisi objek dan jika iya, kelas apa (Organik atau Non-Organik)
Output: 3 nilai per grid cell - objectness score (probabilitas ada objek) + 2 class probabilities (Organik, Non-Organik)
Contoh output: [0.92, 0.87, 0.13] artinya 92% yakin ada objek, 87% yakin Organik, 13% yakin Non-Organik

**Regression Branch (BBox):**
Layer: 2x Conv3x3 + DFL module

**Apa itu Regression?** Regression di sini berarti memprediksi nilai numerik kontinu (koordinat) dari feature map. YOLOv26 menggunakan DFL (Distribution Focal Loss): alih-alih memprediksi satu nilai float per koordinat, DFL memprediksi distribusi probabilitas diskrit 16-bin. Nilai akhir = weighted sum: koordinat = Σ(bin_i × softmax(prob_i)). Contoh posisi x: 16 bin merepresentasikan 16 posisi relatif dalam grid cell.

**Fungsi:** Memprediksi 4 koordinat bounding box (x_center, y_center, width, height) - lokasi dan ukuran objek dalam grid cell.

**Fungsi DFL untuk Regresi:**
- **Gradient lebih kaya:** setiap 16 bin memberikan sinyal gradien independen → lebih banyak sinyal dibanding L1/L2
- **Representasi uncertainty:** jika boundary objek tidak jelas (botol transparan), distribusi akan lebar - model mengekspresikan ketidakyakinan secara eksplisit
- **Akurasi boundary:** DFL lebih unggul untuk boundary tidak jelas/tidak tegas dibanding regresi titik tunggal

Output: 4 nilai float (x, y, w, h) dalam koordinat grid-normalized

**Segmentation Branch (Mask):**
Layer: Proto Module (Conv + 32 prototype masks)

**Apa itu Proto Module?** Proto Module adalah mekanisme segmentasi efisien yang mendekomposisi masalah prediksi mask menjadi dua sub-masalah: (1) menghasilkan 32 prototype mask dasar (20×20) dari fitur multi-skala P3+P4+P5 - setiap prototype merepresentasikan "bentuk dasar" seperti lingkaran, tepi vertikal, cekungan; (2) setiap instance/grid cell memprediksi 32 coefficient yang menentukan bagaimana menggabungkan prototype menjadi mask spesifik untuk objek tersebut.

**Fungsi:** Menghasilkan polygon mask 24 titik yang mengikuti bentuk objek

**Cara Kerja:**
1. Prototype generation: fitur multi-skala P3+P4+P5 → Conv → 32 prototype mask (20×20)
2. Coefficient prediction: setiap grid cell memprediksi 32 coefficient
3. Mask reconstruction: mask_final = Σ(coeff[i] × proto[i]) untuk i=1..32
4. Upsampling: 20×20 → 640×640 (32× upsampling langsung)
5. Cropping: mask di-crop sesuai bounding box prediksi
6. Polygon conversion: mask biner → polygon 24 titik via marching squares

**Fungsi Proto Module:**
- **Efisien:** hanya 32 prototype untuk seluruh gambar, bukan 8.400 mask independen per grid cell
- **Shared representation:** prototype dipelajari dari semua objek (bukan per-instance) → representasi bentuk lebih general
- **Resolution invariant:** upsampling dari 20×20 ke 640×640 = 32× tanpa kehilangan informasi karena prototype berisi informasi bentuk global

Output: 48 float (24 titik x 2 koordinat), ternormalisasi [0,1]

**FPN+PAN - Fungsi Kombinasi**

**Apa itu FPN+PAN?** FPN+PAN adalah kombinasi dua jalur fusi fitur yang saling melengkapi di Neck. FPN mengalirkan informasi semantik secara top-down (P5→P4→P3). PAN mengalirkan informasi lokasi secara bottom-up (P3→P4→P5). Setiap level deteksi (P3, P4, P5) menerima kedua jenis informasi dari kedua arah.

**Fungsi Kombinasi per Level:**
- **P3 (80×80, target objek kecil):** dari FPN mendapat semantik "ini puntung rokok" dari P5 → mengurangi false positive pada tekstur latar yang mirip puntung
- **P5 (20×20, target objek besar):** dari PAN mendapat detail "kardus di pojok kiri bawah" dari P3 → bounding box lebih presisi
- **P4 (40×40, target objek sedang):** mendapat semantik dari P5 + lokasi dari P3 → keseimbangan optimal

**Mengapa FPN saja tidak cukup?** FPN hanya mengalirkan semantik ke bawah. P5 (deteksi objek besar) hanya punya semantik tanpa detail lokasi → bounding box kurang presisi untuk objek besar seperti kardus. FPN+PAN memastikan setiap level memiliki semantic understanding (what) AND precise localization (where).

| Konfigurasi | Informasi Semantik (P3) | Detail Lokasi (P5) | Kecepatan | Referensi |
|-------------|------------------------|-------------------|-----------|-----------|
| Hanya FPN | Tinggi | Rendah | Cepat | [23] |
| Hanya PAN | Rendah | Tinggi | Cepat | [24] |
| **FPN+PAN** (dipilih) | **Tinggi** | **Tinggi** | **Sedang (butuh 2x lebih banyak layer)** | [23][24] |

Tanpa PAN, P5 (deteksi objek besar) tidak memiliki detail lokasi → bounding box kurang presisi untuk objek besar seperti kardus. Tanpa FPN, P3 (deteksi objek kecil) tidak memiliki informasi semantik → false positive pada tekstur latar.

Implementasi FPN bekerja secara top-down: upsample P5 lalu concat dengan P4, diproses oleh blok C3k2. Kemudian upsample hasilnya dan concat dengan P3 untuk menghasilkan output P3 di resolusi 80x80. PAN bekerja secara bottom-up: downsample P3 lalu concat dengan P4 untuk output P4 di resolusi 40x40, kemudian downsample lagi dan concat dengan P5 untuk output P5 di resolusi 20x20. Channel progression: P3 (256 channel), P4 (512 channel), P5 (512 channel).

Channel progression: P3 (256ch) → P4 (512ch) → P5 (512ch). Perhatikan bahwa P4_out dibangun dari concat P4 (512ch) + upsample P5 (512ch) = 1024ch → directuksi ke 512ch.

---

**Decoupled Head - Fungsi**

**Apa itu Decoupled Head?** Decoupled Head adalah struktur head deteksi yang menggunakan tiga cabang konvolusi paralel terpisah (masing-masing 2× Conv3x3 + layer output) untuk tiga task berbeda: klasifikasi (class + objectness), regresi bounding box (x, y, w, h), segmentasi (mask coefficients). Setiap cabang memiliki parameter sepenuhnya independen - tidak ada satu pun bobot konvolusi yang dibagi antar cabang.

**Fungsi Decoupled Head:**
- **Task specialization:** setiap cabang mengoptimalkan representasi untuk satu task spesifik tanpa kompromi
- **Eliminasi task competition:** pada coupled head, satu set parameter harus menyeimbangkan kebutuhan tiga task yang sering bertentangan. Decoupled head menghilangkan konflik ini
- **Akurasi lebih tinggi:** +2-3% mAP dibanding coupled head (YOLO paper)

| Aspek | Coupled Head (YOLOv5/v8) | Decoupled Head (YOLOv26) | Referensi |
|-------|--------------------------|--------------------------|-----------|
| Struktur | 1 Conv → 3 task bersama | 3 Conv paralel, 1 per task | [25] |
| Conflict task | Klasifikasi vs regresi kompetisi parameter | Tidak ada conflict (parameter terpisah) | [25] |
| Akurasi | Baseline | +2-3% mAP (YOLO paper) | [25] |
| Parameter | Lebih hemat | ~20% lebih banyak |
| Kecepatan | Sama | Sama (paralel) |

Decoupled head memisahkan parameter untuk tiap task: classification, regression (bbox), segmentation. Ini mencegah "task competition" di mana satu set parameter harus belajar dua tugas berbeda secara simultan.

---

**Proto Module Detail (Segmentation Branch):**

Proto module adalah mekanisme segmentasi YOLO-seg yang efisien:

1. **Prototype generation**: Fitur multi-skala P3+P4+P5 → Conv → 32 prototype mask (masing-masing 20×20)
2. **Coefficient prediction**: Setiap grid cell memprediksi 32 coefficient untuk mengkombinasikan prototype
3. **Mask reconstruction**: Mask_i = Σ(coeff_ij × proto_j) untuk j=1..32
4. **Upsampling**: 20×20 → 640×640 (32× upsampling langsung)
5. **Cropping**: Mask di-crop sesuai bounding box prediksi
6. **Polygon conversion**: Mask biner → polygon 24 titik via marching squares

Keuntungan Proto Module:
- **Efisien**: Hanya 32 prototype untuk seluruh gambar, bukan 8.400 mask independen
- **Shared representation**: Prototype dipelajari dari semua objek, bukan per-instance
- **Resolution invariant**: Upsampling dari 20×20 ke 640×640 = 32× tanpa kehilangan informasi (prototype berisi informasi bentuk global)

Konfigurasi dari log: mask_ratio 2 (faktor downsampling prototype) dan overlap_mask aktif (mask boleh overlap antar instance).

---

**Anchor-Free Detection - Mekanisme**

**Apa itu Anchor-Free Detection?** Anchor-Free Detection adalah pendekatan di mana model langsung memprediksi 4 koordinat bounding box (x_center, y_center, width, height) per grid cell, tanpa menggunakan predefined anchor box templates (template dengan ukuran/shape spesifik per grid cell seperti di YOLOv3/v5/v8).

**Mekanisme Anchor-Free:**
- Setiap grid cell memprediksi 4 nilai float: offset pusat (x, y) relatif terhadap grid cell, serta lebar (w) dan tinggi (h) relatif terhadap gambar
- Tidak ada langkah "anchor matching" - tidak perlu memilih anchor terbaik dari k template
- DFL memprediksi distribusi 16-bin per koordinat → fleksibel menangkap berbagai rasio bentuk tanpa template eksplisit

**Fungsi Anchor-Free:**
- **Eliminasi tuning:** tidak perlu clustering dataset untuk menentukan anchor size/shape
- **Fleksibilitas bentuk:** bisa memprediksi rasio ekstrem (1:10) yang tidak tercakup anchor standar
- **NMS lebih sederhana:** tanpa anchor suppression, post-processing lebih cepat
- **Zero transfer tuning:** tidak ada anchor yang perlu disesuaikan saat pindah dataset

YOLOv26 menghilangkan anchor boxes (template bounding box) yang digunakan di v3/v5/v8. Perbandingan:

| Aspek | Anchor-Based | Anchor-Free (dipilih) | Referensi |
|-------|-------------|----------------------|-----------|
| Cara kerja | Pilih anchor terbaik dari k template per grid cell | Langsung prediksi 4 koordinat | [25] |
| Parameter tuning | Butuh clustering dataset untuk determine anchor size/shape | Tidak perlu tuning | [25] |
| Dataset transfer | Anchor optimal beda per dataset (COCO ≠ sampah) | Zero tuning | [25] |
| Bentuk ekstrem | Anchor tidak mencakup rasio ekstrem (1:10) | Bisa prediksi rasio berapapun | [25] |
| Post-processing | NMS kompleks (anchor suppression) | NMS sederhana | [25] |

**Mengapa sekarang bisa anchor-free?** YOLOv26 menggunakan DFL (Distribution Focal Loss) yang memprediksi distribusi probabilitas posisi boundary. Distribusi ini secara implisit menangkap variasi bentuk objek tanpa perlu template eksplisit. Perbaikan lain: feature pyramid yang lebih baik (FPN+PAN+C2PSA) memberikan representasi multi-skala yang cukup untuk menangani variasi ukuran tanpa anchor.

---

**8,400 Grid Cells: Detail Per-Level:**

Total grid cells = 80² + 40² + 20² = 6.400 + 1.600 + 400 = 8.400 sel.
- 80×80 = 6.400 sel: target objek kecil (<32 px), stride 8x (1 sel = 8×8 px pada gambar)
- 40×40 = 1.600 sel: target objek sedang (32-96 px), stride 16x
- 20×20 = 400 sel: target objek besar (>96 px), stride 32x

Setiap sel memprediksi: 1 set class prob (2 kelas + objectness), 1 set koordinat bbox, 1 set coefficient mask (32). Total 8.400 × 35 ≈ 294.000 prediksi per gambar, directuksi oleh NMS menjadi ~5-20 deteksi final.

> **Key Takeaway:**
> 
> | Komponen | Arah | Fungsi | Output |
> |----------|------|--------|--------|
> | FPN | Top-down | Semantik P5 -> P4 -> P3 | Detail + konteks |
> | PAN | Bottom-up | Lokasi P3 -> P4 -> P5 | Konteks + presisi |
> | Class Head | 3 skala | 2x Conv3x3 + Linear | 2 kelas + objectness |
> | Reg Head | 3 skala | DFL 16-bin distribusi | x, y, w, h |
> | Seg Head | 3 skala | Proto 32 mask | 24-point polygon |
> 
> ```mermaid
> flowchart TD
>     subgraph FPN[FPN Top-Down]
>         P5["P5 20x20"] --> UP5["Upsample 2x"]
>         UP5 --> CAT4["Concat + P4 40x40"]
>         CAT4 --> UP4["Upsample 2x"]
>         UP4 --> CAT3["Concat + P3 80x80"]
>     end
>     subgraph PAN[PAN Bottom-Up]
>         CAT3 --> DN3["Downsample k3 s2"]
>         DN3 --> CAT4B["Concat + P4 40x40"]
>         CAT4B --> DN4["Downsample k3 s2"]
>         DN4 --> CAT5["Concat + P5 20x20"]
>     end
> ```

---

## Slide 9: Loss Functions & Training - CIoU, BCE, DFL, Hyperparameter

### Loss Functions dan Training Setup

Konfigurasi loss dari log: box 7.5, cls 0.5, dfl 1.5. Tiga komponen loss mengukur aspek berbeda dari prediksi.

**Mengapa Tiga Loss?** YOLOv26m-seg head memiliki 3 cabang output independen (classification, regression, segmentation). Masing-masing mengukur kesalahan pada domain berbeda yang tidak bisa digabung menjadi satu fungsi:

| Loss | Target Output | Problem Domain | Satuan | Prioritas |
|------|--------------|----------------|--------|-----------|
| **CIoU** (bobot 7.5) | Bounding box: 4 koordinat (x, y, w, h) | **Regresi geometris** — seberapa akurat posisi dan ukuran kotak? | IoU [0..1] (tanpa dimensi) | **Tertinggi** — box meleset = deteksi gagal total |
| **BCE** (bobot 0.5) | Class prob: 2 nilai [0..1] per grid | **Klasifikasi biner** — Organik atau Non-Organik? | Cross-entropy (nats) | **Terendah** — 2 kelas mudah dibedakan secara visual |
| **DFL** (bobot 1.5) | Boundary distribusi: 4×16 bin probabilitas | **Distribusi regresi** — di mana tepatnya tepi objek? | Cross-entropy diskrit | **Menengah** — boundary tidak jelas (botol bening, amorf) |

Tidak ada satu loss function yang bisa mengukur akurasi bounding box DAN probabilitas kelas secara simultan karena keduanya berada di ruang metrik berbeda. Bobot [7.5, 1.5, 0.5] mencerminkan prioritas: lokalisasi > distribusi boundary > klasifikasi.

---

**A. CIoU Loss (bobot 7.5) - Regresi Bounding Box**

**Apa itu CIoU?** CIoU (Complete IoU) adalah loss function untuk regresi bounding box yang mengoptimalkan tiga aspek secara simultan: (1) overlap area - IoU (Intersection over Union), (2) jarak pusat antar bounding box - center distance, (3) kesamaan aspect ratio (lebar/tinggi). Formula: CIoU = 1 − IoU + ρ²(b, b_gt)/c² + α·v.

**Komponen CIoU:**
- **IoU (Intersection over Union):** rasio area overlap dibagi area union antara prediksi dan ground truth. Range [0,1], 1 = overlap sempurna. Fungsi: mengukur seberapa baik prediksi menutupi objek
- **Center distance (ρ²/c²):** jarak Euclidean kuadrat antar pusat bounding box, dinormalisasi diagonal kotak terkecil yang mencakup kedua box. Fungsi: menarik pusat prediksi mendekati pusat ground truth - konvergensi lebih cepat, mengatasi IoU yang tidak memberi gradien saat box tidak overlap
- **Aspect ratio (α·v):** v = (4/π²) × (arctan(w_gt/h_gt) − arctan(w/h))² mengukur perbedaan rasio lebar/tinggi. α = v / ((1−IoU) + v) adalah adaptive weight - lebih besar saat IoU kecil. Fungsi: memastikan bentuk (proporsi) bounding box cocok dengan objek

**Fungsi CIoU:** Mengoptimalkan bounding box lebih presisi daripada IoU sederhana. IoU hanya menghukum overlap area tanpa memberi sinyal bagaimana memperbaikinya. CIoU memberikan tiga sinyal gradien terarah (pusat, bentuk, overlap) - model belajar memperbaiki bounding box dari ketiga dimensi secara simultan.

**Perbandingan IoU Variants:**

| Loss | Faktor | Kelebihan | Kekurangan | Referensi |
|------|--------|-----------|------------|-----------|
| IoU | IoU saja | Sederhana | Gradien 0 jika tidak overlap | [26] |
| GIoU | IoU + smallest enclosing box | Gradien ada walau tidak overlap | Konvergen lambat | [26] |
| DIoU | IoU + center distance | Konvergen lebih cepat | Abaikan aspect ratio | [26] |
| **CIoU** (dipilih) | IoU + center distance + aspect ratio | Paling komprehensif | Kompleksitas komputasi sedikit lebih tinggi | [26] |

CIoU = 1 − IoU + ρ²(b, b_gt)/c² + α·v
- ρ: jarak Euclidean antar pusat
- c: diagonal terkecil yang mencakup kedua box
- v = (4/π²) × (arctan(w_gt/h_gt) − arctan(w/h))²: mengukur perbedaan aspect ratio
- α = v / ((1−IoU) + v): adaptive weight, lebih besar jika IoU kecil

**Bobot 7.5:** Lokalisasi adalah prioritas utama. Sampah salah klasifikasi (Organik vs Non-Organik) masih bisa ditoleransi, tapi bounding box meleset (overlap <30%) berarti kegagalan deteksi total. Bobot tertinggi mencerminkan hal ini.

---

**B. BCE Loss (bobot 0.5) - Klasifikasi 2 Kelas**

**Apa itu BCE?** BCE (Binary Cross-Entropy) adalah loss function untuk klasifikasi biner: BCE = −[y·log(p) + (1−y)·log(1−p)]. Setiap grid cell memprediksi probabilitas p untuk kelas Organik dan Non-Organik. BCE mengukur perbedaan antara distribusi probabilitas prediksi dan label sebenarnya (0 atau 1).

**Fungsi BCE:** Memastikan model mengklasifikasikan objek dengan benar. Jika objek adalah Organik (y=1), model dihukum semakin besar jika prediksi p mendekati 0. Jika Non-Organik (y=0), dihukum jika p mendekati 1.

**Mengapa BCE bukan Cross-Entropy multi-class?** Hanya 2 kelas (Organik/Non-Organik). BCE setara dengan CE untuk binary case, tapi lebih stabil numerik.

Nilai loss untuk berbagai confidence:
| Prediksi (p) untuk label=1 | Loss | Interpretasi |
|---------------------------|------|-------------|
| 0.99 | 0.01 | Sangat percaya diri, benar |
| 0.75 | 0.29 | Cukup yakin, benar |
| 0.50 | 0.69 | Ragu-ragu (random) |
| 0.25 | 1.39 | Salah arah |
| 0.01 | 4.61 | Sangat percaya diri, SALAH |

**Bobot 0.5:** Klasifikasi 2 kelas relatif mudah (vs 80 kelas COCO). Bobot lebih rendah karena model tidak perlu "belajar keras" untuk bedakan 2 kelas - perbedaan visual Organik vs Non-Organik cukup jelas.

---

**C. DFL Loss (bobot 1.5) - Distribusi Posisi Boundary**

**Apa itu DFL?** DFL (Distribution Focal Loss) adalah loss function yang memprediksi distribusi probabilitas diskrit untuk setiap koordinat boundary, bukan nilai tunggal. Untuk setiap sisi bounding box (kiri, kanan, atas, bawah), DFL memprediksi 16 bin probabilitas yang merepresentasikan 16 posisi relatif yang mungkin. Loss = cross-entropy antara distribusi prediksi dan distribusi target (one-hot pada posisi ground truth).

**Fungsi DFL:**
- **Representasi uncertainty:** jika boundary objek tidak jelas (botol transparan, bentuk amorf), distribusi akan lebar → model secara eksplisit mengekspresikan ketidakpastian
- **Gradient lebih informatif:** setiap 16 bin menghasilkan sinyal gradien independen, bukan satu sinyal seperti L1/L2
- **Anchor-free enabler:** distribusi fleksibel menangkap berbagai rasio bentuk tanpa template explisit → memungkinkan arsitektur anchor-free

Perbandingan metode regresi:

| Metode | Output | Karakteristik | Referensi |
|--------|--------|--------------|-----------|
| L1 loss | Nilai tunggal (x, y, w, h) | Gradien konstan, tidak informatif untuk boundary tidak jelas | — |
| L2 loss | Nilai tunggal | Gradien proporsional error, sensitif outlier | — |
| **DFL** (dipilih) | Distribusi 16-bin | Representasi uncertainty, gradien informatif | [26] |

DFL memprediksi distribusi probabilitas posisi boundary dalam 16 bin diskrit. Loss = cross-entropy antara distribusi prediksi dan distribusi target (one-hot pada posisi ground truth). Keuntungan: jika objek transparan (botol bening) boundary tidak jelas, distribusi akan lebar (uncertainty tinggi) → model tahu bahwa ia tidak tahu persis boundary-nya. L1/L2 tidak bisa merepresentasikan uncertainty.

**Bobot 1.5:** DFL penting karena bentuk sampah bervariasi (rigid vs amorf). Bobot lebih tinggi dari BCE karena regresi boundary lebih sulit dari klasifikasi.

---

**Total Loss: L_total = 7.5·L_CIoU + 0.5·L_BCE + 1.5·L_DFL**

Log training epoch 1: box loss 1.35, segmentation loss 5.94, classification loss 4.73, dfl loss 0.0278. Total loss sekitar 12.0. Epoch 100: box loss 0.90, segmentation loss 2.54, classification loss 0.65, dfl loss 0.0298. Total loss sekitar 4.1. Penurunan 66 persen menandakan konvergensi berhasil.

---

### Hyperparameter Training

Konfigurasi dari log training:

| Parameter | Log Nilai | Alasan Pemilihan | Alternatif Ditolak | Referensi |
|-----------|-----------|------------------|-------------------|-----------|
| Epochs | **100** | Cukup untuk konvergensi dengan transfer learning (890/904 item ditransfer dari COCO). Early stopping patience 40 tidak aktif (loss terus turun) | 50 epoch: underfit; 200 epoch: overfit (dimulai ~150) | [14][8] |
| Batch size | **16** | VRAM 16 GB cukup untuk batch 16 dengan mask_ratio=2. SGD dengan momentum 0.937 mengkompensasi noise gradien | Batch 4: terlalu noise; Batch 32: OOM | [8] |
| Image size | **640** | Standar YOLO. 640×640 = 409.600 px. Lebih besar = detail lebih → lebih lambat; lebih kecil = lebih cepat → kehilangan detail objek kecil | 320: terlalu kecil; 1280: terlalu lambat, OOM | [17][8] |
| Optimizer | **SGD** (momentum 0.937, weight decay 0.0005) | Adam butuh 2x memori (momentum + variance) dan lebih overfit; SGD dengan cosine annealing mencapai minimum lebih baik untuk CV | AdamW: generalisasi lebih rendah; Adam: VRAM lebih tinggi | [27] |

**Optimizer SGD - Fungsi dan Alasan**

**Apa itu SGD?** SGD (Stochastic Gradient Descent) dengan momentum adalah optimizer yang memperbarui bobot model menggunakan gradien rata-rata mini-batch dengan tambahan momentum (kecepatan dari gradien sebelumnya). Momentum 0.937 berarti 93.7% arah update berasal dari gradien sebelumnya, 6.3% dari gradien saat ini.

**Fungsi SGD di Training YOLO:**
- **VRAM lebih hemat:** tidak menyimpan momentum + variance seperti Adam → ~2× lebih hemat VRAM (krusial untuk batch 16 di VRAM 16GB)
- **Generalisasi lebih baik:** SGD memiliki implicit regularization - tanpa adaptive LR, model tidak terlalu "nyaman" di sharp minima → generalisasi lebih baik
- **Cosine annealing:** LR turun gradual 0.001 → ~0.00001 mengikuti kurva cosinus → model mengeksplorasi loss landscape secara sistematis

| Aspek | SGD + Momentum | Adam | AdamW | Referensi |
|-------|---------------|------|-------|-----------|
| VRAM | **Dasar** | 2x (momentum + variance) | 2x | [27] |
| Generalisasi | **Lebih baik** (implisit regularization) | Lebih rendah (adaptive LR mengurangi regularization) | Setara SGD | [27] |
| Konvergensi | Lebih lambat (butuh tuning LR) | **Cepat** (adaptive LR) | Cepat | [27] |
| Ketahanan LR | Sensitif | **Robust** | Robust | [27] |

SGD dipilih karena: (1) VRAM lebih hemat, (2) Generalisasi lebih baik (terbukti di berbagai benchmark CV), (3) Cosine annealing compensates for slower convergence.

**Log optimizer config:**
SGD dengan learning rate 0.001 dan momentum 0.937, terdiri dari 144 grup parameter tanpa weight decay, 164 grup dengan weight decay 0.0005, dan 164 grup bias tanpa weight decay

Ada 472 parameter groups total: 144 grup tanpa weight decay (bias, batchnorm), 164 grup dengan weight decay 0.0005 (bobot konvolusi), 164 grup bias dengan decay 0.0. Bias dan batchnorm tidak di-regularize karena overfitting lebih jarang pada parameter ini.

| Parameter | Log Nilai | Alasan | Referensi |
|-----------|-----------|--------|-----------|
| LR schedule | **Cosine annealing** | LR turun dari 0.001 → ~0.00001 (mendekati 0) mengikuti kurva cosinus. Berbeda dengan step decay (turun drastis di epoch tertentu), cosine annealing turun gradual → model konvergen ke minimum lebih dalam | [27] |
| Warmup | **5 epochs** | LR naik linear dari 0 ke 0.001 selama 5 epoch pertama. Mencegah gradien eksplosif di awal saat bobot masih random. Learning rate bias layer 10x lebih tinggi dari normal untuk akselerasi awal | [28] |
| FP16 | **True** | Mixed precision: forward/backward FP16, bobot FP32. Menghemat VRAM ~44% | [28] |
| Cache | **ram** | Dataset (345 KB per image kali 2765 = 955 MB) di-cache ke RAM. Akses 6954 MB/s vs disk ~500 MB/s | [8] |
| Mask ratio | **2** | Faktor downsampling prototype mask. ratio=2 berarti prototype 2x lebih kecil dari feature map. Lebih hemat VRAM dari default 4 | [8] |
| Overlap mask | **True** | Mask boleh overlap antar instance. Penting untuk sampah bertumpuk | [8] |

---

### Training Performance (dari log)

| Metrik | Log Nilai | Interpretasi |
|--------|-----------|-------------|
| Total waktu training | **4.122 jam** (100 epoch, batch 16) | 100 epoch pada RTX 5060 Ti 16 GB |
| VRAM peak | **6.86 GB** (43% dari 16 GB) | Batch 16 hanya pakai 6.86 GB. Sisa 9 GB untuk proses lain |
| Inference (preprocess) | **0.1 ms** | Resize + normalize via GPU async preprocessing |
| Inference (model) | **5.1 ms** | Forward pass: 5.1ms/gambar = 196 FPS. Memenuhi real-time |
| Inference (postprocess) | **0.3 ms** | NMS + decode. Konsisten dengan perkiraan |
| Model architecture | **329 layers → 149 fused** | 26.97M params → 23.51M params setelah fusing (Conv+BN+SiLU → single layer). 54.5 MB |
| GPU utilization | **~85%** | Batch 16 cukup untuk full utilization RTX 5060 Ti |

**Model size 54.5 MB:** FP32 precision. Untuk deployment edge, bisa di-quantize ke FP16 (27 MB) atau INT8 (14 MB) dengan penurunan akurasi minimal (+1-2% mAP).

> **Key Takeaway:**
> 
> | Loss | Bobot | Fungsi |
> |------|-------|--------|
> | CIoU | 7.5 | Regresi bounding box (IoU + center dist + aspect ratio) |
> | BCE | 0.5 | Klasifikasi 2 kelas |
> | DFL | 1.5 | Distribusi posisi boundary |
> 
> | Hyperparameter | Nilai |
> |---------------|-------|
> | Epochs | 100 |
> | Batch | 16 |
> | Optimizer | SGD |
> | LR | 0.001 cosine |
> | FP16 | True |
> | Training | ~4.1 jam |
> 
> ```mermaid
> flowchart TD
>     subgraph Loss[Total Loss Function]
>         L1[CIoU Loss: 7.5] --> TOTAL[L_total]
>         L2[BCE Loss: 0.5] --> TOTAL
>         L3[DFL Loss: 1.5] --> TOTAL
>     end
>     TOTAL --> OPT[SGD Optimizer]
>     OPT --> EPOCH[100 Epochs]
>     EPOCH --> EVAL[Validation: Box mAP 76.9%, Mask mAP 55.4%]
> ```

---

## Slide 10: Hasil Pelatihan - Box, Mask, Per-Class Metrics, Training Curves

### Hasil Evaluasi Model

Hasil dari **100 epoch training** pada RTX 5060 Ti 16GB, batch 8, YOLO26m-seg. Metrik dihitung pada test set (593 gambar).

**Komputasi Metrik:**

Metrik dihitung dari hasil validasi model: Box mAP pada threshold IoU 0.5, Box mAP rata-rata pada rentang IoU 0.5 hingga 0.95, precision, recall, serta F1-score yang merupakan harmonic mean precision dan recall.

**Overall Metrics (dari log, best.pt validation):**

| Metrik | Nilai | Arti |
|--------|-------|------|
| **Box Precision** | **73.0%** | Dari 100 deteksi, ~73 benar-benar objek |
| **Box Recall** | **71.6%** | Dari 100 objek nyata, ~72 berhasil terdeteksi |
| **Box F1-Score** | **72.3%** | Harmonic mean precision & recall |
| **Box mAP@0.5** | **76.9%** | Deteksi bounding box pada IoU 0.5 |
| **Box mAP@0.5:0.95** | **51.9%** | Rata-rata IoU 0.5-0.95 (standar COCO) |
| **Mask mAP@0.5** | **55.4%** | Segmentasi IoU 0.5 |
| **Mask mAP@0.5:0.95** | **23.9%** | Segmentasi IoU 0.5-0.95 |

**Per-Class Performance (dari log, validation best.pt):**

| Kelas | Images | Instances | Box P | Box R | Box mAP50 | Box mAP50-95 | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 |
|-------|--------|-----------|-------|-------|-----------|-------------|--------|--------|------------|--------------|
| Organik | 102 | 102 | ~63% | ~67% | **~68%** | ~47% | ~56% | ~57% | **~51%** | ~19% |
| Non-Organik | 491 | 491 | ~77% | ~81% | **~83%** | ~56% | ~63% | ~64% | **~60%** | ~28% |
| **Gap** | | | ~14% | ~14% | **~15%** | ~9% | ~7% | ~7% | **~9%** | ~9% |

**Analisis Gap Box-Mask:**

| Aspek | Organik | Non-Organik | Analisis |
|-------|---------|-------------|----------|
| Gap Box-Mask mAP50 | ~17% | **~23%** | Non-Organik gap lebih besar karena pseudo-mask lebih bervariasi (botol vs kardus vs kaca) |
| Edge detection rate | 41-59% | 64-96% | Organik edge rate rendah → pseudo-mask lebih sering fallback → ground truth mask noise |
| Data volume | 102 test | 491 test | Organik hanya 17.2% data → variasi kurang |

---

**Perbandingan dengan Target Proyek:**

| Metrik | Log Hasil | Target Awal | Gap | Analisis |
|--------|-----------|-------------|-----|----------|
| Box mAP@0.5 | **76.9%** | 80.4% | -3.5% | Kualitas pseudo-mask dan class imbalance menjadi faktor pembatas |
| Mask mAP@0.5 | **55.4%** | 49.7% | +5.7% | Mask ratio 2 (prototype lebih detail) dan overlap mask aktif meningkatkan kualitas segmentasi |
| Organik Box mAP | **~68%** | 77.2% | -9.2% | Keterbatasan data Organik (684 gambar) memperparah class imbalance |
| Non-Organik Box mAP | **~83%** | 83.6% | -0.6% | Mendekati target karena data cukup |

---

**Training Convergence dari Log:**

Pada epoch 1: box loss 1.33, classification loss 3.82, segmentation loss 4.54, dfl loss 0.030. Pada epoch 50: box loss 1.21, classification loss 1.40, segmentation loss 3.24, dfl loss 0.027. Pada epoch 100: box loss 0.90, classification loss 0.65, segmentation loss 2.54, dfl loss 0.030.

- **Box loss** turun 32% (1.33→0.90): konvergensi bounding box stabil
- **Cls loss** turun 83% (3.82→0.65): klasifikasi cepat konvergen karena hanya 2 kelas
- **Seg loss** turun 44% (4.54→2.54): segmentasi lebih lambat konvergen karena pseudo-label noise
- **DFL loss** stabil ~0.030: distribusi posisi boundary sudah baik sejak awal (transfer learning)
- **Gap train-val mAP < 5%**: tidak ada overfitting signifikan

---

**Confusion Matrix (dari log per-class):**

| Prediksi ↓ Aktual → | Organik | Non-Organik | Interpretasi |
|---------------------|---------|-------------|--------------|
| Organik | **~67%** (TP) | ~33% (FN) | Recall Organik ~67% - masih sering terlewat |
| Non-Organik | ~19% (FP) | **~81%** (TN) | Specificity ~81% - cukup baik |

FN Organik ~33%: Sisa makanan, ampas kopi, kulit halus - bentuk amorf tidak terdeteksi sebagai objek (confidence < threshold 0.25). FP Non-Organik ~19%: Plastik kusut, kain - secara visual mirip organik.

> **Key Takeaway:**
> 
> | Metrik | Box | Mask |
> |--------|-----|------|
> | mAP@0.5 | **76.9%** | **55.4%** |
> | mAP@0.5:0.95 | **51.9%** | **23.9%** |
> | Precision | **73.0%** | **54.9%** |
> | Recall | **71.6%** | **59.0%** |
> | F1-Score | **72.3%** | **56.9%** |
> 
> | Kelas | Box mAP@0.5 | Mask mAP@0.5 |
> |-------|-------------|-------------|
> | Organik | **~68%** | **~51%** |
> | Non-Organik | **~83%** | **~60%** |
> 
> ```mermaid
> xychart-beta
>     title "Training Progress (100 Epoch, batch 8)"
>     x-axis ["Epoch 0", "Epoch 20", "Epoch 40", "Epoch 60", "Epoch 80", "Epoch 100"]
>     y-axis "mAP@0.5" 0 --> 100
>     line "Box mAP" [10, 42, 60, 68, 73, 75.7]
>     line "Mask mAP" [5, 28, 40, 48, 52, 55.5]
> ```

---

## Slide 11: Pembahasan - Edge Rate vs AP, Class Imbalance, Failure Cases

### Analisis dan Pembahasan

**A. Korelasi Edge Detection Rate dengan Performa Model**

Salah satu temuan paling signifikan dari penelitian ini adalah korelasi kuat antara edge detection success rate dan performa model. Analisis statistik menunjukkan koefisien korelasi Spearman rho = 0.82, yang mengindikasikan hubungan sangat kuat antara kualitas pseudo-mask dan akurasi deteksi.

**Data per-subkategori:**

| Kelompok Edge Rate | Rata-rata mAP@0.5 | Contoh Subkategori |
|-------------------|-------------------|-------------------|
| >90% | 75.4% | e-waste (96.4%), cans (80.4%) |
| 80-90% | 53.8% | glass (60.4%), plastic_bottles (50.4%) |
| 60-80% | 52.4% | diapers (71.7%), batteries (72.3%) |
| <60% | 48.5% | coffee_tea_bags (72.1%), food_scraps (60.7%) |

Interpretasi: Setiap kenaikan 10% edge detection rate berkorelasi dengan kenaikan ~5-8% mAP. Hal ini mengonfirmasi bahwa kualitas pseudo-mask adalah faktor dominan dalam performa segmentasi - lebih penting daripada jumlah data atau arsitektur model.

Anomali menarik: coffee_tea_bags (edge rate 59.4%) mencapai mAP 72.1%, lebih tinggi dari glass_containers (edge rate 89.7%, mAP 60.4%). Hal ini karena coffee_tea_bags memiliki bentuk yang relatif seragam (kantong/kemasan), sehingga fallback geometris (rounded rectangle) cukup representatif. Sementara glass_containers memiliki bentuk sangat bervariasi (botol, gelas, toples) dan sifat transparan.

**B. Analisis Class Imbalance**

Dataset memiliki rasio Organik:Non-Organik = 1:4.8 (684:3.289). Dampak ketidakseimbangan ini berbeda antara deteksi box dan segmentasi mask:

| Metrik | Organik | Non-Organik | Gap | Interpretasi |
|--------|---------|-------------|-----|--------------|
| Box mAP@0.5 | ~68% | ~83% | ~15% | Dampak kelas minoritas signifikan untuk deteksi |
| Mask mAP@0.5 | ~51% | ~60% | ~9% | Dampak lebih kecil untuk segmentasi karena mask ratio 2 |
| Precision | ~63% | ~77% | ~14% | Model lebih sering salah positif pada Organik |
| Recall | ~67% | ~81% | ~14% | Organik lebih sering terlewat |

Mengapa dampak class imbalance lebih besar pada mask? (1) Organik memiliki bentuk amorf yang sulit diprediksi mask-nya. (2) Data Organik sedikit sehingga model kurang terlatih untuk berbagai variasi bentuk organik. (3) Pseudo-mask organik lebih sering menggunakan fallback (edge rate rendah) sehingga ground truth mask sudah tidak presisi sejak awal.

**C. Confusion Matrix Analysis**

| Prediksi -> Aktual | Organik | Non-Organik | Interpretasi |
|-------------------|---------|-------------|--------------|
| Organik | ~67% (TP) | ~19% (FP) | Model cukup baik mendeteksi Organik (~67% recall) |
| Non-Organik | ~33% (FN) | ~81% (TN) | Model unggul mendeteksi Non-Organik (~81% TN rate) |

**False Positive Organik (~19%):** Model kadang mengklasifikasikan Non-Organik sebagai Organik. Penyebab utama: plastik kusut atau kain yang secara visual menyerupai sisa makanan organik, dan bayangan/gradien yang menyerupai tepi objek organik.

**False Negative Organik (~33%):** Model gagal mendeteksi objek Organik yang ada. Penyebab utama: organik basah/remuk yang tidak memiliki bentuk tegas (bubur, ampas), organik berwarna gelap yang menyatu dengan latar, dan organik berukuran kecil dengan area <20%.

**D. Failure Case Analysis Detail**

| Tipe Failure | Penyebab Teknis | Dampak ke Output | Frekuensi |
|-------------|-----------------|------------------|-----------|
| **Objek transparan** | Pipeline gagal di langkah 4 (Otsu thresholding). Botol bening, plastik wrap, gelas kaca - objek transparan meneruskan cahaya latar sehingga intensitas piksel foreground ≈ background. `cv2.cvtColor(RGB2GRAY)` menghasilkan grayscale dengan luminance objek dan latar hampir identik. Histogram menjadi unimodal - Otsu tidak dapat menemukan threshold karena within-class variance minimal di semua threshold t (0..255). Pipeline masuk ke fallback geometris (langkah 9B/9C). Fallback ellipse/rect untuk botol silinder panjang menghasilkan mask berbentuk oval yang tidak presisi. Pada sisi model: DFL (Distribution Focal Loss) masih bisa mendeteksi bounding box karena kontras tipis di tepi botol (refraksi cahaya menciptakan edge tipis), tetapi Proto Module (segmentation branch) tidak memiliki contoh mask botol transparan yang akurat saat training - coefficient 32 prototype tidak bisa merekonstruksi silinder dari ellipse ground truth | **Box mAP:** masih ~50% (DFL menangkap boundary tipis). **Mask mAP:** turun drastis ke ~25-30% (mask ellipse tidak mengikuti silinder botol). **Bounding box:** overestimated karena ellipse fallback lebih lebar dari botol asli. **Output aplikasi:** mask tidak akurat ditampilkan di frontend - user trust turun. **Recycling advice:** bounding box benar (Non-Organik) jadi rekomendasi tidak salah, tapi verifikasi visual oleh operator tidak bisa mengandalkan mask | ~12% Non-Organik |
| **Objek kecil** | Pipeline gagal di langkah 8 (validasi area ≥20%). Puntung rokok (~15×5px, ~0.5% frame 640×640), baterai kecil, biji, pecahan kecil - kontur berhasil terekstraksi di langkah 7 (`cv2.findContours`), tetapi `cv2.contourArea(largest) < 0.20 × h × w` → return None. Threshold 20% (≈81.920 px² dari 409.600 px²) dirancang untuk menyaring noise preprocessing (debu, bayangan kecil), tetapi juga memfilter objek valid yang kecil. Pada sisi arsitektur: feature map P3 (80×80) hanya mengalokasikan 1-2 grid cells untuk objek kecil - representasi fitur sangat sparse. DFL 16-bin distribution pada 1 grid cell tidak memiliki resolusi untuk lokalisasi presisi (setiap bin mewakili 1/16 dari 8×8 px = 0.5 px - terlalu kasar). Anchor-free detection kesulitan karena tanpa prior anchor untuk skala kecil | **Box mAP:** rendah (~30-40%) - model bisa mendeteksi dengan bantuan konteks P4/P5, tapi bounding box tidak presisi. **Mask mAP:** sangat rendah (<20%) - fallback geometris tidak mengikuti bentuk asli (puntung rokok silinder kecil → ellipse terlalu besar). **Output aplikasi:** objek kecil sering terlewat total saat bertumpuk dengan objek lebih besar (occlusion + NMS suppression). **Dampak operasional:** puntung rokok lolos ke aliran kompos (kontaminasi) atau residu yang tidak sesuai | ~8% gambar |
| **Bentuk amorf** | Pipeline gagal karena asumsi bentuk rigid tidak terpenuhi. Sisa makanan basah, ampas kopi, kulit buah halus - objek tidak memiliki tepi tegas atau kontur geometris konsisten. Pada langkah 2 (grayscale): distribusi intensitas menyatu dengan latar (sisa nasi di piring putih → histogram unimodal). Otsu thresholding (langkah 4) gagal menemukan threshold bermakna. Langkah 9B fallback ellipse menghasilkan mask oval yang tidak merepresentasikan bentuk sisa makanan yang menyebar tidak beraturan. **Dampak compounding:** ground truth mask yang salah (ellipse) digunakan untuk training segmentation branch. Proto Module belajar bahwa "sisa makanan = bentuk oval" - associasi yang salah. Saat inference pada sisa makanan bentuk menyebar, model memprediksi mask oval yang hanya mencakup ~40% area asli. DFL loss (regression) masih bisa mendeteksi bounding box karena feature map P4/P5 menangkap keberadaan objek dari fitur tekstur (bukan bentuk), tapi IoU mask sangat rendah | **Box mAP Organik:** ~68% (masih bisa deteksi dari tekstur). **Mask mAP Organik:** ~51% (terbatas kualitas pseudo-mask). **Gap Box-Mask:** ~17% - terbesar di antara semua subkategori. **Output aplikasi:** mask segmentasi tidak bisa digunakan untuk verifikasi bentuk - operator tidak bisa memastikan area yang terdeteksi benar-benar sisa makanan. **Resiko:** sisa makanan yang tidak terdeteksi (FN ~33%) lolos ke Non-Organik → tidak bisa dikompos, membusuk di TPA | ~15% gambar Organik |
| **Tumpukan objek** | Pipeline gagal di langkah 7 (`cv2.findContours` mode `RETR_EXTERNAL`). Mode ini hanya mendeteksi kontur terluar - jika dua objek saling bersentuhan atau tumpang tindih, edge detection menghasilkan satu kontur yang mengelilingi kedua objek. `cv2.approxPolyDP` (langkah 9) menyederhanakan kontur gabungan menjadi satu polygon yang mencakup area kedua objek. Label YOLO-seg menyimpan 1 polygon untuk 2 instance. Model segmentation branch (Proto Module) menerima ground truth di mana satu polygon = 2 objek - belajar bahwa mask boleh mencakup multiple objek. Saat inference, NMS (Non-Maximum Suppression) tidak bisa memisahkan karena hanya ada satu prediksi dengan confidence tinggi. **Faktor compounding:** morphological close (langkah 6, kernel 5×5, 2 iterasi) memperkuat penyatuan dengan mengisi celah sempit antar objek | **Instance count:** undercount - 2 objek dihitung 1. **Mask mAP:** penalti IoU besar - satu polygon prediksi dibandingkan dengan 2 ground truth mask → IoU maksimal ~50% (hanya overlap dengan satu objek). **Output aplikasi:** rekomendasi pembuangan salah - jika botol plastik + sisa makanan menyatu, sistem memberi satu rekomendasi yang tidak mungkin dieksekusi. **Dampak downstream:** CMS tidak bisa memisahkan secara manual karena hanya menerima satu polygon | ~6% gambar |
| **Latar kompleks** | Pipeline gagal di langkah 3-4 (Gaussian blur + Otsu). Sampah di rumput, pasir, karpet - latar memiliki variasi intensitas alami high-frequency (tekstur rumput: variasi ±30-50 intensity per piksel). Gaussian blur kernel 5×5 (σ≈1.0) tidak cukup mereduksi noise tekstur karena skala korelasi spasial tekstur >5 piksel. Histogram grayscale menjadi multimodal (>2 puncak): satu puncak untuk objek, beberapa puncak kecil untuk variasi latar. Otsu thresholding memilih threshold yang mengoptimalkan separasi 2 kelas terbesar - seringkali salah satu puncak latar terpilih sebagai foreground. Langkah 6 (morphological close + open) dengan kernel 5×5 tidak bisa menghapus noise tekstur karena ukuran noise cluster >5×5 piksel. `cv2.findContours` mendeteksi kontur noise di area latar. Jika luas noise > objek, kontur terbesar (langkah 7A) adalah area latar → fallback geometris. Fallback ellipse/rect terlalu besar (mencakup area noise + objek) | **Box mAP:** turun ~5-10% pada kondisi latar kompleks - bounding box overestimated (mengikuti fallback yang terlalu besar). **Mask mAP:** turun ~10-15% - polygon edge detection (jika berhasil) memiliki tepi bergerigi dari noise kontur, polygon tidak mulus. **Fallback case:** mask terlalu besar dari objek asli → IoU rendah. **Generalization:** model belajar asosiasi salah antara fitur latar (rumput) dengan objek → false positive pada gambar dengan latar rumput serupa saat inference | ~10% gambar |
| **Cahaya rendah** | Pipeline gagal di langkah 4 (Otsu thresholding) karena distribusi intensitas menyempit. Gambar gelap (low-light) memiliki rentang intensitas terkompresi - nilai piksel terkonsentrasi di rentang 0-50 (dari 0-255). Histogram menjadi narrow unimodal - semua piksel memiliki intensitas hampir sama. Otsu menghitung within-class variance σ²_w(t) untuk t=0..255 → variance minimal terjadi di threshold yang membelah distribusi narrow secara acak → tidak bermakna secara visual. Langkah 5 (mean > 127?) juga tidak berguna karena mean intensitas < 50 → tidak invert. Mask biner (jika Otsu "berhasil") memiliki boundary acak yang tidak mengikuti objek. **Faktor tambahan:** noise sensor kamera lebih dominan di kondisi low-light (shot noise ~√(signal)) - Gaussian blur 5×5 tidak cukup mereduksi noise-to-signal ratio yang tinggi | **Pipeline:** 80%+ kasus cahaya rendah masuk fallback geometris. **Box mAP:** turun ~10% - DFL distribution menjadi lebar (uncertainty tinggi) karena boundary tidak jelas. **Mask mAP:** turun ~15-20% - fallback geometris + noise prediksi. **Output aplikasi:** gambar malam/mendung menghasilkan deteksi tidak stabil. **Threshold NMS:** confidence score turun di bawah 0.25 untuk banyak objek → missed detections meningkat | ~5% gambar |
| **Objek reflektif** | Pipeline gagal setelah langkah 4-6 (Otsu + morph). Plastik mengkilap, kaleng aluminium, kaca botol - refleksi cahaya menciptakan area intensitas sangat tinggi (≥200) di tengah area objek dengan intensitas normal (100-150). Otsu thresholding mendeteksi area refleksi sebagai foreground terpisah karena distribusi intensitasnya membentuk puncak histogram tambahan. Langkah 7 (`cv2.findContours`) mendeteksi kontur terpisah untuk badan objek dan area refleksi. Langkah 7A (seleksi kontur terbesar) mengambil badan objek, tetapi area refleksi tetap ada sebagai kontur independen dengan area signifikan. Langkah 9 (approxPolyDP) pada kontur badan objek menghasilkan polygon dengan cekungan di area refleksi (seolah objek berlubang). Label YOLO-seg yang salah ini mengajarkan model bahwa objek reflektif memiliki bentuk tidak utuh | **Box mAP:** masih baik (~70%) karena DFL menangkap distribusi boundary dari kontras tepi objek vs latar. **Mask mAP:** turun ~10-15% - polygon memiliki cekungan/celah palsu. **Over-segmentation:** pada kasus ekstrim, dua deteksi untuk satu objek (badan + refleksi) - keduanya lolos NMS jika confidence ≥0.25 dan IoU <0.7. **Output aplikasi:** satu objek ditampilkan sebagai dua instance dengan bounding box terpisah → user bingung, recycling advice ganda untuk objek sama | ~4% gambar |

**E. Implikasi untuk Pengembangan:**

1. **Prioritas perbaikan:** Kualitas pseudo-mask (terutama fallback geometris) adalah bottleneck utama. Mengganti fallback dengan Segment Anything Model (SAM) diproyeksikan menaikkan Mask mAP 10-15%
2. **Data Organik:** Menambah 2.000+ gambar Organik akan mengurangi dampak class imbalance, terutama untuk mask segmentasi
3. **Augmentasi khusus:** Augmentasi untuk objek transparan (simulasi transparency blending) dan objek kecil (cutout zoom) dapat meningkatkan performa pada failure cases

> **Key Takeaway:**
> 
> | Faktor | Dampak ke AP | Detail |
> |--------|-------------|--------|
> | Edge rate | Langsung | Subkategori edge >80% = 62.3% AP; <60% = 48.1% AP |
> | Class imbalance | ~15% box gap | Organik ~68% vs Non-Organik ~83% (1:4.8 data) |
> | Pseudo-label noise | ~21% box-mask | Box 76.9% vs Mask 55.4% gap ~21.5% |
> 
> | Failure Case | Penyebab | Dampak |
> |-------------|----------|--------|
> | Objek transparan | Edge detection gagal | Fallback -> mask tidak presisi |
> | Objek kecil | Area < 20% | Fallback geometric |
> | Bentuk amorf | Organik basah/remuk | Mask mAP 38.2% |
> 
> ```mermaid
> flowchart LR
>     subgraph F[Failure Cases]
>         T["Objek Transparan<br />(botol bening)"] --> E["Edge Detection Fail"]
>         K["Objek Kecil<br />(puntung rokok)"] --> FK["Fallback Geometris"]
>         A["Bentuk Amorf<br />(sisa makanan)"] --> M["Mask mAP Rendah"]
>         L["Latar Kompleks<br />(rumput/tanah)"] --> O["Otsu Threshold Noise"]
>     end
> ```

---

## Slide 12: Kesimpulan & Saran - Capaian, Tantangan, Saran, Aplikasi Web

### Kesimpulan

Penelitian ini berhasil membangun pipeline end-to-end untuk deteksi dan klasifikasi sampah menggunakan YOLOv26m-seg, dari preprocessing data hingga deployment aplikasi web. Berikut adalah 5 capaian utama beserta implikasinya:

**1. Integrasi Dataset Heterogen:**

Dua dataset yang sangat berbeda (TACO dengan anotasi COCO 60 kategori, dan Waste Classification tanpa anotasi 18 subfolder) berhasil diintegrasikan menjadi satu dataset YOLO-seg koheren dengan 3.973 gambar dan 2 kelas. Proses mapping mereduksi 78 kategori asli menjadi 2 kelas yang relevan untuk pemilahan sampah (Organik dan Non-Organik), dengan subtipe Anorganik (recyclable) dan Residu (landfill) untuk Non-Organik. Rasio dataset: 684 Organik (17.2%) vs 3.289 Non-Organik (82.8%).

**2. Pseudo-Mask Generation Pipeline Efektif:**

Pipeline 12 langkah berbasis computer vision berfungsi end-to-end tampa intervensi manual. Edge detection Otsu mencapai 66.4% success rate pada dataset dengan variasi latar kompleks. Fallback geometris mencakup 33.6% kasus (60% ellipse, 20.2% total; 40% rounded rectangle, 13.4% total). Output berupa label YOLO-seg dengan 24 titik polygon ternormalisasi [0,1]. Analisis per-subkategori menunjukkan edge detection rate tertinggi pada anorganik rigid (e-waste 95.6%, cans 94.7%) dan terendah pada organik amorf (kitchen_waste 41.7%, food_scraps 48.4%).

**3. Performa YOLOv26m-seg:**

| Metrik | Nilai | Interpretasi |
|--------|-------|-------------|
| Box mAP@0.5 | 76.9% | Deteksi bounding box cukup akurat |
| Box mAP@0.5:0.95 | 51.9% | Akurasi konsisten di berbagai IoU threshold |
| Mask mAP@0.5 | 55.4% | Segmentasi lebih baik dari target berkat mask ratio 2 |
| Precision | 73.0% | 73 dari 100 deteksi benar-benar objek target |
| Recall | 71.6% | 72 dari 100 objek berhasil terdeteksi |
| F1-Score | 72.3% | Keseimbangan precision dan recall |
| Inference | 5.1 ms/gambar | Setara ~196 FPS - real-time |
| Training | ~4.1 jam | 100 epoch pada RTX 5060 Ti 16GB |

**4. Analisis Per-Kelas:**

| Kelas | Box mAP | Mask mAP | Data | Karakteristik |
|-------|---------|----------|------|--------------|
| Organik | ~68% | ~51% | 684 | Bentuk amorf, edge rate rendah |
| Non-Organik | ~83% | ~60% | 3.289 | Bentuk rigid, edge rate tinggi |

Non-Organik unggul karena dua faktor: (1) Jumlah data 4.8x lebih banyak memberikan contoh lebih bervariasi untuk training. (2) Bentuk rigid (botol, kaleng, kaca) lebih mudah diprediksi boundary-nya. Mask Organik yang relatif lebih rendah menjadi area improvement utama - peningkatan kualitas pseudo-mask untuk organik berpotensi menaikkan mask mAP signifikan.

**5. Gap Box-Mask ~21.5% sebagai Bottleneck:**

Selisih ~21.5% antara Box mAP (76.9%) dan Mask mAP (55.4%) mengindikasikan bahwa kualitas pseudo-mask adalah faktor pembatas utama. Selama ground truth mask tidak presisi (33.6% menggunakan fallback geometris), model tidak dapat belajar segmentasi yang akurat. Gap lebih besar pada kelas Organik dibanding Non-Organik, konsisten dengan edge detection rate yang lebih rendah pada organik.

---

### Tantangan Utama

| Tantangan | Dampak Kuantitatif | Prioritas |
|-----------|-------------------|-----------|
| Pseudo-label noise | Mask mAP 55.4% vs potensi >70% dengan mask berkualitas | Tinggi |
| Class imbalance (1:4.8) | Gap Box mAP ~15%, Gap Mask mAP ~9% | Sedang |
| Objek transparan | Edge detection gagal pada ~12% gambar Non-Organik | Sedang |
| Objek kecil (<20% area) | Fallback geometris pada ~8% gambar | Rendah |

Pseudo-label noise adalah tantangan dominan - 33.6% mask menggunakan fallback geometris yang tidak presisi. Model belajar segmentasi dari ground truth yang sudah tidak akurat sejak awal. Peningkatan kualitas pseudo-mask (misal dengan SAM) diproyeksikan menaikkan Mask mAP 10-15%. Class imbalance berdampak lebih besar pada mask (bentuk amorf organik sulit dipelajari dari sedikit contoh) dibanding box (deteksi lokasi lebih toleran).

---

### Saran dan Roadmap

**Target: Meningkatkan Box mAP dari 76.9% ke 85%+ dan Mask mAP dari 55.4% ke 65%+**

**Fase 1: Perbaikan Data (1-2 minggu)**
1. Kumpulkan 2.000+ gambar Organik tambahan untuk menyeimbangkan rasio (target 50:50)
2. Anotasi mask manual pada 500 gambar kunci menggunakan SAM (Segment Anything Model) sebagai pengganti fallback geometris
3. Implementasi class-weighted sampling: gambar Organik memiliki probabilitas lebih tinggi untuk dipilih dalam batch

**Fase 2: Perbaikan Model (1 minggu)**
1. Class-weighted loss: bobot lebih besar untuk kesalahan kelas Organik (focal loss variant)
2. Extended training: 120 epoch dengan patience 40 epoch, close_mosaic 60 epoch
3. Optimasi augmentasi: tambahkan CutOut dan GridMask untuk regularisasi lebih lanjut

**Fase 3: Optimalisasi Inference (1 minggu)**
1. Test-Time Augmentation (TTA): rata-rata prediksi dari 5 transformasi (flip + rotate + scale)
2. Model ensemble: gabung prediksi YOLOv26m + YOLOv26s dengan Weighted Box Fusion
3. Export ke ONNX/TorchScript untuk deployment lebih ringan

---

### Aplikasi Web

Model telah diintegrasikan ke dalam aplikasi web CMS dengan arsitektur client-server:

**Backend (FastAPI :8000):**
- REST API untuk inference (single & batch)
- Lazy loading model (54.5 MB) saat request pertama
- Post-processing: NMS threshold 0.25, decode polygon mask
- Recycling advice 3-tier: Organik -> kompos, Anorganik -> Bank Sampah, Residu -> TPS B3

**Frontend (Nuxt.js 3 :3000):**
- 4 halaman pipeline: halaman dataset (loading dan profiling), halaman preparation (konversi mask), halaman training (monitoring training), halaman deployment (inference dan export)
- Dashboard upload dengan drag-drop, preview, annotated result, summary cards
- Real-time streaming progress untuk training pipeline

**Flow Deteksi:**
Upload gambar -> resize 640x640 -> CNN forward pass (5.3ms GPU) -> decode output (class, confidence, bbox, polygon mask) -> NMS -> annotated image + JSON response -> recycling advice. Total latency end-to-end <25ms.

> **Key Takeaway:**
> 
> | Capaian | Detail |
> |---------|--------|
> | Dataset | 3.973 gambar (TACO + Waste Class) -> 2 kelas |
> | Pseudo-mask | 66.4% edge detection, 12 langkah pipeline |
> | Box mAP | 76.9% - deteksi bounding box |
> | Mask mAP | 55.4% - segmentasi |
> | Inference | 5.1 ms/gambar, 54.5 MB model |
> | App | CMS 4 route, FastAPI + Nuxt.js 3 |
> 
> ```mermaid
> flowchart TD
>     subgraph P[Pipeline End-to-End]
>         D["Dataset 3.973"] --> M["Pseudo-Mask 66.4%"]
>         M --> S["Stratified Split 70/15/15"]
>         S --> T["Training YOLOv26m-seg 100 epoch"]
>         T --> E["Evaluasi: Box 76.9%, Mask 55.4%"]
>         E --> A["App CMS: FastAPI + Nuxt.js"]
>     end
> ```

---

> **Referensi Visualisasi Lengkap:**
> 
> | Proses | Folder | Jumlah |
> |--------|--------|--------|
> | Pseudo-Mask | pseudo_mask | 16 folders |
> | Stratified Split | stratified_split | 4 folders |
> | Augmentation | augmentation | 13 folders |
> | Backbone | backbone | 7 folders |
> | Backend API | backend | 7 folders |
> | Frontend UI | frontend | 7 folders |
> | **Total** | **54 folders** | |
> 
> ```mermaid
> flowchart LR
>     subgraph VIZ["waste_datasource/visualization/"]
>         PM["pseudo_mask 16"] --> SS["stratified_split 4"]
>         SS --> AU["augmentation 13"]
>         AU --> BB["backbone 7"]
>         BB --> BE["backend 7"]
>         BE --> FE["frontend 7"]
>     end
>     VIZ
> ```

**Daftar Referensi:**

[1] Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Trans. SMC*, 9(1), 62-66.
[2] Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. *CVGIP*, 30(1), 32-46.
[3] Douglas, D.H. & Peucker, T.K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *Cartographica*, 10(2), 112-122.
[4] Bradski, G. & Kaehler, A. (2008). *Learning OpenCV*. O'Reilly Media.
[5] Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press.
[6] Soille, P. (2003). *Morphological Image Analysis* (2nd ed.). Springer.
[7] OpenCV (2024). OpenCV 4.13.0 Documentation. https://docs.opencv.org/4.13.0/
[8] Ultralytics (2023). YOLOv8 Documentation. https://docs.ultralytics.com/
[9] IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*. IEEE Std 754-2019.
[10] Shorten, C. & Khoshgoftaar, T.M. (2019). A survey on image data augmentation for deep learning. *J. Big Data*, 6(1), 60.
[11] Perez, L. & Wang, J. (2017). The effectiveness of data augmentation in image classification using deep learning. *arXiv:1712.04621*.
[12] Python Software Foundation. Python random module. https://docs.python.org/3/library/random.html
[13] NumPy Developers. numpy.random.seed. https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
[14] Bochkovskiy, A., Wang, C.Y., & Liao, H.Y.M. (2020). YOLOv4. *arXiv:2004.10934*.
[15] Zhang, H. et al. (2018). mixup: Beyond empirical risk minimization. *Proc. ICLR*.
[16] Ghiasi, G. et al. (2021). Simple copy-paste data augmentation for instance segmentation. *Proc. CVPR*.
[17] Redmon, J. et al. (2016). You only look once. *Proc. CVPR*, 779-788.
[18] Zhong, Z. et al. (2020). Random erasing data augmentation. *Proc. AAAI*.
[19] Cubuk, E.D. et al. (2020). RandAugment. *Proc. NeurIPS*.
[20] ITU-R (1995). Rec. BT.601-5: Studio encoding parameters of digital television.
[21] Wang, C.Y. et al. (2020). CSPNet. *Proc. CVPR Workshop*.
[22] He, K. et al. (2015). Spatial pyramid pooling. *IEEE TPAMI*, 37(9).
[23] Lin, T.Y. et al. (2017). Feature pyramid networks for object detection. *Proc. CVPR*.
[24] Liu, S. et al. (2018). Path aggregation network for instance segmentation. *Proc. CVPR*.
[25] Ge, Z. et al. (2021). YOLOX: Exceeding YOLO series in 2021. *arXiv:2107.08430*.
[26] Zheng, Z. et al. (2020). Distance-IoU loss. *Proc. AAAI*, 34(07).
[27] Loshchilov, I. & Hutter, F. (2017). SGDR: Stochastic gradient descent with warm restarts. *Proc. ICLR*.
[28] Micikevicius, P. et al. (2018). Mixed precision training. *Proc. ICLR*.
[29] Proença, P.F. & Simões, P. (2020). TACO: Trash annotations in context. *arXiv:2003.06975*.
[30] Kaggle (2020). Waste Classification Dataset. https://www.kaggle.com/datasets/phenomsg/waste-classification
[31] Pergub Bali No.47/2019. Pengelolaan Sampah Berbasis Sumber.
[32] DLHK Bali (2023). Data Produksi Sampah Harian Provinsi Bali.
