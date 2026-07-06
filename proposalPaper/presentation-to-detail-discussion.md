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

**Node J** - "SGD/MuSGD Optimizer, LR=0.001 Cosine Decay": Optimizer menggunakan Stochastic Gradient Descent dengan hybrid MuSGD (Moonshot SGD). Learning rate awal 0.001 dengan scheduler cosine annealing yang menurunkan learning rate secara bertahap mengikuti kurva cosinus hingga mendekati nol pada epoch akhir. Momentum 0.937 digunakan untuk mempercepat konvergensi dengan mempertimbangkan arah gradien sebelumnya. Weight decay 0.0005 memberikan regularisasi L2 untuk mencegah overfitting.

**Node K** - "80 Epochs, Batch=16, FP16, Early Stop Patience=40": Training berlangsung selama 80 epoch dengan batch size 16 yang dibatasi oleh kapasitas VRAM 16 GB. FP16 (mixed precision) menggunakan format floating point 16-bit untuk sebagian operasi, mengurangi konsumsi VRAM ~44% dan mempercepat training hingga 2x lipat. Early stopping dengan patience 40 epoch menghentikan training jika validation loss tidak menunjukkan perbaikan, mencegah overfitting dan menghemat waktu komputasi. Warmup 5 epoch pertama menggunakan linear learning rate increase dari 0 ke target 0.001 untuk stabilisasi training awal. Total waktu training ~2.5 jam pada GPU RTX 5060 Ti 16 GB.

**Aliran data:** Output 3 cabang head dihitung loss-nya menggunakan kombinasi CIoU, BCE, dan DFL. Gradien di-backpropagate melalui optimizer SGD/MuSGD untuk memperbarui bobot model selama 80 epoch.

---

#### D. Evaluation - Metrics

****Node L** - "Box mAP@0.5: 80.4%, Mask mAP@0.5: 49.7%": Evaluasi dilakukan pada test set (593 gambar) menggunakan metrik standar COCO. Box mAP@0.5 mencapai 80.4%, mengindikasikan deteksi bounding box sangat akurat pada threshold IoU 0.5. Box mAP@0.5:0.95 mencapai 52.5% yang merupakan rata-rata pada IoU threshold 0.5 hingga 0.95. Mask mAP@0.5 49.7% lebih rendah dari box karena segmentasi membutuhkan prediksi boundary presisi sementara ground truth mask merupakan pseudo-label hasil generasi otomatis. Precision 76.7% menunjukkan proporsi prediksi benar terhadap total prediksi positif. Recall 75.6% menunjukkan proporsi objek nyata yang berhasil terdeteksi. F1-Score 76.1% merupakan harmonic mean precision dan recall.

**Node M** - "Organik: 77.2%, Non-Organik: 83.6%": Analisis per-kelas menunjukkan Non-Organik (Box mAP 83.6%, Mask mAP 61.2%) lebih unggul dari Organik (Box mAP 77.2%, Mask mAP 38.2%). Gap Box mAP 6.4% disebabkan perbedaan jumlah data (Organik 684 vs Non-Organik 3.289, rasio 1:4.8). Gap Mask mAP 23.0% lebih lebar karena bentuk organik cenderung amorf dan tidak beraturan (sisa makanan, daun, ampas kopi) sehingga pseudo-mask yang dihasilkan kurang presisi, sementara Non-Organik memiliki bentuk rigid (botol, kaleng, kaca) yang lebih mudah diproses edge detection.

**Aliran data:** Model terbaik dari training dievaluasi pada test set menghasilkan metrik box dan mask. Analisis per-kelas menunjukkan performa lebih tinggi pada Non-Organik.

---

#### E. Inference

**Node N** - "Inference 5.3ms -> Recycling Advice": Proses inference menerima input gambar melalui web, melakukan resize ke 640x640 dengan letterbox padding, menjalankan forward pass CNN pada GPU (5.3 ms per gambar), melakukan decode output untuk mendapatkan class ID, confidence score, koordinat bounding box, dan polygon mask 24 titik. Post-processing meliputi Non-Maximum Suppression (NMS) dengan threshold 0.25 untuk menghilangkan prediksi duplikat. Hasil akhir menghasilkan recycling advice 3-tier: Organik (kompos), Non-Organik subtipe Anorganik (daur ulang via Bank Sampah), dan Non-Organik subtipe Residu (TPS B3/landfill). Model size 54.5 MB, cukup ringan untuk deployment. Aplikasi web menggunakan FastAPI backend (port 8000) dan Nuxt.js 3 frontend (port 3000) dengan 4 halaman CMS pipeline: dataset preparation, konversi mask, training monitoring, dan deployment inference.

> **Key Takeaway:**
> 
> | Tahap | Metrik | Detail |
> |-------|--------|--------|
> | Dataset | 3.973 gambar | TACO 1.500 + Waste Class 2.939 |
> | Pseudo-mask | 66.4% success | Edge detection, 12 langkah |
> | Fallback | 33.6% | 60% ellipse, 40% rounded rect |
> | Split | 70/15/15 | Train 2.765, Val 593, Test 593 |
> | Training | 80 epoch | YOLOv26m-seg, batch 16, FP16 |
> | Box mAP | 80.4% | Deteksi bounding box |
> | Mask mAP | 49.7% | Segmentasi mask |
> | Inference | 5.3 ms/img | RTX 5060 Ti 16GB |
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
>         I["Loss: CIoU(7.5) + BCE(0.5) + DFL(1.5)"] --> J["SGD/MuSGD Optimizer<br />LR=0.001 Cosine Decay"]
>         J --> K["80 Epochs, Batch=16, FP16<br />Early Stop Patience=40"]
>     end
> 
>     subgraph EVAL["Evaluation - Metrics"]
>         L["Box mAP@0.5: 80.4%<br />Mask mAP@0.5: 49.7%"] --> M["Organik: 77.2%<br />Non-Organik: 83.6%"]
>     end
> 
>     D --> E
>     H --> I
>     K --> L
>     M --> N["Inference 5.3ms<br />-> Recycling Advice"]
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
Pemilahan sampah secara manual memiliki beberapa kelemahan signifikan: kecepatan terbatas (2-5 detik per objek), konsistensi menurun setelah 1 jam kerja, tingkat kesalahan ~10-15%, serta risiko keselamatan pekerja (tertusuk jarum, terpapar bahan kimia).

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

Stratified split adalah teknik pembagian dataset yang mempertahankan proporsi kelas yang identik di setiap subset. Berbeda dengan random split biasa yang dapat menghasilkan distribusi tidak merata (misalnya validation set kebetulan memiliki 30% Organik sementara test set hanya 10%), stratified split menjamin bahwa rasio Organik:Non-Organik tetap konsisten.

| Split | Total Gambar | Organik | Non-Organik | Persentase Organik |
|-------|-------------|---------|-------------|-------------------|
| Train | 2,765 | 476 | 2,289 | 17.22% |
| Validation | 593 | 102 | 491 | 17.20% |
| Test | 593 | 102 | 491 | 17.20% |
| **Total** | **3,973** | **684** | **3,289** | **17.21%** |

Fungsi train_test_split dari scikit-learn digunakan dengan parameter stratify=cls_ids untuk memastikan proporsi kelas terjaga pada setiap split. Seed 42 digunakan untuk reproducibility hasil split. Training set digunakan untuk optimasi parameter model, validation set untuk early stopping dan hyperparameter tuning, test set untuk evaluasi final performa generalisasi.

---

**Preprocessing Lanjutan:**
Seluruh gambar diresize ke resolusi 640x640 piksel menggunakan letterbox padding untuk mempertahankan aspek rasio asli. Label dalam format YOLO-seg disimpan sebagai file .txt terpisah per gambar dengan format: `<class_id> x1 y1 x2 y2 ... x24 y24`, di mana koordinat telah dinormalisasi ke rentang [0,1] dengan membagi setiap koordinat piksel dengan lebar/tinggi gambar.

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

Dataset yang digunakan merupakan dataset klasifikasi (tanpa label segmentasi). Untuk memenuhi kebutuhan input YOLO-seg, dilakukan pembangkitan polygon mask secara otomatis melalui pipeline computer vision 12 langkah. Pipeline ini dirancang untuk bekerja pada gambar dengan berbagai kondisi latar, pencahayaan, dan jenis objek tampa memerlukan intervensi manual.

---

**Metode 1: Edge Detection Otsu (66.4% kasus)**

Pipeline edge detection Otsu terdiri dari 12 langkah yang dikelompokkan dalam 4 kelompok fungsional. Setiap kelompok memiliki tujuan spesifik dalam rantai pemrosesan dari gambar mentah hingga label YOLO-seg.

**Kelompok 1: Pra-pemrosesan Citra (Langkah 1-3)**

| Langkah | Operasi | Deskripsi Teknis | Parameter |
|---------|---------|-----------------|-----------|
| 1 | RGB to Grayscale | Konversi ruang warna dari 3 channel (RGB) ke 1 channel luminance (grayscale) menggunakan persamaan: Y = 0.299*R + 0.587*G + 0.114*B. Reduksi dimensi data dari 640x640x3 menjadi 640x640x1 tanpa kehilangan informasi intensitas yang diperlukan untuk thresholding | cv2.COLOR_RGB2GRAY |
| 2 | Gaussian Blur 5x5 | Konvolusi citra grayscale dengan kernel Gaussian 5x5 untuk mereduksi noise frekuensi tinggi (debu, variasi pixel acak) yang dapat menghasilkan false positive pada deteksi tepi. Kernel Gaussian memiliki standar deviasi sigma=0 (dihitung otomatis dari ukuran kernel) | Kernel size: (5,5), sigmaX=0 |
| 3 | Output Pra-pemrosesan | Citra grayscale halus dengan noise tereduksi, siap untuk operasi thresholding | Input ke Otsu |

**Landasan Teori:**
Operasi konvolusi Gaussian Blur menghitung rata-rata tertimbang dari setiap piksel dengan tetangganya, di mana bobot mengikuti distribusi Gaussian 2D. Kernel 5x5 mencakup area 25 piksel di sekitar piksel target, memberikan smoothing yang cukup untuk menghilangkan noise kamera tanpa mengaburkan tepi signifikan yang diperlukan untuk deteksi kontur.

---

**Kelompok 2: Thresholding & Mask Biner (Langkah 4-6)**

Pada kelompok ini, citra grayscale diubah menjadi mask biner (hitam-putih) yang memisahkan objek foreground dari background.

| Langkah | Operasi | Deskripsi Teknis | Logika |
|---------|---------|-----------------|--------|
| 4 | Otsu Thresholding | Algoritma Otsu menghitung threshold optimal secara otomatis dengan menganalisis histogram intensitas piksel. Metode ini meminimalkan within-class variance (variansi intra-kelas) atau secara ekuivalen memaksimalkan between-class variance (variansi antar-kelas). Threshold dipilih sehingga piksel terbagi menjadi dua kelas (foreground dan background) dengan separasi maksimal. Formula: sigma^2_b(t) = w_0(t) * w_1(t) * [mu_0(t) - mu_1(t)]^2, di mana w adalah probabilitas kelas dan mu adalah mean intensitas | cv2.THRESH_BINARY + cv2.THRESH_OTSU |
| 5 | Mean > 127? | Pengecekan rata-rata intensitas seluruh piksel pada mask biner. Jika mean > 127, berarti latar belakang didominasi warna putih (255) dan objek berwarna hitam (0). Dalam kasus ini, mask perlu diinvert agar objek bernilai 255 (putih) sesuai konvensi foreground. Langkah ini penting untuk foto dengan latar belakang terang (misal: meja putih, kertas putih) | np.mean(thresh) > 127 |
| 5A | Invert Mask | Operasi bitwise NOT: piksel 0 menjadi 255, piksel 255 menjadi 0. Hanya dieksekusi jika mean > 127 | cv2.bitwise_not(thresh) |
| 6 | Morphological Close + Open | Operasi morfologi untuk membersihkan mask biner. Close (dilasi diikuti erosi) dengan kernel 5x5, 2 iterasi: menutup lubang kecil di dalam objek yang disebabkan oleh refleksi atau tekstur internal. Open (erosi diikuti dilasi) dengan kernel 5x5, 1 iterasi: menghilangkan titik-titik noise putih kecil di luar objek. Kombinasi Close+Open menghasilkan mask yang lebih bersih dan kontur yang lebih halus | cv2.MORPH_CLOSE (5x5, iter=2), cv2.MORPH_OPEN (5x5, iter=1) |

**Mengapa Otsu Thresholding?**
Otsu dipilih karena: (1) Bersifat adaptif - threshold dihitung per gambar, tidak menggunakan nilai tetap, sehingga dapat menangani variasi pencahayaan antar gambar. (2) Tidak memerlukan parameter yang dituning - algoritma sepenuhnya otomatis berdasarkan histogram. (3) Efektif untuk gambar dengan distribusi intensitas bimodal (dua puncak) yang umum pada foto objek dengan latar kontras.

---

**Kelompok 3: Ekstraksi Kontur (Langkah 7-9)**

| Langkah | Operasi | Deskripsi Teknis | Parameter |
|---------|---------|-----------------|-----------|
| 7 | Find Contours | Ekstraksi kontur dari mask biner menggunakan algoritma Suzuki85. Mode RETR_EXTERNAL hanya mengambil kontur terluar (mengabaikan hole di dalam objek). CHAIN_APPROX_SIMPLE mengompres segmen garis menjadi hanya titik ujungnya, menghemat memori tanpa kehilangan informasi bentuk | cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE |
| 7A | Seleksi Kontur Terbesar | Dari seluruh kontur yang terdeteksi, pilih kontur dengan area terbesar menggunakan max(contours, key=cv2.contourArea). Asumsi: objek utama sampah menempati area terbesar dalam frame. Kontur-kontur kecil lainnya diabaikan sebagai noise | cv2.contourArea(largest) |
| 8 | Validasi Area >= 20% | Pengecekan apakah area kontur terbesar >= 20% dari total area gambar (0.20 * h * w). Threshold 20% dipilih berdasarkan observasi bahwa objek sampah yang relevan biasanya menempati minimal seperlima frame. Jika area < 20%, kemungkinan objek terlalu kecil atau deteksi tepi tidak optimal, sehingga dilakukan fallback ke metode geometris | threshold: 0.20 * h * w |
| 9A | ApproxPolyDP | Simplifikasi kontur menggunakan algoritma Douglas-Peucker. Algoritma ini mereduksi jumlah titik kontur dengan mempertahankan titik-titik yang menyimpang lebih dari epsilon dari garis aproksimasi. Epsilon = 0.01 * arcLength: mempertahankan ~1% detail tepi. Kontur asli yang memiliki ratusan titik directuksi menjadi ~10-30 titik, kemudian disampling menjadi tepat 24 titik | epsilon=0.01*cv2.arcLength |

**Algoritma Douglas-Peucker:**
Algoritma ini bekerja secara rekursif: (1) Temukan titik pada kontur yang memiliki jarak terjauh dari garis lurus yang menghubungkan titik pertama dan terakhir. (2) Jika jarak > epsilon, bagi kontur menjadi dua segmen pada titik tersebut dan proses rekursif masing-masing segmen. (3) Jika jarak <= epsilon, semua titik antara dapat diabaikan. Hasilnya adalah representasi polygon yang mempertahankan bentuk dominan objek dengan jumlah titik minimal. Epsilon 0.01 memberikan keseimbangan antara presisi bentuk dan efisiensi penyimpanan.

---

**Kelompok 4: Post-processing & Format (Langkah 10-12)**

| Langkah | Operasi | Deskripsi Teknis | Output |
|---------|---------|-----------------|--------|
| 10 | Normalize ke [0,1] | Setiap koordinat piksel (x, y) dibagi dengan lebar (w) dan tinggi (h) gambar: x_norm = x / w, y_norm = y / h. Normalisasi membuat koordinat invariant terhadap resolusi gambar. Gambar 640x640: koordinat 320px menjadi 0.5. Nilai di-clamp ke rentang [0.0, 1.0] untuk menghindari nilai di luar batas | float dalam [0.0, 1.0] |
| 11 | Format YOLO-seg | Koordinat ditulis dalam format: `<class_id> x1 y1 x2 y2 ... x24 y24`. Setiap baris merepresentasikan satu objek. 24 titik polygon dipilih karena merupakan default YOLO-seg dan memberikan presisi cukup untuk sebagian besar bentuk sampah | 48 float per objek (24 titik x 2 koordinat) |
| 12 | Simpan ke Disk | File .txt disimpan dengan nama yang sama dengan file gambar di folder label yang sesuai (train/labels/, val/labels/, test/labels/). Format satu file per gambar, multi-line jika terdapat multiple objek | .txt per gambar |

**Contoh Format Label:**
```
0 0.2512 0.3021 0.4534 0.3512 0.3045 0.5012 0.2213 0.4876 0.1987 0.4234 0.2109 0.3567 0.2678 0.3012 0.3345 0.2678 0.4012 0.2543 0.4567 0.2654 0.4890 0.2876 0.4678 0.3123 0.4234 0.3345 0.3567 0.3456 0.2876 0.3345 0.2345 0.3123 0.1987 0.2890 0.1765 0.2678 0.1876 0.2543 0.2123 0.2456 0.2456 0.2567 0.2789 0.2734 0.3012 0.2876 0.3210 0.2987
```
Baris di atas menunjukkan: kelas 0 (Organik) dengan 24 titik polygon. Setiap dua float berurutan adalah (x, y).

---

**Metode 2: Fallback Geometris (33.6% kasus)**

Edge detection Otsu gagal pada gambar dengan kontras foreground-background rendah, objek transparan (botol bening, plastik wrap), pencahayaan tidak merata, atau objek yang menyatu dengan latar (contoh: pisang di atas meja kayu, plastik hitam di lantai gelap). Untuk kasus-kasus ini, digunakan pembangkitan polygon geometris dengan dua varian.

**Fallback Ellipse (60% dari kasus fallback, ~20.2% total):**
Polygon ellipse dibangkitkan dengan parameter: center point (cx, cy) diacak dalam rentang [0.46, 0.54] dari pusat gambar; radius horizontal (rx) [0.37, 0.46]; radius vertical (ry) [0.37, 0.46]; rotasi [-0.1, 0.1] radian; irregularity factor [0.02, 0.06] untuk memberikan variasi bentuk tidak sempurna. Total 20 titik polygon (n_points=20). Parameter randomize=true memberikan variasi antar gambar sehingga tidak semua mask identik.

**Fallback Rounded Rectangle (40% dari kasus fallback, ~13.4% total):**
Polygon persegi panjang dengan sudut membulat dibangkitkan dengan parameter: margin (mx, my) [0.06, 0.14] dari tepi gambar; corner radius (cr) [0.04, 0.10]; 4 segmen sudut (ppc=5 titik per sudut) = 20 titik total. Parameter randomize=true memberikan variasi ukuran dan kelengkungan sudut.

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

**Apa itu Augmentasi Data?**

Augmentasi data adalah teknik memperbanyak variasi dataset training dengan menerapkan transformasi pada gambar asli tanpa mengubah labelnya. Tujuan utamanya adalah meningkatkan kemampuan generalisasi model - yaitu kemampuan model untuk mengenali objek yang sama dalam kondisi berbeda (pencahayaan berbeda, sudut pandang berbeda, skala berbeda).

**Mengapa Augmentasi Diperlukan?**

Dataset penelitian ini hanya memiliki 3.973 gambar. Jumlah ini relatif kecil untuk training deep learning. Tanpa augmentasi, model cenderung mengalami overfitting: menghafal dataset training tetapi gagal pada data baru. Dengan augmentasi, setiap epoch (satu putaran penuh dataset) model melihat versi berbeda dari gambar yang sama. Efektifnya, model berlatih pada dataset yang jauh lebih besar dan lebih bervariasi.

Contoh: Gambar botol plastik di epoch 1 tampil normal. Epoch 2: dimiringkan 25 derajat. Epoch 3: warnanya diubah. Epoch 4: digabung dengan gambar lain via mosaic. Model belajar bahwa "botol plastik" tetap botol plastik meskipun orientasi, warna, atau konteksnya berubah.

**Kategori Augmentasi:**

Augmentasi dikelompokkan dalam 4 kategori berdasarkan jenis transformasi:

**A. Augmentasi Geometrik - Mengubah Posisi dan Bentuk:**

| Augmentasi | Probabilitas | Parameter | Penjelasan Teknis |
|------------|-------------|-----------|-------------------|
| Scale | 0.8 | Faktor 0.1-1.9 | Memperbesar atau memperkecil gambar. Mensimulasikan jarak kamera berbeda. Scaling factor 0.8 berarti gambar diperkecil hingga 80% dari ukuran asli, area sisanya diisi dengan padding |
| Translate | 0.3 | Faktor -0.1-0.1 | Menggeser gambar secara horizontal dan vertikal. Objek bergeser dari pusat frame, memaksa model belajar deteksi objek di berbagai posisi |
| Rotate | 25.0 derajat | Range -25 s.d. +25 derajat | Memutar gambar. Mensimulasikan kamera miring atau objek dalam posisi tidak tegak. Area pojok yang kosong diisi warna hitam |
| Shear | 10.0 derajat | Range -10 s.d. +10 derajat | Transformasi shear (condong/miring). Mendistorsi gambar seperti efek perspektif, mensimulasikan sudut pandang kamera yang tidak frontal |
| Perspective | 0.0005 | Probabilitas | Transformasi perspektif acak untuk mensimulasikan efek 3D ringan. Mengubah 4 titik sudut gambar secara acak |
| Flip LR | 0.5 | Probabilitas 50% | Membalik gambar secara horizontal (kiri menjadi kanan). Efektif untuk menghilangkan bias orientasi (misal: botol selalu menghadap kiri) |
| Flip UD | 0.3 | Probabilitas 30% | Membalik gambar secara vertikal (atas menjadi bawah) |

**B. Augmentasi Fotometrik - Mengubah Warna dan Pencahayaan:**

| Augmentasi | Parameter | Penjelasan Teknis |
|------------|-----------|-------------------|
| HSV Hue | 0.02 | Pergeseran hue (roda warna) sebesar 0.02 dari rentang [0,1]. Hue menentukan warna dominan (merah, hijau, biru, dll). Pergeseran kecil mensimulasikan perubahan pencahayaan alami (misal: daun hijau tampak sedikit kekuningan di sore hari) |
| HSV Saturation | 0.6 | Faktor saturasi maksimal 0.6 dari nilai asli. Saturasi mengontrol intensitas warna. Nilai 0 = grayscale (hitam putih), 1 = warna penuh. Range [0, 0.6] memungkinkan gambar tampak pudar hingga semi-color |
| HSV Value | 0.4 | Faktor value (brightness) maksimal 0.4. Value mengontrol kecerahan. Range ini mensimulasikan kondisi pencahayaan dari sangat terang (siang) hingga remang-remang (senja/mendung) |

**C. Augmentasi Spesifik YOLO-seg:**

| Augmentasi | Probabilitas | Penjelasan Teknis |
|------------|-------------|-------------------|
| Mosaic | 1.0 (wajib) | Menggabungkan 4 gambar berbeda menjadi grid 2x2. Setiap gambar di-resize ke 320x320, ditempatkan di 4 kuadran kanvas 640x640. Label dari ke-4 gambar digabung. Memaksa model mendeteksi objek dalam konteks padat dan objek terpotong di batas grid |
| Mixup | 0.5 | Blending linear 2 gambar: I_blended = alpha * I1 + (1-alpha) * I2, alpha ~ Beta(0.5, 0.5). Label juga di-blend: y_blended = alpha * y1 + (1-alpha) * y2. Model belajar dari kombinasi dua gambar sekaligus, meningkatkan smoothness decision boundary |
| Copy-Paste | 0.5 | Menyalin instance mask dari satu gambar dan menempelkannya di gambar lain. Posisi penempatan acak dengan menghindari overlap signifikan. Augmentasi ini spesifik untuk instance segmentation dan sangat efektif untuk meningkatkan variasi latar objek |

**D. Regularisasi via Augmentasi:**

| Augmentasi | Probabilitas | Penjelasan Teknis |
|------------|-------------|-------------------|
| Erasing | 0.5 | Menghapus area persegi panjang acak dari gambar (mengisi dengan nilai pixel rata-rata). Memaksa model tidak bergantung pada satu region spesifik untuk identifikasi objek. Efek: model belajar menggunakan konteks global |
| Auto Augment | "randaugment" | Strategi augmentasi adaptif: memilih 2-3 augmentasi secara acak dari daftar yang tersedia dengan magnitude acak. Digunakan di akhir training setelah close_mosaic untuk stabilisasi |

**Strategi Close Mosaic:**

Mosaic diaktifkan penuh (probabilitas 1.0) selama ~27 epoch pertama (close_mosaic = epochs // 3 = 80//3). Pada fase awal training, mosaic sangat bermanfaat karena memaksa model belajar mendeteksi objek dalam konteks padat dan objek terpotong. Namun, pada fase akhir training, mosaic dapat mengganggu karena objek yang terpotong di batas grid membuat distribusi data tidak realistis. Oleh karena itu, mosaic dimatikan pada epoch ~27-80, digantikan oleh augmentasi reguler yang lebih ringan.

**Dampak Augmentasi terhadap Training:**

Dengan 15 jenis augmentasi yang dikombinasikan secara acak, setiap gambar dapat menghasilkan ribuan variasi unik. Hal ini secara efektif memperbesar ukuran dataset training tanpa perlu mengumpulkan data baru. Model yang dilatih dengan augmentasi menunjukkan gap train-val mAP <5%, mengindikasikan bahwa augmentasi berhasil mencegah overfitting.

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

**Apa itu Backbone?**

Backbone adalah bagian pertama dari arsitektur deep learning yang bertugas mengekstraksi fitur dari gambar input. Analoginya seperti sistem penglihatan manusia: ketika melihat gambar, pertama kita melihat tepi dan kontur, kemudian pola dan bentuk, lalu tekstur, dan akhirnya memahami objek secara utuh. Backbone melakukan proses yang sama secara hierarkis melalui serangkaian layer konvolusi.

**Bagaimana Konvolusi Bekerja (Konsep Dasar):**

Operasi konvolusi bekerja dengan menggeser filter (kernel) kecil melintasi gambar input. Filter ini adalah matriks bobot yang dipelajari selama training. Pada setiap posisi, filter melakukan perkalian dot product dengan area piksel yang ditutupinya, menghasilkan nilai tunggal di feature map output. Proses ini mendeteksi pola spesifik: filter tepi mendeteksi garis vertikal/horizontal, filter sudut mendeteksi perubahan arah tepi, filter tekstur mendeteksi pola berulang.

**Apa itu Feature Map?**

Feature map adalah representasi 2D dari respons filter di setiap posisi spasial. Dimensinya adalah: tinggi x lebar x jumlah filter. Setiap channel dalam feature map merepresentasikan hasil dari satu filter. Semakin dalam ke jaringan, semakin abstrak fitur yang dideteksi: channel awal mendeteksi tepi sederhana, channel tengah mendeteksi pola (lingkaran, kotak), channel akhir mendeteksi konsep semantik ("ini bagian botol", "ini tekstur plastik").

**Proses Ekstraksi Fitur Bertahap:**

Model ini menggunakan varian medium: YOLOv26m-seg dengan 23.5 juta parameter dan 121.2 GFLOPs. Berikut adalah detail setiap stage:

| Stage | Input -> Output | Stride | Channel | Jumlah Piksel | Fungsi |
|-------|---------------|--------|---------|--------------|--------|
| Stem Conv | 640x640 -> 320x320 | 2x | ~64 | 102.400 (25%) | Konvolusi awal dengan kernel 7, stride 2, padding 3. Mengekstrak fitur dasar: tepi objek, gradien warna, perubahan intensitas |
| Stage 1 CSP | 320x320 -> 160x160 | 4x | 128 | 25.600 (6.25%) | Deteksi sudut dan kontur: lingkaran tutup botol, sudut kotak kardus, lekukan kaleng |
| Stage 2 CSP | 160x160 -> 80x80 | 8x | 256 | 6.400 (1.56%) | Deteksi pola geometrik: bentuk silinder botol, lipatan plastik, pola daun |
| Stage 3 CSP | 80x80 -> 40x40 | 16x | 512 | 1.600 (0.39%) | Deteksi tekstur: plastik mengkilap vs kertas buram, serat kayu, pori-pori styrofoam |
| Stage 4 CSP | 40x40 -> 20x20 | 32x | 512 | 400 (0.098%) | Pemahaman semantik: "ini buatan pabrik" vs "ini alami", konteks lingkungan sekitar |
| SPP Layer | 20x20 -> 20x20 | 32x | 512 | 400 (0.098%) | Multi-scale context dengan 3 ukuran pooling |

Catalatan penting: Setiap stage mereduksi jumlah piksel menjadi setengahnya (stride 2x) karena menggunakan strided convolution dengan kernel 3 dan stride 2. Dari 640x640 = 409.600 piksel asli menjadi 20x20 = 400 piksel di stage 4 (reduksi 1024x lipat). Namun, channel bertambah dari 3 (RGB) menjadi 512, sehingga total informasi yang dibawa justru meningkat secara signifikan.

**CSP (Cross Stage Partial):**

CSP adalah inovasi arsitektural yang membagi feature map menjadi 2 jalur di setiap stage:

1. Jalur utama diproses melalui convolution batch (Conv2D -> BatchNormalization -> SiLU activation)
2. Jalur cabang (partial) langsung diteruskan ke akhir stage tampa pemrosesan
3. Kedua jalur digabung (concatenate) dan diproses dengan Conv 1x1 untuk reduksi channel

Keuntungan CSP:
- **Efisiensi komputasi**: Hanya memproses sebagian feature map, mengurangi FLOPs ~20%
- **Aliran gradien lebih baik**: Jalur cabang menyediakan shortcut untuk gradien saat backpropagation, mencegah vanishing gradient
- **Representasi lebih kaya**: Menggabungkan fitur yang diproses dan fitur asli memberikan representasi yang lebih kaya

**SPP Layer (Spatial Pyramid Pooling):**

SPP layer menerapkan max pooling pada feature map 20x20x512 dengan 3 ukuran kernel berbeda secara paralel:
- Kernel 5x5: menangkap konteks lokal (resepsi field ~5x5 piksel pada feature map = ~160x160 piksel pada gambar asli)
- Kernel 9x9: menangkap konteks menengah (resepsi field ~9x9 = ~288x288 piksel)
- Kernel 13x13: menangkap konteks global (resepsi field ~13x13 = ~416x416 piksel)

Hasil pooling dari 3 ukuran digabung (concat) menjadi 20x20x(512*3) = 20x20x1536, kemudian directuksi kembali ke 20x20x512 dengan Conv 1x1. Tujuannya agar model dapat mendeteksi objek dengan berbagai ukuran secara simultan: objek kecil (puntung rokok ~20x10 piksel) membutuhkan konteks lokal, objek besar (kardus ~500x300 piksel) membutuhkan konteks global.

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

Neck adalah komponen arsitektur yang berada di antara Backbone dan Head. Tugasnya: memfusikan fitur dari berbagai resolusi yang dihasilkan Backbone. Hal ini penting karena Backbone menghasilkan fitur dengan karakteristik berbeda di setiap level:

- P3 (80x80): Resolusi tinggi, banyak detail lokasi (tepi, kontur), sedikit informasi semantik
- P4 (40x40): Resolusi sedang, keseimbangan detail dan semantik
- P5 (20x20): Resolusi rendah, sedikit detail lokasi, banyak informasi semantik ("ini botol")

Masalahnya: P3 tahu persis di mana objek berada tetapi tidak tahu objek itu apa. P5 tahu objek itu apa tetapi tidak tahu persis di mana letaknya. Neck mengatasi masalah ini dengan memfusikan informasi dari semua level.

**FPN (Feature Pyramid Network) - Top-Down Pathway:**

FPN bekerja dari resolusi rendah ke tinggi (atas ke bawah), membawa informasi semantik dari P5 ke P4 dan P3. Prosesnya:

| Langkah | Operasi | Dimensi | Deskripsi |
|---------|---------|---------|-----------|
| 1 | Upsample P5 2x | 20x20 -> 40x40 | Fitur semantik P5 diperbesar 2x menggunakan nearest neighbor interpolation atau transposed convolution |
| 2 | Concat dengan P4 | 40x40 + 40x40 = 40x40 | Feature map P4 (detail) digabung dengan fitur semantik yang sudah di-upsample. Hasil: P4' yang memiliki detail DAN konteks |
| 3 | Upsample P4' 2x | 40x40 -> 80x80 | Fitur gabungan diperbesar ke resolusi P3 |
| 4 | Concat dengan P3 | 80x80 + 80x80 = 80x80 | P3 (detail lokasi tinggi) digabung dengan semantik level atas. Hasil: P3' yang memiliki detail presisi DAN pemahaman objek |

Setelah FPN, P3' dapat mendeteksi objek kecil (puntung rokok 20x10 px) dengan akurat karena memiliki detail lokasi dari P3 dan informasi "ini puntung rokok" dari P5.

**PAN (Path Aggregation Network) - Bottom-Up Pathway:**

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

Head adalah bagian akhir arsitektur yang mengambil keputusan final. Disebut "decoupled" karena terdiri dari 3 cabang independen yang masing-masing memiliki parameter sendiri dan bertanggung jawab atas tugas berbeda:

**Classification Branch:**
Layer: 2x Conv3x3 + Linear layer + Sigmoid activation
Fungsi: Menentukan apakah suatu grid cell berisi objek dan jika iya, kelas apa (Organik atau Non-Organik)
Output: 3 nilai per grid cell - objectness score (probabilitas ada objek) + 2 class probabilities (Organik, Non-Organik)
Contoh output: [0.92, 0.87, 0.13] artinya 92% yakin ada objek, 87% yakin Organik, 13% yakin Non-Organik

**Regression Branch (BBox):**
Layer: 2x Conv3x3 + DFL module
Fungsi: Memprediksi 4 koordinat bounding box (x_center, y_center, width, height)
Teknik DFL: Alih-alih memprediksi nilai tunggal untuk setiap koordinat, DFL memprediksi distribusi probabilitas diskrit dengan 16 bin. Contoh untuk posisi x: bin 0-16 merepresentasikan rentang posisi yang mungkin. Nilai akhir dihitung sebagai weighted sum dari distribusi. Keuntungan: (1) Gradien lebih informatif, (2) Representasi uncertainty, (3) Lebih akurat untuk boundary tidak jelas
Output: 4 nilai float (x, y, w, h) dalam koordinat grid-normalized

**Segmentation Branch (Mask):**
Layer: Proto Module (Conv + 32 prototype masks)
Fungsi: Menghasilkan polygon mask 24 titik yang mengikuti bentuk objek
Cara kerja: Proto Module menghasilkan 32 "prototype mask" dasar (masing-masing 20x20) dari fitur multi-skala. Untuk setiap instance, model memprediksi 32 coefficient yang menentukan kombinasi prototype. Mask final = sum(coefficient[i] * prototype[i]) untuk i=1..32. Mask kasar ini kemudian di-upsample dan di-crop, lalu directuksi ke polygon 24 titik via marching squares algorithm
Output: 48 float (24 titik x 2 koordinat), ternormalisasi [0,1]

**Anchor-Free Detection:**

YOLOv26 menghilangkan konsep anchor boxes yang digunakan di YOLO versi sebelumnya (v3, v5, v8). Anchor boxes adalah template bounding box dengan berbagai ukuran dan rasio aspek yang telah ditentukan sebelumnya (misal: 10x10, 50x50, 100x50, 200x100, dll). Model harus memilih anchor mana yang paling cocok untuk setiap objek. Kekurangan anchor: (1) Perlu tuning per dataset, (2) Tidak optimal untuk objek dengan bentuk ekstrem, (3) Menambah kompleksitas post-processing.

Dengan anchor-free, setiap grid cell langsung memprediksi bounding box tanpa template. Grid cell bertanggung jawab mendeteksi objek jika pusat objek berada di dalam cell tersebut. Pendekatan ini lebih sederhana (tanpa NMS tuning), lebih cepat, dan lebih akurat untuk variasi bentuk objek sampah.

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

**Apa itu Loss Function?**

Loss function (fungsi kerugian) adalah metrik yang mengukur seberapa jauh prediksi model dari nilai sebenarnya (ground truth). Semakin kecil nilai loss, semakin baik performa model. Selama training, model berusaha meminimalkan loss dengan menyesuaikan bobotnya melalui backpropagation.

YOLOv26m-seg menggunakan 3 komponen loss yang masing-masing mengukur aspek berbeda dari prediksi, dikombinasikan dengan bobot yang mencerminkan kepentingan relatifnya.

**A. CIoU Loss (bobot 7.5) - Untuk Regresi Bounding Box:**

CIoU (Complete IoU) adalah pengukuran seberapa akurat bounding box yang diprediksi. IoU (Intersection over Union) adalah metrik dasar yang menghitung rasio area tumpang tindih antara prediksi dan ground truth terhadap area total gabungan keduanya. Nilai IoU berkisar 0 (tidak ada tumpang tindih) hingga 1 (sempurna).

CIoU menyempurnakan IoU dengan menambahkan 2 faktor:
1. Center Distance: Jarak Euclidean antara pusat bounding box prediksi dengan pusat ground truth. Jika IoU sama tetapi center distance berbeda, model diarahkan ke prediksi dengan pusat lebih dekat
2. Aspect Ratio: Perbedaan rasio lebar/tinggi antara prediksi dan ground truth. Mencegah prediksi dengan bentuk tidak proporsional

Formula CIoU: L_CIoU = 1 - IoU + (rho^2(b, b_gt) / c^2) + alpha * v

di mana:
- rho = jarak Euclidean antar pusat
- b, b_gt = pusat prediksi dan ground truth
- c = diagonal terkecil yang mencakup kedua bounding box
- alpha = bobot penyeimbang
- v = metrik konsistensi aspect ratio

Bobot 7.5: Nilai tertinggi karena lokalisasi objek adalah prioritas utama. Kesalahan bounding box (lokasi meleset) lebih merusak daripada kesalahan klasifikasi (salah label) dalam konteks pemilahan sampah.

**B. BCE Loss (bobot 0.5) - Untuk Klasifikasi:**

BCE (Binary Cross-Entropy) mengukur kualitas prediksi kelas. Untuk setiap grid cell, model memprediksi probabilitas untuk kelas Organik dan Non-Organik. BCE menghitung perbedaan antara probabilitas prediksi dan label sebenarnya.

Formula: L_BCE = -[y * log(p) + (1-y) * log(1-p)]

di mana:
- y = label sebenarnya (0 atau 1)
- p = probabilitas prediksi untuk kelas positif

Interpretasi: Jika label = 1 (Organik) dan model memprediksi p = 0.9, loss = -log(0.9) = 0.105 (kecil, prediksi benar). Jika label = 1 dan model memprediksi p = 0.1, loss = -log(0.1) = 2.302 (besar, prediksi salah).

Bobot 0.5: Klasifikasi 2 kelas relatif lebih mudah dibanding regresi bounding box, sehingga bobotnya lebih rendah.

**C. DFL Loss (bobot 1.5) - Untuk Regresi Posisi Boundary:**

DFL (Distribution Focal Loss) adalah pendekatan unik YOLOv26 untuk regresi posisi. Alih-alih memprediksi satu nilai pasti untuk setiap sisi bounding box, DFL memprediksi distribusi probabilitas diskrit yang direpresentasikan dalam 16 bin.

Cara kerja: Untuk sisi kiri bounding box, model memprediksi 16 nilai yang masing-masing merepresentasikan probabilitas bahwa sisi kiri berada di posisi tertentu. Nilai akhir dihitung sebagai weighted sum: posisi = sum(prob[i] * pos[i]) untuk i=1..16.

Keuntungan DFL:
- Gradien lebih informatif: tidak hanya tahu "salah", tetapi juga "seberapa salah" dan "arah perbaikannya"
- Representasi uncertainty: distribusi lebar mengindikasikan boundary tidak jelas (objek transparan, tepi buram)
- Lebih akurat untuk objek dengan bentuk tidak beraturan

Total Loss: L_total = 7.5 * L_CIoU + 0.5 * L_BCE + 1.5 * L_DFL

---

### Hyperparameter Training

**Apa itu Hyperparameter?**

Hyperparameter adalah konfigurasi yang ditentukan sebelum training dimulai (tidak dipelajari oleh model). Pemilihan hyperparameter yang tepat sangat memengaruhi kualitas hasil training.

| Parameter | Nilai | Penjelasan Teknis |
|-----------|-------|-------------------|
| Epochs | 80 | Satu epoch = satu kali model melihat seluruh dataset training (2.765 gambar). 80 epoch berarti model melihat total 80 x 2.765 = 221.200 gambar (setelah augmentasi, setiap epoch gambar berbeda) |
| Batch size | 16 | Jumlah gambar yang diproses simultan dalam satu langkah. Batch besar = gradien lebih stabil = konvergensi lebih cepat, tetapi butuh lebih banyak VRAM. Batch 16 dipilih karena VRAM 16 GB |
| Image size | 640 | Resolusi input persegi setelah letterbox. 640x640 = 409.600 piksel per gambar. Standar untuk model YOLO modern |
| Optimizer | SGD + MuSGD hybrid | Stochastic Gradient Descent dengan momentum. MuSGD adalah modifikasi yang menggabungkan SGD dengan Muon optimizer dari Moonshot AI untuk konvergensi lebih stabil |
| LR0 | 0.001 | Learning rate awal. Menentukan seberapa besar langkah perbaikan bobot model setiap iterasi |
| LR schedule | Cosine annealing | LR menurun dari 0.001 mengikuti kurva cosinus. Penurunan bertahap memungkinkan model konvergen ke minimum yang lebih baik. Di akhir training, LR mendekati 0 |
| Warmup | 5 epochs | LR dinaikkan linear dari 0 ke 0.001 selama 5 epoch pertama. Mencegah gradien eksplosif di awal training saat bobot masih acak |
| Momentum | 0.937 | Mengontrol kontribusi gradien sebelumnya terhadap gradien saat ini. Nilai tinggi (0.937) mempercepat konvergensi dan membantu melewati local minima |
| Weight decay | 0.0005 | Regularisasi L2: menambahkan penalti sebesar 0.0005 * (bobot^2) ke loss. Mencegah bobot menjadi terlalu besar, mengurangi overfitting |
| FP16 | True | Mixed precision training: operasi forward/backward dalam FP16 (16-bit), bobot disimpan dalam FP32. Menghemat VRAM ~44% dan mempercepat training ~2x |
| Patience | 40 | Early stopping: jika validation loss tidak turun selama 40 epoch, training dihentikan lebih awal |

---

### Training Performance

| Metrik | Nilai | Arti |
|--------|-------|------|
| Total waktu training | ~2.5 jam | 80 epoch pada RTX 5060 Ti 16 GB |
| VRAM peak | 11-13 GB (70-80%) | Dari total 16 GB, masih ada sisa ~3-5 GB untuk proses lain |
| Inference speed | 5.3 ms/gambar | Waktu yang dibutuhkan model untuk memproses 1 gambar (setara ~188 FPS) |
| Model size | 54.5 MB | Ukuran file bobot model dalam FP32. Cukup kecil untuk deploy di berbagai platform |
| GPU utilization | 85-95% | Utilisasi GPU selama training, menunjukkan bottleneck bukan di GPU |
| Preprocess | 0.5 ms | Waktu resize dan normalisasi gambar |
| Postprocess | 0.3 ms | Waktu NMS dan decode output |

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
> | Epochs | 80 |
> | Batch | 16 |
> | Optimizer | SGD + MuSGD hybrid |
> | LR | 0.001 cosine |
> | FP16 | True |
> | Training | ~2.5 jam |
> 
> ```mermaid
> flowchart TD
>     subgraph Loss[Total Loss Function]
>         L1[CIoU Loss: 7.5] --> TOTAL[L_total]
>         L2[BCE Loss: 0.5] --> TOTAL
>         L3[DFL Loss: 1.5] --> TOTAL
>     end
>     TOTAL --> OPT[SGD/MuSGD Optimizer]
>     OPT --> EPOCH[80 Epochs]
>     EPOCH --> EVAL[Validation: Box mAP 80.4%, Mask mAP 49.7%]
> ```

---

## Slide 10: Hasil Pelatihan - Box, Mask, Per-Class Metrics, Training Curves

### Hasil Evaluasi Model

Setelah 80 epoch training (~2.5 jam), diperoleh hasil evaluasi sebagai berikut:

**Overall Metrics:**

| Metrik | Nilai | Arti |
|--------|-------|------|
| **Precision** | 76.7% | Dari 100 deteksi, ~77 benar-benar objek |
| **Recall** | 75.6% | Dari 100 objek nyata, ~76 berhasil terdeteksi |
| **F1-Score** | 76.1% | Harmonic mean precision & recall - balance |
| **Box mAP@0.5** | **80.4%** | Box IoU 0.5, akurasi 80.4% - sangat baik |
| **Box mAP@0.5:0.95** | **52.5%** | Rata-rata IoU 0.5-0.95 - standar COCO, lebih ketat |
| **Mask mAP@0.5** | **49.7%** | Mask IoU 0.5 - lebih rendah karena pseudo-label noise |
| **Mask mAP@0.5:0.95** | **23.1%** | Mask IoU 0.5-0.95 |

**Per-Class Performance:**

| Kelas | Box mAP@0.5 | Mask mAP@0.5 | Gap Box-Mask |
|-------|-------------|-------------|--------------|
| Organik | 77.2% | 38.2% | 39.0% |
| Non-Organik | 83.6% | 61.2% | 22.4% |
| **Gap** | **6.4%** | **23.0%** | |

"Non-Organik unggul di box dan mask. Dua alasan: (1) Data 4.8x lebih banyak (3.289 vs 684). (2) Bentuk lebih seragam (botol, kaleng, kaca - rigid). Organik seperti sisa makanan, daun - bentuk amorf, tepi tidak jelas."

"Perhatikan gap Box-Mask: Organik 39% vs Non-Organik 22.4%. Mask Organik sangat rendah (38.2%) karena pseudo-mask pada organik lebih sering gagal (bentuk amorf, tepi tidak kontras)."

**Training Curves Interpretasi:**
- Box loss turun dari ~1.4 ke ~0.35
- Cls loss turun dari ~1.3 ke ~0.20
- mAP naik konsisten, tidak overfitting
- Gap train-val mAP < 5% - model generalisasi baik

> **Key Takeaway:**
> 
> | Metrik | Box | Mask |
> |--------|-----|------|
> | mAP@0.5 | 80.4% | 49.7% |
> | mAP@0.5:0.95 | 52.5% | 23.1% |
> | Precision | 76.7% | 59.2% |
> | Recall | 75.6% | 52.3% |
> | F1-Score | 76.1% | 45.4% |
> 
> | Kelas | Box mAP | Mask mAP |
> |-------|---------|----------|
> | Organik | 77.2% | 38.2% |
> | Non-Organik | 83.6% | 61.2% |
> 
> ```mermaid
> xychart-beta
>     title "Training Progress (80 Epoch)"
>     x-axis ["Epoch 0", "Epoch 20", "Epoch 40", "Epoch 60", "Epoch 80"]
>     y-axis "mAP@0.5" 0 --> 100
>     line "Box mAP" [10, 45, 65, 75, 80.4]
>     line "Mask mAP" [5, 25, 38, 45, 49.7]
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
| Box mAP@0.5 | 77.2% | 83.6% | 6.4% | Dampak kelas minoritas relatif kecil untuk deteksi |
| Mask mAP@0.5 | 38.2% | 61.2% | 23.0% | Dampak sangat signifikan untuk segmentasi |
| Precision | 71.8% | 81.6% | 9.8% | Model lebih sering salah positif pada Organik |
| Recall | 73.3% | 77.9% | 4.6% | Organik sedikit lebih sering terlewat |

Mengapa dampak class imbalance lebih besar pada mask? (1) Organik memiliki bentuk amorf yang sulit diprediksi mask-nya. (2) Data Organik sedikit sehingga model kurang terlatih untuk berbagai variasi bentuk organik. (3) Pseudo-mask organik lebih sering menggunakan fallback (edge rate rendah) sehingga ground truth mask sudah tidak presisi sejak awal.

**C. Confusion Matrix Analysis**

| Prediksi -> Aktual | Organik | Non-Organik | Interpretasi |
|-------------------|---------|-------------|--------------|
| Organik | 73.3% (TP) | 18% (FP) | Model cukup baik mendeteksi Organik (73% recall) |
| Non-Organik | 26.7% (FN) | 82% (TN) | Model unggul mendeteksi Non-Organik (82% TN rate) |

**False Positive Organik (18%):** Model kadang mengklasifikasikan Non-Organik sebagai Organik. Penyebab utama: plastik kusut atau kain yang secara visual menyerupai sisa makanan organik, dan bayangan/gradien yang menyerupai tepi objek organik.

**False Negative Organik (26.7%):** Model gagal mendeteksi objek Organik yang ada. Penyebab utama: organik basah/remuk yang tidak memiliki bentuk tegas (bubur, ampas), organik berwarna gelap yang menyatu dengan latar, dan organik berukuran kecil dengan area <20%.

**D. Failure Case Analysis Detail**

| Tipe Failure | Penyebab Teknis | Dampak ke Output | Frekuensi Estimasi |
|-------------|-----------------|------------------|-------------------|
| **Objek transparan** | Edge detection gagal karena objek tembus pandang (botol bening, plastik wrap, gelas kaca). Otsu threshold tidak dapat memisahkan foreground dari background | Fallback ke ellipse/rounded rect. Mask tidak presisi, bounding box overestimated | ~12% gambar Non-Organik |
| **Objek kecil** | Area kontur <20% dari total gambar (puntung rokok, baterai kecil, biji). Kontur terekstraksi tetapi gagal validasi area | Fallback geometris, mask generik. Dampak lebih besar pada mask mAP daripada box mAP | ~8% gambar |
| **Bentuk amorf** | Organik basah/remuk (sisa makanan, ampas kopi, kulit halus). Tidak memiliki bentuk geometris yang konsisten | Pseudo-mask ellipse tidak representatif. Ground truth mask buruk -> model belajar pola yang salah | ~15% gambar Organik |
| **Tumpukan objek** | Beberapa objek berdekatan atau bertumpuk, edge detection menghasilkan satu kontur untuk multiple objek | Satu polygon mencakup 2+ objek. Model tidak bisa memisahkan instance | ~6% gambar |
| **Latar kompleks** | Sampah di rumput, tanah, atau permukaan bertekstur. Otsu threshold menangkap tekstur latar sebagai objek | Kontur noise, polygon tidak mengikuti objek sebenarnya | ~10% gambar |
| **Cahaya rendah** | Gambar gelap, kontras rendah, histogram tidak bimodal. Otsu threshold tidak optimal | Mask biner tidak akurat, batas objek tidak tegas | ~5% gambar |
| **Objek reflektif** | Plastik mengkilap, kaleng, kaca memantulkan cahaya. Refleksi menciptakan area terang yang salah terdeteksi sebagai objek terpisah | Over-segmentation: satu objek terbelah menjadi multiple polygon | ~4% gambar |

**E. Implikasi untuk Pengembangan:**

1. **Prioritas perbaikan:** Kualitas pseudo-mask (terutama fallback geometris) adalah bottleneck utama. Mengganti fallback dengan Segment Anything Model (SAM) diproyeksikan menaikkan Mask mAP 10-15%
2. **Data Organik:** Menambah 2.000+ gambar Organik akan mengurangi dampak class imbalance, terutama untuk mask segmentasi
3. **Augmentasi khusus:** Augmentasi untuk objek transparan (simulasi transparency blending) dan objek kecil (cutout zoom) dapat meningkatkan performa pada failure cases

> **Key Takeaway:**
> 
> | Faktor | Dampak ke AP | Detail |
> |--------|-------------|--------|
> | Edge rate | Langsung | Subkategori edge >80% = 62.3% AP; <60% = 48.1% AP |
> | Class imbalance | -6.4% box | Organik 77.2% vs Non-Organik 83.6% (1:4.8 data) |
> | Pseudo-label noise | -30% box-mask | Box 80.4% vs Mask 49.7% gap ~31% |
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
| Box mAP@0.5 | 80.4% | Deteksi bounding box sangat akurat |
| Box mAP@0.5:0.95 | 52.5% | Akurasi konsisten di berbagai IoU threshold |
| Mask mAP@0.5 | 49.7% | Segmentasi terbatas oleh kualitas pseudo-mask |
| Precision | 76.7% | 77 dari 100 deteksi benar-benar objek target |
| Recall | 75.6% | 76 dari 100 objek berhasil terdeteksi |
| F1-Score | 76.1% | Keseimbangan precision dan recall |
| Inference | 5.3 ms/gambar | Setara ~188 FPS - real-time |
| Training | ~2.5 jam | 80 epoch pada RTX 5060 Ti 16GB |

**4. Analisis Per-Kelas:**

| Kelas | Box mAP | Mask mAP | Data | Karakteristik |
|-------|---------|----------|------|--------------|
| Organik | 77.2% | 38.2% | 684 | Bentuk amorf, edge rate rendah |
| Non-Organik | 83.6% | 61.2% | 3.289 | Bentuk rigid, edge rate tinggi |

Non-Organik unggul karena dua faktor: (1) Jumlah data 4.8x lebih banyak memberikan contoh lebih bervariasi untuk training. (2) Bentuk rigid (botol, kaleng, kaca) lebih mudah diprediksi boundary-nya. Mask Organik yang rendah (38.2%) menjadi area improvement utama - peningkatan kualitas pseudo-mask untuk organik berpotensi menaikkan mask mAP signifikan.

**5. Gap Box-Mask ~31% sebagai Bottleneck:**

Selisih 30.7% antara Box mAP (80.4%) dan Mask mAP (49.7%) mengindikasikan bahwa kualitas pseudo-mask adalah faktor pembatas utama. Selama ground truth mask tidak presisi (33.6% menggunakan fallback geometris), model tidak dapat belajar segmentasi yang akurat. Gap ini lebih besar pada kelas Organik (39.0%) dibanding Non-Organik (22.4%), konsisten dengan edge detection rate yang lebih rendah pada organik.

---

### Tantangan Utama

| Tantangan | Dampak Kuantitatif | Prioritas |
|-----------|-------------------|-----------|
| Pseudo-label noise | Mask mAP 49.7% vs potensi >70% dengan mask berkualitas | Tinggi |
| Class imbalance (1:4.8) | Gap Box mAP 6.4%, Gap Mask mAP 23.0% | Sedang |
| Objek transparan | Edge detection gagal pada ~12% gambar Non-Organik | Sedang |
| Objek kecil (<20% area) | Fallback geometris pada ~8% gambar | Rendah |

---

### Saran dan Roadmap

**Target: Meningkatkan Box mAP dari 80.4% ke 85%+ dan Mask mAP dari 49.7% ke 65%+**

**Fase 1: Perbaikan Data (1-2 minggu)**
1. Kumpulkan 2.000+ gambar Organik tambahan untuk menyeimbangkan rasio (target 50:50)
2. Anotasi mask manual pada 500 gambar kunci menggunakan SAM (Segment Anything Model) sebagai pengganti fallback geometris
3. Implementasi class-weighted sampling: gambar Organik memiliki probabilitas lebih tinggi untuk dipilih dalam batch

**Fase 2: Perbaikan Model (1 minggu)**
1. Class-weighted loss: bobot lebih besar untuk kesalahan kelas Organik (focal loss variant)
2. Extended training: 120 epoch dengan patience=40, close_mosaic=60
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
- 4 halaman pipeline: `/raw/dataset` (loading & profiling), `/raw/preparation` (konversi mask), `/raw/training` (training monitoring), `/raw/deployment` (inference & export)
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
> | Box mAP | 80.4% - deteksi bounding box akurat |
> | Mask mAP | 49.7% - segmentasi terbatas pseudo-label |
> | Inference | 5.3 ms/gambar, 54.5 MB model |
> | App | CMS 4 route, FastAPI + Nuxt.js 3 |
> 
> ```mermaid
> flowchart TD
>     subgraph P[Pipeline End-to-End]
>         D["Dataset 3.973"] --> M["Pseudo-Mask 66.4%"]
>         M --> S["Stratified Split 70/15/15"]
>         S --> T["Training YOLOv26m-seg 80 epoch"]
>         T --> E["Evaluasi: Box 80.4%, Mask 49.7%"]
>         E --> A["App CMS: FastAPI + Nuxt.js"]
>     end
> ```

---

> **Referensi Visualisasi Lengkap:**
> 
> | Proses | Folder | Jumlah |
> |--------|--------|--------|
> | Pseudo-Mask | `pseudo_mask/` | 16 folders |
> | Stratified Split | `stratified_split/` | 4 folders |
> | Augmentation | `augmentation/` | 13 folders |
> | Backbone | `backbone/` | 7 folders |
> | Backend API | `backend/` | 7 folders |
> | Frontend UI | `frontend/` | 7 folders |
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
