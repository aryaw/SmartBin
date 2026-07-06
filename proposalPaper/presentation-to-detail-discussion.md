# Presentation Detail Discussion
## Panduan Presentasi untuk Tim SmartBin
### Gaya: Professor mengajar mahasiswa - sederhana, mendalam, dan menyenangkan

---

> **Visualization Output:** Semua pipeline process images di `waste_datasource/visualization/` (54 folders, 6 processes: pseudo_mask, stratified_split, augmentation, backbone, backend, frontend). Run `backend/app/scripts/generate_visualizations.py` to regenerate.
>
> **Quick Reference:**
> - Pseudo-Mask: `waste_datasource/visualization/pseudo_mask/` (16 subfolders)
> - Stratified Split: `waste_datasource/visualization/stratified_split/` (4 chart folders)
> - Augmentation: `waste_datasource/visualization/augmentation/` (13 subfolders)
> - Backbone: `waste_datasource/visualization/backbone/` (7 diagram folders)
> - Backend API: `waste_datasource/visualization/backend/` (7 diagram folders)
> - Frontend UI: `waste_datasource/visualization/frontend/` (7 diagram folders)

---

## Slide 1: Judul
**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

### Narasi

"Selamat pagi/siang, teman-teman. Hari ini kita bahas SmartBin - sistem deteksi dan klasifikasi sampah otomatis pakai AI. Bayangkan tempat sampah pintar yang langsung tahu: 'Oh ini organik, buang ke komposter!' atau 'Ini botol plastik, masuk daur ulang!'. Itulah yang kita bangun."

**Pipeline End-to-End (baca dari kiri ke kanan):**

"Mari jelajahi peta perjalanan kita dari kiri ke kanan."

**Dataset (3.973 Gambar):**
"Dua sumber data utama: TACO dari Microsoft (1.500 gambar, 60 kategori dengan anotasi COCO) dan Waste Classification dari Kaggle (2.939 gambar, 18 subfolder). Dulu 60 kategori spesifik, kita sederhanakan jadi 2 kelas besar: Organik dan Non-Organik."

"Anggap kita punya lemari file raksasa dengan 60 laci. Kita ambil semua isi laci, kelompokkan ulang ke 2 kotak besar: Organik dan Non-Organik."

Hasil merge: 684 Organik + 3.289 Non-Organik = 3.973 gambar.

**Pseudo-Polygon Mask Generation:**
"Ini bagian paling kreatif. Dataset hanya foto biasa tanpa mask. Solusi: buat pseudo-mask otomatis! 12 langkah: RGB ke Grayscale, Blur, Otsu threshold, Morphology, Kontur, Polygon."

Edge detection Otsu berhasil 66.4%. Fallback geometris 33.6% (60% elips, 40% rounded rect).

"Seperti menggambar bentuk daun dengan 24 titik penghubung."

**Stratified Split:**
Train 2.765, Val 593, Test 593. Stratified = proporsi Organik/Non-Organik sama di setiap split.

**Training YOLOv26m-seg:**
80 Epoch, patience 40, batch 16, FP16 mixed precision. Backbone CSPDarknet, Neck FPN+PAN, Decoupled Head.

**Evaluasi:**
Box mAP@0.5 = 80.4%, Mask mAP@0.5 = 49.7%. Organik Box mAP = 77.2%, Non-Organik Box mAP = 83.6%.

**Inference:** 5.3 ms/gambar. Training: ~2.5 jam di RTX 5060 Ti 16GB.

### Code References
- `kaggle_service.py:289-420` - `prepare_from_local()`, pipeline data utama
- `kaggle_service.py:100-106` - `_generate_mask()`, dispatch edge vs fallback
- `kaggle_service.py:25-57` - `_generate_edge_polygon()`, edge detection Otsu
- `kaggle_service.py:60-74` - `_generate_ellipse_polygon()`, fallback elips
- `config.py:41` - `ORGANIC_CATEGORIES = {25}`
- `cli/train.py:30-68` - `train_one()`, konfigurasi training

> **Key Takeaway:**
> SmartBin pipeline: 2 dataset (TACO + Waste Classification) -> 3.973 gambar (684+3.289) -> pseudo-mask 66.4% edge -> stratified split -> YOLOv26m-seg 80 epoch -> Box mAP@0.5 80.4% -> web app. Satu klik di `POST /api/kaggle/pipeline/run-full`.

> **Visualization:** `waste_datasource/visualization/pseudo_mask/` (16 subfolder, step 01-12 + summary). Trace gambar dari RGB sampai YOLO label via nama file konsisten.

---

## Slide 2: Outline Presentasi

### Narasi

"Kita akan lewati 12 slide dalam presentasi ini. Bayangkan naik gunung:"

| Slide | Judul | Analogi |
|-------|-------|---------|
| 1 | Judul & Pipeline | Peta perjalanan |
| 2 | Outline | Daftar pos pendakian |
| 3 | Latar Belakang | Kenapa kita di sini? |
| 4 | Dataset & Preprocessing | Bekal dan perlengkapan |
| 5 | Pseudo-Polygon Mask Generation | Membuat peta dari nol |
| 6 | Online Augmentation | Latihan di berbagai medan |
| 7 | Backbone: CSPDarknet | Mata pendaki |
| 8 | Neck: FPN+PAN & Decoupled Head | Otak analisis |
| 9 | Loss Functions & Training | Kompas dan aturan main |
| 10 | Hasil Pelatihan | Puncak gunung |
| 11 | Pembahasan | Evaluasi perjalanan |
| 12 | Kesimpulan & Saran | Pulang dengan ilmu baru |

### Waktu Presentasi
- Total: 20-25 menit
- Per slide: ~1-2 menit
- Q&A: 10 menit

### Peta Waktu per Slide

| Slide | Waktu | Konten |
|-------|-------|--------|
| 1-3 | 4 menit | Judul, outline, latar belakang |
| 4-5 | 5 menit | Dataset, pseudo-mask |
| 6 | 2 menit | Augmentasi |
| 7-9 | 7 menit | Arsitektur, training |
| 10-11 | 5 menit | Hasil, pembahasan |
| 12 | 2 menit | Kesimpulan + app |

> **Key Takeaway:**
> 12 slide, ~25 menit. Linear: masalah -> data -> arsitektur -> training -> hasil -> penutup. 90% fokus pipeline deep learning.

> **Visualization:** `waste_datasource/visualization/` - diagram setiap proses.

---

## Slide 3: Latar Belakang - Krisis Sampah Bali, Pergub, Deep Learning

### Narasi

"Teman-teman, mari mulai dengan masalah nyata."

**Krisis Sampah di Bali:**
"Bali menghasilkan ~1.340 ton sampah SETIAP HARI. Coba bayangkan: 1 ton = berat 10 orang. 1.340 ton = berat 13.400 orang dewasa! Setiap hari!"

60% dari pariwisata. Setiap turis ~3.5 kg sampah/hari. Komposisi: 60% organik, 30% plastik, 10% lainnya.

**Masalah Utama: Pemilahan Manual**
"Petugas harus pilah manual - lambat, tidak efisien, berbahaya (jarum, bahan kimia). Salah pilah bikin daur ulang kacau."

**Pergub No.47/2019:**
Pemerintah Bali tetapkan 3 kategori: Organik (kompos), Anorganik (daur ulang), Residu (TPA). Tapi implementasi masih manual.

**Solusi: AI Deep Learning**
"Komputer bisa 'melihat' sampah dan klasifikasi real-time. Kita gunakan computer vision."

**3 Level Computer Vision:**
"Ada 3 level CV:"

1. **Klasifikasi** - "Ini organik." Cuma label. Tidak tahu di mana objeknya.
2. **Deteksi / Bounding Box** - "Ini organik di kotak ini." Lokasi perkiraan, tapi tidak presisi untuk bentuk tidak beraturan.
3. **Instance Segmentation** - "Ini organik, tepat di area ini." Setiap pixel diklasifikasi. Paling detail.

"Kita pilih level 3 karena sampah bertumpuk dan bentuk tidak beraturan. Bounding box tidak cukup presisi. Bayangkan botol plastik penyok - bounding box akan potong area kosong, mask paham bentuk sebenarnya."

**Kenapa 2 Kelas?**
"Bukan 60 kelas - model tidak perlu ribet. Dengan 2 kelas, SEMUA ORANG bisa verifikasi: 'Ini organik atau bukan?'. Non-Organik nanti dipetakan ke Anorganik (recyclable) dan Residu (landfill) di backend."

**Transfer Learning:**
"Kita tidak latih model dari nol. YOLOv26m-seg sudah dilatih di COCO (200 ribu+ gambar, 80 kelas). Kita ambil model yang sudah pintar, terus kita 'spesialisasikan' ke sampah. Seperti ambil anak SD dan kursusin jadi ahli sampah dalam 80 epoch."

### Code References
- `config.py:41` - `ORGANIC_CATEGORIES = {25}`
- `config.py:43` - `is_organic()` function
- `kaggle_service.py:267` - `ORGANIC_SUBS` untuk hierarchical structure
- `cli/train.py:18` - `YOLO(pretrained)` - load pretrained model

> **Key Takeaway:**
> Bali darurat sampah (1.340 ton/hari). Pemilahan manual gagal. 3 level CV: klasifikasi > deteksi > segmentasi - kita pilih segmentasi. 2 kelas cukup validasi publik, subkategori di backend. Transfer learning dari COCO.

> **Visualization:** `waste_datasource/visualization/stratified_split/` - distribusi dataset.

---

## Slide 4: Dataset & Preprocessing - TACO + Waste Classification, Merged 3.973 Gambar

### Narasi

"Model AI butuh data - semakin banyak dan beragam, semakin pintar. Kita pakai 2 sumber."

**Sumber 1: TACO Dataset (1.500 gambar)**
"TACO = Trash Annotations in Context. 1.500 gambar sampah di lingkungan alami (pantai, hutan, jalan). 60 kategori, anotasi COCO polygon - kualitas tinggi."

TACO jadi referensi format dan validasi pipeline. Kontribusi: 684 gambar Organik + 816 Non-Organik ke dataset final.

**Sumber 2: Waste Classification Dataset (2.939 gambar)**
"Dari Kaggle: phenomsg/waste-classification. 2.939 gambar dalam 18 subfolder. Format klasifikasi - setiap folder = 1 kategori."

**Organik (5 subfolder, ~674 dari Waste Class + sisanya dari TACO):**
- coffee_tea_bags, egg_shells, food_scraps, kitchen_waste, yard_trimmings

**Non-Organik (13 subfolder, ~2.265 dari Waste Class + sisanya dari TACO):**
- e-waste, cans, glass, paper, plastic, batteries, paints, pesticides, ceramic, diapers, plastic_bags, sanitary, styrofoam

"Total: 1.500 + 2.939 = 3.973 gambar. Organik 684 (17.2%), Non-Organik 3.289 (82.8%)."

**Merge Pipeline:**
`prepare_from_local()` di `kaggle_service.py:289-420` membaca semua gambar, deteksi struktur folder (flat atau hierarchical), mapping setiap file ke bin_id (0=Organik, 1=Non-Organik), salin ke folder tujuan.

**Stratified Split 70/15/15:**
"Stratified = berlapis. Proporsi Organik/Non-Organik dijaga SAMA di setiap split. Kalau total 17.2% Organik, train juga 17.2%, val juga, test juga."

| Split | Total | Organik | Non-Organik | % Organik |
|-------|-------|---------|-------------|-----------|
| Train | 2,765 | 476 | 2,289 | 17.2% |
| Val | 593 | 102 | 491 | 17.2% |
| Test | 593 | 102 | 491 | 17.2% |
| **Total** | **3,973** | **684** | **3,289** | **17.2%** |

"Kenapa Stratified? Supaya evaluasi ADIL. Kalau random split, val mungkin kebetulan 30% Organik, test 10% Organik - metrik jadi tidak fair."

### Code References
- `kaggle_service.py:289-420` - `prepare_from_local()` pipeline utama
- `kaggle_service.py:261-286` - `_collect_hierarchical()` baca struktur subfolder
- `kaggle_service.py:235-258` - `_collect_flat()` baca struktur flat
- `kaggle_service.py:346-351` - stratified split dengan `train_test_split(stratify=cls_ids)`
- `kaggle_service.py:18` - `SEED = 42` untuk reproducibility

> **Key Takeaway:**
> 2 dataset: TACO (1.500, 60 cats, COCO) + Waste Classification (2.939, 18 subs) -> merge ke 3.973 gambar (684 Organik + 3.289 Non-Organik). Stratified split 70/15/15 jaga proporsi. Seed 42.

> **Visualization:** `waste_datasource/visualization/stratified_split/03_class_distribution_chart/` - bar chart distribusi.

---

## Slide 5: Pseudo-Polygon Mask Generation - Edge Detection 66.4%, Fallback 33.6%

### Narasi

"Tantangan: dataset tidak punya anotasi mask. YOLO-seg butuh mask polygon. Solusi: kita BUAT mask sendiri secara otomatis."

**Dua Jalur Utama:**

**Jalur 1: Edge Detection Otsu (66.4% kasus)**

12 langkah presisi, dari RGB ke polygon normalisasi:

**Kelompok 1: Pra-pemrosesan Citra (Langkah 1-3)**

| Langkah | Operasi | Fungsi |
|---------|---------|--------|
| 1 | RGB to Grayscale | Foto warna jadi hitam-putih. `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` |
| 2 | Gaussian Blur 5x5 | Hilangkan noise kecil (debu, pixel acak). Kernel 5x5, sigma=0 |
| 3 | Siap untuk threshold | Output: grayscale halus siap Otsu |

**Kelompok 2: Thresholding & Mask Biner (Langkah 4-6)**

| Langkah | Operasi | Fungsi |
|---------|---------|--------|
| 4 | Otsu Thresholding | Komputer hitung threshold OTOMATIS dari histogram. Pisahkan objek dari latar |
| 5 | Mean > 127? Cek dominasi | Jika rata-rata >127 (didominasi putih), invert mask |
| 6 | Morphological Close + Open | Close (5x5, 2 iter): tutup lubang kecil. Open (5x5, 1 iter): hapus titik noise |

**Kelompok 3: Ekstraksi Kontur (Langkah 7-9)**

| Langkah | Operasi | Fungsi |
|---------|---------|--------|
| 7 | Find Contours | `cv2.findContours()`, ambil kontur TERBESAR (asumsi objek utama penuhi frame) |
| 8 | Area >= 20%? | Validasi: kontur menutupi minimal 20% area gambar? Jika tidak -> fallback |
| 9A | ApproxPolyDP | Sederhanakan kontur: `epsilon = 0.01 * arcLength`. Kurangi ratusan titik jadi ~24 titik |

"Seperti menggambar bentuk daun dengan 24 titik penghubung. Titik di sudut tajam, lebih jarang di bagian lurus."

**Kelompok 4: Post-processing & Format (Langkah 10-12)**

| Langkah | Operasi | Fungsi |
|---------|---------|--------|
| 10 | Normalize ke [0,1] | Bagi koordinat dengan lebar/tinggi gambar (640). Format universal |
| 11 | Format YOLO-seg | `class_id x1 y1 x2 y2 ... x24 y24` - 24 titik per polygon |
| 12 | Simpan ke disk | .txt per gambar di folder label train/val/test |

**Jalur 2: Fallback Geometris (33.6% kasus)**
"Edge detection gagal kalau objek tidak kontras dengan latar (misal: pisang di meja kayu, plastik transparan)."

- **60%: Elips** - bentuk telur/botol. Pusat acak, radius 0.37-0.46, rotasi kecil, irregularity 0.02-0.06
- **40%: Rounded Rectangle** - bentuk kertas/kotak. 4 sudut membulat (corner radius 0.04-0.10)

"Kenapa 60-40? Dari observasi: mayoritas objek punya bentuk ellipsoidal (botol, kaleng, telur). Selebihnya kotak (kardus, kertas)."

**Statistik Pseudo-Mask:**
| Metrik | Nilai |
|--------|-------|
| Edge detection success | 66.4% (2.638 gambar) |
| Fallback geometris | 33.6% (1.335 gambar) - 60% ellipse, 40% rounded rect |
| Total gambar diproses | 3.973 |

**Format Label YOLO-seg:**
```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```
Contoh: `0 0.25 0.30 0.45 0.35 0.30 0.50`
Kelas 0 (Organik), 24 titik polygon, koordinat ternormalisasi [0,1].

### Code References
- `_generate_mask()`: `kaggle_service.py:100-106` - dispatch logic
- `_generate_edge_polygon()`: `kaggle_service.py:25-57` - edge detection Otsu
- `_generate_ellipse_polygon()`: `kaggle_service.py:60-74` - fallback elips
- `_generate_rounded_rect_polygon()`: `kaggle_service.py:77-97` - fallback persegi panjang
- Label write: `kaggle_service.py:386-387`

> **Key Takeaway:**
> 2 metode pseudo-mask: Edge detection Otsu (66.4%, akurat, ~24 titik polygon via ApproxPolyDP) dan Fallback geometris (33.6%, elips 60% + rounded rect 40%). 12 langkah CV pipeline. Label YOLO-seg dengan koordinat [0,1].

> **Visualization:** `waste_datasource/visualization/pseudo_mask/` (16 subfolder) - trace gambar dari step 01 sampai 12.

---

## Slide 6: Online Augmentation - Mosaic, Mixup, Copy-Paste, HSV, Geometric

### Narasi

"Model tidak boleh cuma hafal data training. Dia harus bisa GENERALISASI ke situasi baru. Augmentasi = memberi kacamata berbeda pada model setiap kali belajar."

"Setiap epoch, model lihat versi BERBEDA dari gambar yang sama. Di epoch 1 dia lihat foto normal. Epoch 2: fotonya dimiringkan 25 derajat. Epoch 3: warnanya diubah. Epoch 4: digabung dengan foto lain. Model jadi tidak \'kaget\' ketika di lapangan ada miring, gelap, atau barang bertumpuk."

**Hyperparameter Augmentasi:**

| Augmentasi | Probabilitas | Parameter | Efek |
|------------|-------------|-----------|------|
| Mosaic | 1.0 | Wajib | 4 foto digabung grid 2x2, model belajar deteksi di konteks padat |
| Mixup | 0.5 | Alpha blend | 2 foto di-blend transparan, model belajar fitur dari overlap |
| Copy-Paste | 0.5 | Instance copy | Objek dipindah antar foto - augmentasi spesifik segmentasi |
| HSV Hue | 0.02 | Pergeseran hue | Warna bergeser pelan (misal: daun hijau jadi agak kuning) |
| HSV Sat | 0.6 | Saturasi | Warna bisa pudar (hitam putih) atau sangat tajam |
| HSV Value | 0.4 | Terang/gelap | Simulasi kondisi pencahayaan berbeda (siang, malam, mendung) |
| Scale | 0.8 | Zoom in/out | Objek bisa tampak lebih besar atau kecil (jarak kamera bervariasi) |
| Translate | 0.3 | Geser | Objek bergeser dari pusat frame |
| Rotate | 25.0 derajat | Miring | Sampah bisa terfoto miring (dari conveyor miring) |
| Shear | 10.0 derajat | Condong | Distorsi perspektif |
| Perspective | 0.0005 | Transform | Efek 3D ringan |
| Flip LR | 0.5 | Horizontal | Cermin kiri-kanan |
| Flip UD | 0.3 | Vertikal | Terbalik atas-bawah |
| Erasing | 0.5 | Random erase | Kotak acak dihapus - paksa model pakai konteks, bukan 1 region spesifik |
| Auto Augment | "randaugment" | Adaptif | Augmentasi dipilih otomatis berdasarkan data |

**Kenapa Augmentasi Penting?**
"Dataset kita hanya 3.973 gambar. Dengan augmentasi, setiap epoch model lihat versi berbeda. Efektif 'memperbanyak' data tanpa benar-benar ambil foto baru."

**Mosaic & Close Mosaic:**
"Mosaic selalu ON. Tapi di 80 epoch, kita matikan mosaic di epoch ~27 (`close_mosaic = epochs // 3`). Kenapa? Mosaic bagus awal training (belajar deteksi di konteks padat), tapi akhir training mengganggu (objek terpotong batas 4 foto)."

### Code References
- `cli/train.py:30-68` - `train_one()` dengan semua parameter augmentasi
- `cli/train.py:42` - `close_mosaic = epochs // int(os.getenv("CLOSE_MOSAIC_DIV", "3"))`
- `cli/train.py:54-55` - `mixup`, `copy_paste` dari environment variable

> **Key Takeaway:**
> 15 augmentasi online. Mosaic wajib (1.0), probabilistic untuk lainnya. Close mosaic di epoch ~27. Augmentasi vital untuk generalisasi dengan dataset terbatas (3.973 gambar).

> **Visualization:** `waste_datasource/visualization/augmentation/` (13 subfolders) - contoh setiap augmentasi.

---

## Slide 7: Backbone: CSPDarknet - 4 Stage, SPP Layer, Resolution Progression

### Narasi

"Backbone adalah 'mata' model. Tugasnya: dari gambar 640x640 pixel, ekstrak fitur bertahap."

"Bayangkan kita punya foto sampah 640x640. Backbone memproses dari detail kecil ke konsep besar:"

| Stage | Input -> Output | Stride | Channel | Fungsi pada Deteksi Sampah |
|-------|---------------|--------|---------|---------------------------|
| Stem Conv | 640x640 -> 320x320 | 2x | ~64 | Ekstraksi awal: tepi botol vs latar |
| Stage 1 CSP | 320x320 -> 160x160 | 4x | 128 | Deteksi sudut: lingkaran tutup botol, kotak kardus |
| Stage 2 CSP | 160x160 -> 80x80 | 8x | 256 | Deteksi pola: bentuk kaleng melengkung, lipatan plastik |
| Stage 3 CSP | 80x80 -> 40x40 | 16x | 512 | Deteksi tekstur: plastik mengkilap vs kertas buram |
| Stage 4 CSP | 40x40 -> 20x20 | 32x | 512 | Pemahaman semantik: 'ini buatan pabrik' vs 'ini alami' |
| SPP Layer | 20x20 -> 20x20 | 32x | 512 | Multi-scale context: pooling 5, 9, 13 |

"Setiap stage, resolusi turun setengah (stride 2). Channel naik 2x lipat. Dari 640 pixel jadi 20 pixel - kehilangan detail lokasi, tapi dapat pemahaman 'apa' objeknya."

**CSP (Cross Stage Partial):**
"Inovasi penting: setiap stage bagi feature map jadi 2 jalur."
- **Jalur utama** -> diproses convolution batch (Conv -> BN -> SiLU)
- **Jalur cabang** -> langsung concat ke output

"Hasil: ~20% lebih hemat FLOPs dibanding backbone standar dengan akurasi setara. Memungkinkan model lebih dalam tanpa peningkatan komputasi signifikan."

**SPP Layer: 3 Kaca Pembesar:**
"Spatial Pyramid Pooling dengan 3 kernel: 5, 9, 13. Seperti melihat objek dengan 3 kaca pembesar berbeda secara bersamaan. Detail kecil (puntung rokok 20x10 pixel) dan besar (kardus 500x300 pixel) tertangkap semua."

"Input 20x20x512 -> MaxPool k=5, k=9, k=13 paralel -> Concat -> Conv 1x1 reduksi ke 512."

### Code References
- Model loading: `detector.py:36-47` `load_model()` - `YOLO(str(MODEL_PATH))`
- Training config: `cli/train.py:30-68` - parameter arsitektur

> **Key Takeaway:**
> Backbone CSPDarknet: 4 stage (640->20, stride 2x tiap stage, channel 64->512) + SPP (3 pooling 5/9/13). CSP hemat ~20% FLOPs. SPP tangkap multi-scale context.

> **Visualization:** `waste_datasource/visualization/backbone/` (7 folders) - overview, CSP, SPP, feature maps, flow.

---

## Slide 8: Neck: FPN+PAN & Decoupled Head - 3 Detektif Multi-Skala

### Narasi

"Backbone sudah ekstrak fitur 3 level: P3 (80x80, detail lokasi), P4 (40x40, menengah), P5 (20x20, semantik). Tapi P5 tahu 'apa' tapi tidak tahu 'di mana'. P3 tahu 'di mana' tapi tidak tahu 'apa'. Neck menghubungkan mereka."

**FPN (Feature Pyramid Network) - dari atas ke bawah:**
"Bawa semantik dari P5 ke P3, P4. Seperti profesor senior (P5) bilang ke junior (P3): 'Itu botol, cari bentuk silinder.'"

| Langkah | Operasi | Efek |
|---------|---------|------|
| 1 | P5 (20x20) -> Upsample 2x | Fitur semantik resolusi rendah diperbesar ke 40x40 |
| 2 | Concat dengan P4 (40x40) | Fusion semantik + detail di resolusi sedang |
| 3 | Upsample 2x -> Concat dengan P3 (80x80) | Informasi semantik mencapai resolusi tinggi |

"FPN membantu deteksi objek kecil - sampah kecil seperti puntung rokok, tutup botol, baterai dapat informasi konteks dari resolusi lebih rendah."

**PAN (Path Aggregation Network) - dari bawah ke atas:**
"Bawa detail lokasi dari P3 ke P4, P5. Junior (P3) bilang ke senior (P5): 'Saya lihat tepi tajam di sini, objek di pojok kiri atas.'"

| Langkah | Operasi | Efek |
|---------|---------|------|
| 1 | P3 (80x80) -> Downsample Conv k3 s2 | Fitur detail diperkecil ke 40x40 |
| 2 | Concat dengan P4 (40x40) | Detail memperkaya fitur semantik |
| 3 | Downsample -> Concat dengan P5 (20x20) | Detail mencapai resolusi rendah |

**3 Detektif:** "P3 (objek kecil: tutup botol, puntung rokok), P4 (sedang: kaleng, botol), P5 (besar: kardus, kantong besar). Mereka bertukar informasi - sekarang semua detektif tahu detail DAN konteks."

**Decoupled Head:**
"Head adalah 'otak' yang ambil keputusan final. Dipisah jadi 3 cabang independen:"

| Cabang | Layer Detail | Output |
|--------|-------------|--------|
| **Classification** | 2x Conv3x3 + Linear | 2 kelas + objectness score |
| **Regression** | DFL (16 bin distribusi per koordinat) | x, y, w, h (bounding box) |
| **Segmentation** | Proto Module - Conv -> 32 prototype masks | 24-point polygon per objek |

**Anchor-Free:**
"YOLOv26 tidak pakai anchor boxes. Setiap grid cell (8.400 total) langsung prediksi bbox sendiri. Lebih sederhana, tidak perlu tuning anchor per dataset."

**DFL (Distribution Focal Loss):**
"Uniknya, DFL memprediksi DISTRIBUSI probabilitas posisi, bukan nilai tunggal. Model bilang: 'tepi kiri kemungkinan di pixel 100-120', bukan 'tepi kiri di pixel 105'. Lebih akurat untuk boundary tidak jelas."

"16 bin distribusi per sisi. Nilai akhir = weighted sum distribusi."

### Code References
- `detector.py:36-47` - `load_model()` memuat model dan arsitektur
- `cli/train.py:30-68` - training dengan decoupled head
- `kaggle_service.py:100-106` - mask generation (input untuk segmentation branch)

> **Key Takeaway:**
> Neck FPN+PAN: FPN (top-down, semantik ke detail) + PAN (bottom-up, detail ke semantik). Decoupled Head: 3 cabang (class, reg, seg). Anchor-free + DFL distribusi 16 bin.

> **Visualization:** `waste_datasource/visualization/backbone/06_neck_fpn_pan/` - FPN+PAN detail flow.

---

## Slide 9: Loss Functions & Training - CIoU, BCE, DFL, Hyperparameter

### Narasi

"Loss function adalah 'pengukur kesalahan'. Model berusaha mengecilkan loss setiap epoch. Semakin kecil, semakin baik prediksi."

**3 Komponen Loss:**

**CIoU Loss (bobot 7.5):**
"Untuk bounding box. CIoU = Complete IoU, 3 faktor:"

1. **IoU** - seberapa besar tumpang tindih antara prediksi dan ground truth
2. **Center Distance** - jarak pusat prediksi vs pusat sebenarnya
3. **Aspect Ratio** - perbedaan bentuk (lebar/tinggi)

"Kenapa bobot 7.5? Deteksi LOKASI adalah prioritas utama. 'Lebih baik salah label tapi kotaknya tepat, daripada label benar kotaknya meleset.'"

Formula: `L_CIoU = 1 - IoU + rho^2(b, b_gt)/c^2 + alpha * v`

**BCE Loss (bobot 0.5):**
"Binary Cross-Entropy untuk klasifikasi 2 kelas. 'Seberapa yakin model bahwa ini organik vs non-organik?'"

Formula: `L_BCE = -[y * log(p) + (1-y) * log(1-p)]`

Bobot kecil (0.5) karena klasifikasi 2 kelas relatif mudah - fokus utama bukan klasifikasi.

**DFL Loss (bobot 1.5):**
"Distribution Focal Loss untuk regresi bbox. Mendorong distribusi probabilitas ke arah nilai target."

Bobot sedang (1.5) karena penting untuk presisi boundary.

**Total Loss:**
`L_total = 7.5 x CIoU + 0.5 x BCE + 1.5 x DFL`

### Hyperparameter Training

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| Epochs | 80 | 80x lihat seluruh dataset. Dipilih berdasarkan konvergensi |
| Batch | 16 | 16 gambar per step. Dibatasi VRAM 16 GB |
| Imgsz | 640 | Resolusi input - standar YOLO |
| Optimizer | SGD (MuSGD hybrid) | SGD dengan Nesterov momentum - konvergensi stabil untuk segmentasi |
| LR0 | 0.001 -> cosine -> ~0 | Learning rate turun mengikuti kurva cosinus - decay smooth |
| Warmup epochs | 5 | LR naik linear dari 0.001 ke 0.01 di 5 epoch pertama |
| Momentum | 0.937 | Akselerasi gradient descent |
| Weight decay | 0.0005 | Regularisasi L2 cegah overfitting |
| FP16 | True | Mixed precision - hemat VRAM ~44%, training 2x lebih cepat |
| Patience | 40 | Hentikan jika val loss tidak turun 40 epoch |

### Training Performance
- **Durasi**: 80 epoch = ~2.5 jam
- **VRAM peak**: ~11-13 GB dari 16 GB (70-80%)
- **Inference**: 5.3 ms/gambar (~188 FPS)
- **Model size**: 54.5 MB

### Code References
- `cli/train.py:30-68` - `train_one()` fungsi training dengan semua hyperparameter
- `cli/train.py:70-73` - `model.val()`, pengambilan metrik
- `kaggle_cms.py:88-101` - pipeline training full dengan grid search support

> **Key Takeaway:**
> 3 loss: CIoU (bbox, bobot 7.5) + BCE (klasifikasi, bobot 0.5) + DFL (posisi distribusi, bobot 1.5). 80 epoch, batch 16, SGD+MuSGD, FP16. Training ~2.5 jam di RTX 5060 Ti.

> **Visualization:** `waste_datasource/visualization/backend/06_model_lifecycle/` - model training lifecycle.

---

## Slide 10: Hasil Pelatihan - Box, Mask, Per-Class Metrics, Training Curves

### Narasi

"Setelah 80 epoch (~2.5 jam), kita lihat hasilnya. Secara keseluruhan, model cukup memuaskan."

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

### Code References
- `cli/train.py:70-95` - `model.val()`, `results_dict`, per-class AP
- Training curves: `runs/segment/full_pipeline/results.png`
- `kaggle_cms.py:303-319` - endpoint `GET /api/kaggle/results`

> **Key Takeaway:**
> Box mAP@0.5 = 80.4%, Mask mAP@0.5 = 49.7%, Precision 76.7%, Recall 75.6%, F1 76.1%. Non-Organik (83.6%) unggul dari Organik (77.2%) data 4.8x lebih banyak. Gap Box-Mask Organik 39% karena pseudo-label noise.

> **Visualization:** `runs/segment/full_pipeline/` - results.png, labels.jpg, F1_curve.png, PR_curve.png, confusion_matrix.png.

---

## Slide 11: Pembahasan - Edge Rate vs AP, Class Imbalance, Failure Cases

### Narasi

"Mari bedah mengapa hasilnya seperti ini."

**Edge Rate vs mAP Korelasi:**
"Subkategori dengan edge rate tinggi cenderung punya mAP lebih tinggi. Contoh:"
- paper_waste (edge rate tinggi, kertas kontras dengan latar) -> mAP tinggi
- plastic_bags_waste (edge rate rendah, plastik transparan) -> mAP lebih rendah

"Kualitas pseudo-mask langsung memengaruhi hasil training. Edge detection Otsu berhasil 66.4%, turun dari dataset sebelumnya yang 83.3%. Kenapa turun? Karena TACO gambar lebih natural (outdoor, latar kompleks) lebih menantang untuk edge detection."

**Class Imbalance:**
"Organik 684 vs Non-Organik 3.289 (1:4.8). Dampak:"
- Organik Box mAP 77.2% vs Non-Organik 83.6% (gap 6.4%)
- Organik Mask mAP 38.2% vs Non-Organik 61.2% (gap 23.0%)
- Model bias ke Non-Organik - lebih sering prediksi Non-Organik

**Confusion Matrix:**
| Prediksi | Organik (aktual) | Non-Organik (aktual) |
|----------|-----------------|---------------------|
| Organik | ~72% (TP) | ~18% (FP) |
| Non-Organik | ~28% (FN) | ~82% (TN) |

**Insight:**
- Model lebih baik mendeteksi Non-Organik (82% true negative rate)
- False Positive Organik ~18% - model kadang kira plastik kusut sebagai organik
- False Negative Organik ~28% - organik sering terlewat, terutama jika bentuk tidak khas

**Failure Cases:**
1. **Objek transparan** (botol bening, plastik wrap) - edge detection gagal -> fallback -> mask tidak presisi
2. **Objek kecil** (puntung rokok, biji) - area <20%, fallback geometric
3. **Tumpukan sampah** - beberapa objek berdekatan, edge detection sulit pisahkan
4. **Latar kompleks** - sampah di rumput/tanah, deteksi tepi campur dengan tekstur latar
5. **Cahaya rendah** - gambar gelap, kontras rendah, Otsu threshold tidak optimal

### Code References
- Edge stats: `kaggle_service.py:378-383`
- Per-class metrics: `cli/train.py:70-95` output
- Model validation: `detector.py:50-56` `_validate_model()`

> **Key Takeaway:**
> Edge rate (66.4%) berkorelasi dengan kualitas mask. Class imbalance (1:4.8) turunkan performa Organik. Confusion matrix: Organik recall ~72%, Non-Organik recall ~82%. Failure cases: transparan, kecil, tumpuk, latar kompleks, cahaya rendah.

> **Visualization:** `waste_datasource/visualization/pseudo_mask/` - contoh failure case di subfolder edge_failures.

---

## Slide 12: Kesimpulan & Saran - 5 Capaian, Tantangan, Saran, Aplikasi Web

### Narasi

"Setelah perjalanan panjang, mari simpulkan."

### 5 Capaian Kunci

**1. Integrasi Dataset Berhasil**
"Dua dataset heterogen (TACO 1.500 COCO + Waste Classification 2.939 folder) digabung jadi 3.973 gambar siap YOLO-seg. 60 kategori -> 2 kelas. 684 Organik + 3.289 Non-Organik."

**2. Pseudo-Mask Pipeline Efektif**
"12 langkah CV pipeline berfungsi end-to-end. Edge detection Otsu 66.4%. Fallback geometris 33.6% (60% ellipse, 40% rounded rect). Format YOLO-seg 24 titik polygon."

**3. YOLOv26m-seg Capaian Kompetitif**
- Box mAP@0.5: **80.4%** - deteksi bounding box sangat akurat
- Mask mAP@0.5: **49.7%** - segmentasi lebih menantang karena pseudo-label noise
- Precision 76.7%, Recall 75.6%, F1 76.1%
- Inference: 5.3 ms/gambar (~188 FPS) - real-time
- Training: 80 epoch, ~2.5 jam di RTX 5060 Ti 16GB

**4. Analisis Per-Kelas**
"Non-Organik (83.6%) unggul dari Organik (77.2%) karena data 4.8x lebih banyak dan bentuk rigid. Mask Organik sangat rendah (38.2%) - area improvement utama."

**5. Gap Box-Mask ~31%**
"Box mAP 80.4% vs Mask mAP 49.7% - ini menunjukkan kualitas pseudo-mask adalah bottleneck utama. Incremental data tidak akan banyak membantu tanpa perbaikan mask."

### Tantangan
- **Pseudo-label noise** - mask tidak sempurna, batasi akurasi maksimal
- **Class imbalance** - 1:4.8, model bias ke Non-Organik
- **Objek transparan** - edge detection gagal konsisten

### Saran
1. **Kumpulkan 2.000+ gambar Organik baru** - seimbangkan rasio
2. **Anotasi mask manual** 500 gambar kunci pakai SAM (Segment Anything Model)
3. **Class-weighted loss** - bobot lebih untuk kelas Organik
4. **Coba YOLOv26l** (48M params) untuk akurasi lebih tinggi

### Aplikasi Web (Sekilas)
"Model sudah tertanam di aplikasi web CMS. FastAPI backend + Nuxt.js 3 frontend. User upload foto, sistem deteksi dalam 5.3 ms, tampilkan kelas, confidence, dan rekomendasi pembuangan."

Routes CMS: `/raw/dataset`, `/raw/preparation`, `/raw/training`, `/raw/deployment`. Satu klik pipeline di `POST /api/kaggle/pipeline/run-full`.

**Tech Stack:** FastAPI :8000, Nuxt.js 3 :3000, PostgreSQL untuk log. Docker siap deploy.

### Code References
- Pipeline: `kaggle_service.py:289-420`
- Training: `cli/train.py:30-68`
- Deployment: `detector.py:58-61` `reload_model()`
- CMS: `kaggle_cms.py`
- Detect endpoint: `detect.py:16-17` `POST /api/detect`
- Schema: `schemas/detection.py:18-26` `DetectResponse`
- Router: `main.py:67-73`

> **Key Takeaway:**
> 5 capaian: (1) Integrasi 3.973 gambar. (2) Pseudo-mask 66.4% edge. (3) Box mAP 80.4%, Mask mAP 49.7%. (4) Non-Organik unggul. (5) Gap Box-Mask ~31%. Aplikasi web CMS siap pakai untuk deteksi real-time.

> **Visualization:** `waste_datasource/visualization/` (54 folders) - seluruh pipeline tervisualisasi.

---

> **Referensi Visualisasi Lengkap:**
> Semua diagram untuk presentasi ini di `waste_datasource/visualization/`:
> - 16 folder pseudo-mask (setiap langkah dari 12)
> - 4 folder stratified split
> - 13 folder augmentation
> - 7 folder backbone
> - 7 folder backend API
> - 7 folder frontend UI
>
> **Jalankan regenerasi:**
> ```bash
> python backend/app/scripts/generate_visualizations.py
> ```
>
> **Cross-reference dengan kode:**
> Setiap visualisasi bisa dilacak ke kode sumber. Detail file path dan function name di setiap slide.
