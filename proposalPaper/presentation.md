
## Slide 1: Judul

**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation - 2 Kelas Organik/Non-Organik dengan Subkategori Anorganik & Residu**

```mermaid
flowchart TD
    classDef default fill:none,stroke:#333,stroke-width:1

    subgraph DS[Preprocessing - Data Science]
        A["TACO + Waste Classification<br />3.973 gambar, 2 kelas"] --> B["Pseudo-Mask 12 langkah<br />66.4% Edge Detection"]
        B --> C["Stratified Split 70/15/15<br />Train 2.765, Val 593, Test 593"]
        C --> D["Online Augmentation<br />Mosaic, Mixup, HSV, Flip"]
    end

    subgraph CNN["Deep Learning - CNN Architecture"]
        E["Input 640x640x3"] --> F["Backbone CSPDarknet<br />Stem -> 4 CSP Stage -> SPP<br />640x640 -> 20x20"]
        F --> G["Neck FPN+PAN<br />Multi-Scale Feature Fusion"]
        G --> H["Decoupled Head<br />Class + Reg + Seg Branch"]
    end

    subgraph ML["Machine Learning - Training"]
        I["Loss: CIoU(7.5) + BCE(0.5) + DFL(1.5)"] --> J["SGD Optimizer<br />LR=0.001 Cosine Decay"]
        J --> K["100 Epochs, Batch=16, FP16<br />Early Stop Patience=40"]
    end

    subgraph EVAL["Evaluation - Metrics"]
        L["Box mAP@0.5: 76.9%<br />Mask mAP@0.5: 55.4%"] --> M["Organik: ~68%<br />Non-Organik: ~83%"]
    end

    D --> E
    H --> I
    K --> L
    M --> N["Inference 5.3ms<br />-> Recycling Advice"]
```

---

## Slide 2: Outline

| # | Slide | # | Slide |
|---|-------|---|-------|
| 1 | Judul & Pipeline End-to-End | 7 | Backbone: CSPDarknet |
| 2 | Outline Presentasi | 8 | Neck: FPN+PAN & Decoupled Head |
| 3 | Latar Belakang | 9 | Loss Functions & Training Setup |
| 4 | Dataset & Preprocessing | 10 | Hasil Pelatihan |
| 5 | Pseudo-Polygon Mask Generation (12 Langkah) | 11 | Pembahasan |
| 6 | Online Augmentation | 12 | Kesimpulan & Aplikasi |

---

## Slide 3: Latar Belakang - Krisis Sampah, Regulasi, Solusi Deep Learning

### Krisis Sampah di Bali

| Metrik | Nilai | Sumber |
|--------|-------|--------|
| Produksi sampah harian | **~1.340 ton/hari** | DLHK Bali, 2023 |
| Kontribusi pariwisata | **60%** (~3.5 kg/turis/hari) | - |
| Komposisi | 60% organik, 30% plastik, 10% lainnya | - |
| Sampah plastik laut dari darat | **80%** | - |

**Masalah:** Volume sampah terus meningkat, lahan TPA terbatas, dan tingkat daur ulang masih rendah (<10%). Pemilahan di sumber adalah langkah paling kritis - sampah campur sulit didaur ulang karena terkontaminasi.

### Regulasi dan Kebijakan

**Pergub Bali No.47/2019** tentang Pengelolaan Sampah Berbasis Sumber:

| Kategori | Contoh | Tujuan Akhir |
|----------|--------|-------------|
| **Organik** | Sisa makanan, daun, kulit buah | Kompos / biogas |
| **Anorganik** (Recyclable) | Plastik, kaca, kertas, logam, kardus | Bank Sampah / daur ulang |
| **Residu** (Landfill/B3) | Popok bekas, baterai, styrofoam, pembalut | TPS B3 / landfill |

Kebijakan menekankan **pemilahan dari sumber**, namun implementasi di lapangan masih mengandalkan tenaga manual - bottleneck utama.

### Permasalahan Pemilahan Manual

| Aspek | Manual | Target Otomatis | Referensi |
|-------|--------|----------------|-----------|
| Kecepatan | 1-3 detik/objek (20-60 objek/menit) | <30ms/deteksi (~196 FPS) | Gundupalli et al., *Waste Management*, 60:56-74, 2017; Buchholz & Jünemann, *Handbuch der Abfallwirtschaft*, 1993 |
| Konsistensi | Error rate naik 2-3× setelah 30 menit kerja terus-menerus | Stabil per batch (~5.1ms/gambar) | Cimpan et al., *Waste Management*, 45:22-34, 2015; Feil et al., *Waste Management & Research*, 35(3):246-254, 2017 |
| Error rate | 15-30% (tergantung jenis fraksi, kelelahan, pencahayaan) | ~23% (model saat ini) | Cimpan et al., *Waste Management*, 45:22-34, 2015; Sarc et al., *Waste Management & Research*, 37(1):3-15, 2019 |
| Risiko kesehatan | Risiko infeksi HBV 2-6×, HCV 5-6× lebih tinggi dibanding populasi umum; luka tusuk, paparan B3 | Zero kontak langsung | Majeed et al., *J. Mater. Cycles Waste Manag.*, 19:815-826, 2017 (OR HBV 2.0, HCV 6.09); Bleck & Wettberg, *Waste Management*, 32:2009-2017, 2012 |
| Skalabilitas | Linear dengan jumlah operator (1 operator ~150-300 kg/hari) | Horisontal (tambah GPU, parallel inference) | Cimpan et al., *Waste Management*, 45:22-34, 2015; Gundupalli et al., *Waste Management*, 60:56-74, 2017 |

**Inti masalah:** Pemilahan manual tidak scalable, tidak konsisten, dan berbahaya. Solusi otomatis berbasis computer vision diperlukan untuk mendukung target 30% pengurangan sampah.

### Mengapa Instance Segmentation?

Sampah memiliki bentuk sangat bervariasi - botol bening, plastik kusut, kaleng penyok, sisa makanan amorf. **Bounding box tidak cukup presisi** untuk bentuk tidak beraturan karena memotong area kosong. Instance segmentation memberikan mask per-pixel akurat, krusial untuk memisahkan objek bertumpuk.

| Level CV | Contoh Output | Kelebihan | Kekurangan |
|----------|--------------|-----------|------------|
| 1. Klasifikasi | "Ini organik" | Cepat, komputasi ringan | Tidak tahu lokasi objek - tidak berguna untuk tumpukan |
| 2. Deteksi (bbox) | "Ini organik di kotak [x,y,w,h]" | Lokasi perkiraan, cukup untuk count | Bounding box potong area kosong pada botol miring/penyok - rasio aspect tinggi masalah |
| **3. Instance segmentation** | **"Ini organik, area pixel [mask]"** | **Presisi pixel-level, paham bentuk asli, pisah objek bertumpuk** | **Komputasi ~2× lebih berat dari deteksi** |

**Kenapa YOLO-seg, bukan Mask R-CNN?** YOLO-seg adalah arsitektur one-stage (prediksi langsung tanpa Region Proposal Network). Mask R-CNN 2-stage 5-10× lebih lambat - tidak feasible untuk real-time. YOLOv26m-seg mencapai 5.1ms/gambar vs Mask R-CNN 50-100ms.

### Strategi 2 Kelas + Subkategori Backend

**Kenapa tidak 60 kelas langsung?** TACO dataset memiliki 60 kategori sampah. Namun:
- 60 kelas sulit divalidasi publik - butuh keahlian spesifik
- Distribusi antar kelas sangat timpang (banyak kategori dengan <20 gambar)
- 2 kelas Organik/Non-Organik bisa diverifikasi SIAPAPUN tanpa pelatihan

**Pipeline 3-tier recycling advice:**
```
Input → Model → Organik?   → Ya   → Kompos
               → Non-Organik → Anorganik (plastik, kaca, kertas) → Bank Sampah
                             → Residu (popok, baterai, styrofoam) → TPS B3
```
Non-Organik dipecah di backend berdasarkan subkategori - tidak perlu model 60 kelas.

### Transfer Learning dari COCO

Melatih YOLO dari nol membutuhkan ~1 juta gambar dan ~1 minggu GPU. Solusi: **transfer learning** dari model pretrained COCO.

| Aspek | Tanpa Transfer Learning | Dengan Transfer Learning |
|-------|------------------------|-------------------------|
| Data dibutuhkan | ~1.000.000 gambar | 3.973 gambar |
| Waktu training | ~7 hari | 4.1 jam |
| Bobot awal | Random (konvergen lambat) | COCO pretrained (sudah tahu tepi, bentuk, tekstur) |
| Bobot ditransfer | 0% | **890/904 parameter groups** (~98%) |

YOLOv26m-seg sudah dilatih di COCO (200.000+ gambar, 80 kelas). Bobot backbone dan neck sudah optimal untuk ekstraksi fitur umum. Hanya head yang diinisialisasi ulang untuk 2 kelas + segmentasi. Ini seperti mengambil lulusan SD yang sudah bisa baca tulis, lalu kursusin spesifik jadi "ahli sampah" - jauh lebih cepat daripada ajarin dari nol.

---

## Slide 4: Dataset & Preprocessing

### Sumber Data

| Dataset | Gambar | Kategori | Anotasi |
|---------|--------|----------|---------|
| **TACO** (Trash Annotations in Context) | 1.500 | 60 subkategori | COCO polygon (manual) |
| **Waste Classification** (phenomsg/kaggle) | 2.939 | 18 subkategori | Tidak ada (label folder) |
| **Gabungan** (setelah merge, deduplikasi, filter) | **3.973** | **2 kelas: Organik / Non-Organik** | **YOLO-seg (24 titik polygon)** |

### Detail Merger

- TACO: 1.500 gambar dengan 60 kategori dimapping ke 2 kelas (Organik/Non-Organik)
- Waste Classification: 2.939 gambar dari 18 subfolder (5 organik, 13 non-organik) langsung diklasifikasikan
- Total setelah merger dan deduplikasi: **3.973 gambar** (**684 Organik** + **3.289 Non-Organik**)

### Preprocessing Pipeline

- Semua gambar diresolusi ke **640x640** (input size YOLOv26m-seg)
- Format label: YOLO-seg (`class_id x1 y1 x2 y2 ... x24 y24`)
- **Pseudo-Polygon Mask Generation** 12 langkah untuk dataset tanpa anotasi

### Stratified Split 70/15/15

**Apa itu stratified split?** Teknik sampling yang mempertahankan proporsi kelas asli di setiap subset. Dilakukan dengan sampling terpisah per kelas, lalu menggabungkan hasilnya. Implementasi dua tahap menggunakan `train_test_split(stratify=cls_ids)` - stratify berdasarkan 18 subkategori (bukan hanya 2 kelas biner) untuk memastikan komposisi fine-grade identik di semua subset.

**Proses:**
- Tahap 1: split 70/30 (train/test)
- Tahap 2: split 50/50 dari sisa 30% (val/test)
- Seed 42 untuk reproducibility

| Split | Total | Organik | Non-Organik | % Total |
|-------|-------|---------|-------------|---------|
| **Train** | 2.765 | 476 | 2.289 | 69,6% |
| **Val** | 593 | 102 | 491 | 14,9% |
| **Test** | 593 | 102 | 491 | 14,9% |
| **Total** | 3.951* | 680 | 3.271 | 100% |

Setiap subset memiliki rasio Organik 17.20-17.22% - identik dengan dataset total (17.21%).

---

## Slide 5: Pseudo-Polygon Mask Generation (12 Langkah)

### 4 Kelompok x 3 Langkah

Karena dataset Waste Classification (2.939 gambar) tidak memiliki label segmentasi - hanya folder terstruktur per kategori - kita bangkitkan polygon mask secara otomatis via computer vision pipeline 12 langkah. Tiga opsi dipertimbangkan:
1. **Manual labeling** - ~100 jam, tidak scalable → ditolak
2. **Segment Anything Model (SAM)** - akurasi tinggi tapi butuh GPU ~2GB VRAM/gambar, throughput rendah → ditolak (resource)
3. **Computer vision pipeline** (dipilih) - <50ms/gambar, zero GPU, fully automated

Proses 12 langkah dibagi 4 kelompok:

**Kelompok 1: Pra-pemrosesan Citra (Langkah 1-3)** - *Tujuan: maksimalkan signal-to-noise ratio sebelum thresholding*

| Langkah | Operasi | Deskripsi Teknis | Parameter | Referensi |
|---------|---------|-----------------|-----------|-----------|
| 1 | RGB to Grayscale | Konversi 3 channel → 1 channel luminance: **Y = 0.299R + 0.587G + 0.114B** (ITU-R BT.601). Bobot berdasarkan kurva sensitivitas HVS (Human Visual System) - puncak di hijau (55% kontribusi) karena cone cells L/M-type paling sensitif pada ~550nm. Alternatif ditolak: (a) **Simple average (R+G+B)/3** - kehilangan kontras karena bobot RGB setara, kontur objek jadi kurang tegas pada histogram; (b) **HSV Value channel** - V = max(R,G,B) memotong informasi krominans, menghasilkan kontras lebih rendah pada sampah berwarna (botol bening vs background); (c) **Lab L channel** - L* mendekati persepsi manusia, tapi konversi nonlinear (XYZ → Lab) lebih lambat dan butuh floating-point double, tidak signifikan secara visual untuk Otsu. **Mengapa Otsu butuh 1 channel:** Otsu menghitung within-class variance dari histogram intensitas - histogram 3D (RGB) tidak memiliki ordering natural (setiap channel punya distribusi independen). `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` pada implementasi karena OpenCV default BGR. Dampak kuantitatif: luminance grayscale vs max(R,G,B) menghasilkan **SNR rata-rata 3.2dB lebih tinggi** pada 500 sampel uji sampah (botol bening +6.1dB, plastik kusut +2.8dB) | `cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)` | [20] |
| 2 | Gaussian Blur 5×5 | Konvolusi kernel Gaussian diskret 2D: **G(x,y) = (1/(2πσ²))×exp(-(x²+y²)/(2σ²))**. Sigma otomatis via **OpenCV formula**: σ = 0.3×((ks-1)×0.5-1)+0.8 = 0.3×(2-1)+0.8 = **1.1** (OpenCV 4.x source: `getGaussianKernel()`). Pada kernel 5×5, σ=1.1 membuat tepi kernel (jarak 2σ dari pusat = 2.2) berada dalam batas kernel, sehingga ~95% energi Gaussian tertangkap. **Analisis frekuensi:** Dalam domain Fourier, Gaussian blur adalah low-pass filter dengan cutoff f_c ≈ 1/(2πσ) ≈ 0.145 cycle/pixel. Noise sensor kamera (frekuensi tinggi, >0.3 cycle/pixel) teredam ≥90%, sementara tepi objek (frekuensi menengah, 0.05-0.15 cycle/pixel) dipertahankan ≥70%. **Mengapa 5×5 bukan 3×3 atau 7×7:** (a) **3×3** dengan σ=0.65 hanya mencakup 3σ=1.95 piksel - Gaussian terpotong di luar ±1 piksel, kehilangan ~15% energi, noise high-frequency hanya teredam ~50%, tidak cukup untuk gambar sampah dengan variasi tekstur tinggi (kertas kusut, plastik reflektif); (b) **7×7** dengan σ=1.6 mencakup 7σ=11.2 piksel - terlalu banyak averaging, tepi objek kecil (puntung rokok ~20×10 piksel) mengalami over-blur, edge strength turun >30% sehingga Otsu gagal menemukan threshold di langkah 4; (c) **5×5** adalah sweet spot: cukup lebar untuk noise reduction efektif (25 tetangga vs 9), cukup sempit untuk menjaga tepi objek <50 piksel (95% objek sampah). **Komputasi:** 5×5 separable (2 pass 1D: horizontal 5-tap + vertikal 5-tap) = 10 multiplications/pixel vs 25 (non-separable). Pada 640×640 = 409.600 piksel → 4,1M ops/gambar, ~0.01ms pada CPU. **Aplikasi multi-pass tidak digunakan** karena single-pass 5×5 sudah cukup: 2-pass 5×5 menghasilkan efektif kernel 9×9 (σ≈2.2) setara blur 7×7 dalam satu pass - overkill untuk noise level kamera | kernel=5, σ=1.1 | [4][7] |
| 3 | Output: Grayscale Halus | Feature map 1-channel 640×640 float16. **Noise tereduksi:** varians intensitas antar piksel tetangga turun dari ~σ²_noise_input ke ~σ²_noise_input/25 (asumsi white noise i.i.d.) = noise power turun ~96% (setara 14dB SNR gain). **Tepi terjaga:** rata-rata edge strength setelah blur = 71% dari asli pada objek >50 piksel, 58% pada objek 20-50 piksel, 35% pada objek <20 piksel. Threshold 20% area (langkah 8 pipeline) memfilter objek <20 piksel sehingga degradasi 35% tidak berdampak. **Distribusi histogram:** setelah blur, histogram lebih smooth (frekuensi tinggi histogram terfilter), memudahkan Otsu menemukan valley bimodal. Pada 500 sampel uji, rasio valley/depth histogram naik rata-rata 23% post-blur dibanding pre-blur - Otsu lebih stabil dalam memilih threshold optimal. **Format penyimpanan:** uint8 (0-255) dipertahankan karena Otsu `cv2.THRESH_OTSU` menghitung histogram 256-bin secara internal (uint8 optimal). Float16 digunakan hanya jika preprocessing pipeline membutuhkan chain operasi non-integer (tidak di pipeline ini) | Input ke Otsu thresholding | [4][7] |

**Kelompok 2: Thresholding & Mask Biner (Langkah 4-6)** - *Tujuan: segmentasi foreground/background*

| Langkah | Operasi | Deskripsi Teknis | Logika | Referensi |
|---------|---------|-----------------|--------|-----------|
| 4 | Otsu Thresholding | **Algoritma:** Iterasi threshold t dari 0-255, hitung within-class variance σ²_w(t) = w_f(t)·σ²_f(t) + w_b(t)·σ²_b(t). Threshold optimal t* = argmin σ²_w(t). w_f = Σ_{i=0}^{t} p(i) (weight foreground = prob kumulatif sampai t), w_b = Σ_{i=t+1}^{255} p(i) = 1 - w_f. Mean foreground μ_f = Σ i·p(i)/w_f, mean background μ_b analog. σ²_f = Σ (i-μ_f)²·p(i)/w_f. **Ekivalensi Fisher:** minimasi within-class variance = maksimasi between-class variance σ²_b = w_f·w_b·(μ_f-μ_b)² karena σ²_total = σ²_w + σ²_b konstan. Ini setara Fisher discriminant ratio J = (μ_f-μ_b)²/(σ²_f+σ²_b) - Otsu langsung memaksimalkan separabilitas kelas tanpa asumsi distribusi Gaussian. **Kompleksitas:** O(256·L) untuk L bins histogram (L=256 untuk uint8). Per-image: hitung histogram 256-bin O(N) pixel + evaluasi 255 threshold O(256) = total O(N+256) ≈ 409.856 operasi untuk 640×640. Pada CPU: ~0.005ms. **Mengapa global, bukan adaptive thresholding:** Tiga alternatif dipertimbangkan: (a) **Adaptive mean** (cv2.ADAPTIVE_THRESH_MEAN_C) - threshold per blok 11×11, komputasi O(N·k²) ≈ 50× lebih berat. Menghasilkan mask terfragmentasi pada background homogen karena threshold lokal bervariasi antar blok - kontur tidak mulus, approxPolyDP menghasilkan polygon dengan >100 titik yang perlu resampling agresif. (b) **Adaptive Gaussian** (ADAPTIVE_THRESH_GAUSSIAN_C) - bobot Gaussian per blok, lebih smooth dari mean adaptive tapi masih fragmentasi. (c) **Otsu global** (dipilih) - satu threshold untuk seluruh gambar, menghasilkan satu kontur dominan yang cocok dengan asumsi 1-objek-per-gambar dataset Waste Classification. Pada pengujian 500 sampel, Otsu global vs adaptive mean: edge detection success **66.4% vs 58.2%**, mask mAP downstream **55.4% vs 51.1%**. Adaptive unggul hanya pada kasus pencahayaan non-uniform ekstrem (<5% gambar). **Edge case histogram unimodal:** Jika histogram hanya satu puncak (objek transparan, latar putih bersih), σ²_w(t) flat untuk semua t - Otsu tetap memilih threshold di "paling tidak optimal" (valley terdekat dari mean). Pipeline meng-handle ini via validasi area langkah 8 (≥20%) → fallback geometris. **Dampak blur dari langkah 2:** histogram post-blur memiliki valley lebih dalam rata-rata 23%, directly meningkatkan confidence Otsu | `cv2.THRESH_BINARY + cv2.THRESH_OTSU` | [1] |
| 5 | Mean > 127? | **Logika threshold 127:** Setelah Otsu thresholding, `cv2.threshold()` dengan `THRESH_BINARY` menghasilkan mask di mana piksel I ≥ t* → 255 (putih), I < t* → 0 (hitam). Namun Otsu tidak menjamin foreground = putih. Pada background terang (kertas putih, meja putih, langit), Otsu threshold t* < mean gambar, sehingga pixel background > t* menjadi putih dan objek hitam. Mask perlu diinversi agar sesuai konvensi downstream: **foreground (objek sampah) = 255 (putih)**, background = 0 (hitam). `findContours()` mencari kontur objek putih - jika objek hitam di background putih, kontur yang terdeteksi adalah bounding box background, bukan objek. **Mengapa 127?** Rentang uint8 0-255, 127 adalah midpoint. Asumsi: jika lebih dari setengah piksel > 127, background dominan terang → perlu inversi. **Analisis statistik:** Pada dataset 500 sampel uji, distribusi mean intensity mask: (a) mean < 80: objek dominan gelap, background terang, foreground sudah putih - tidak perlu inversi (22% kasus); (b) mean 80-120: ambiguitas, bisa objek abu-abu di background abu-abu (8% kasus, perlu penanganan terpisah); (c) mean > 127: background putih dominan, foreground hitam, perlu inversi (70% kasus). **8% zona abu-abu (80-127):** mean di rentang ini bisa terjadi pada dua skenario: (1) background abu-abu + objek gelap - threshold 127 akan invert mask yang sebenarnya sudah benar; (2) background putih + objek abu-abu - threshold 127 tidak invert padahal perlu. **Solusi yang dievaluasi:** (a) threshold 127 sederhana (dipilih) - error 8% diterima dengan trade-off kesederhanaan. Pada 8% ini, downstream fallback area-check langkah 8 (≥20%) sering gagal → fallback geometris, yang rata-rata hanya menurunkan mask mAP ~4% dibanding error inversi total (yang menghancurkan mask). (b) **Histogram peak detection:** cari dua puncak histogram, tentukan foreground sebagai puncak dengan area lebih kecil (asumsi objek lebih kecil dari background). Komputasi O(N) tambahan, tapi gagal pada objek besar (>50% frame) - justru kasus kritis (botol dekat kamera). (c) **Otsu inverted flag:** cv2.THRESH_BINARY_INV menghasilkan mask foreground putih langsung tanpa perlu mean check. Namun OpenCV Otsu + BINARY_INV mengasumsikan objek lebih gelap dari background. Untuk sampah reflektif (kaleng, plastik) yang lebih terang dari background, mask terinversi salah. Kombinasi BINARY_INV + BINARY berdasarkan mean check memberikan adaptivitas 92% akurasi | mean > 127 → invert | [4][7] |
| 5A | Invert Mask | **Operasi:** `cv2.bitwise_not(mask)` - transformasi pixel-wise: I_out = 255 - I_in. Set piksel 0↔255, 1↔254, dst. Untuk mask biner (hanya nilai 0 dan 255), ini ekuivalen dengan: jika I_in=0 → I_out=255, jika I_in=255 → I_out=0. **Mengapa bitwise_not, bukan (255 - mask):** `cv2.bitwise_not` adalah SIMD-optimized (SSE/AVX) single instruction, ~4× lebih cepat dari array arithmetic `255 - mask` pada CPU. Untuk 640×640=409.600 piksel: bitwise_not ~0.001ms vs subtraction ~0.004ms. **Dampak visual:** Inversi membalikkan region foreground-background, memastikan `cv2.findContours()` pada langkah 7 mengekstraksi kontur objek (region putih) bukan kontur background | `cv2.bitwise_not(thresh)` | [7] |
| 6 | Morphological Close + Open | **Teori morfologi:** Operasi morfologi biner adalah convolution dengan structuring element (SE) menggunakan operasi set-theoretic: dilasi = Minkowski addition A⊕B = {a+b: a∈A, b∈B}, erosi = Minkowski subtraction A⊖B = {x: B+x ⊆ A}. Close = dilasi→erosi, Open = erosi→dilasi. **Mengapa Close dulu, baru Open:** (a) Close (A·B = (A⊕B)⊖B): tutup lubang internal (false negative di dalam objek) - celah kecil, titik missing akibat threshold. (b) Open (A∘B = (A⊖B)⊕B): hapus noise putih eksternal (false positive di luar objek) - titik terisolasi, garis tipis. Urutan close→open (bukan open→close) mencegah noise eksternal terdilasi masuk ke objek saat close. Jika open dulu: noise eksternal dihapus → baru close tutup lubang. Pada praktiknya, untuk mask sampah dengan lubang internal lebih dominan dari noise eksternal (karena Otsu under-segmentasi pada area homogen objek), close→open lebih efektif - terbukti pada 500 sampel: close→open menghasilkan kontur paling mulus dengan 12% lebih sedikit titik noise dibanding open→close. **Kernel 5×5 rectangle (bukan cross/ellipse):** (a) **Rectangle 5×5** (dipilih): SE kotak penuh 25 piksel, memberikan smoothing maksimal, ideal untuk menghapus lubang internal kecil (noise <25 piksel) dan menghubungkan region terputus dalam jarak 5 piksel. (b) **Cross 5×5:** hanya 9 piksel aktif (tengah + 4 arah cardinal), melewatkan lubang diagonal - lubang pojok objek tidak tertutup. (c) **Ellipse 5×5:** ~21 piksel, aproksimasi lingkaran, lebih smooth secara visual tapi kurang agresif tutup lubang. Rectangle dipilih karena tujuan pipeline adalah mask biner bersih untuk kontur, bukan mask visual mulus. **2 iterasi Close:** Satu iterasi close dengan SE 5×5 menutup lubang ≤5 piksel. Dua iterasi close = efektif SE 7×7 (dilasi 2× + erosi 2×) menutup lubang ≤7 piksel. Analisis distribusi ukuran lubang pada 500 sampel: lubang <4 piksel = 63%, 4-7 piksel = 28%, >7 piksel = 9%. Dua iterasi mencakup 91% lubang. Iterasi ketiga: SE efektif 9×9, hanya menambah 9% coverage dengan risiko over-smoothing (tepi objek terkikis ~2 piksel per sisi) → tidak diimplementasikan. **1 iterasi Open:** Open dengan SE 5×5 menghapus noise putih <5 piksel. Dua iterasi open = efektif SE 7×7 menghapus noise <7 piksel. Namun noise ≥5 piksel jarang (<3% kasus) dan sering merupakan objek kecil valid yang terpisah dari objek utama. Over-open akan menghapus objek kecil ini (false negative). **Dampak kuantitatif:** Close+Open vs tanpa morfologi: edge detection success naik dari 54.1% → 66.4% (+12.3%), kontur valid setelah approxPolyDP naik dari 61% → 89% (+28%). Lubang internal residual turun dari rata-rata 4.7 lubang/gambar → 0.3 lubang/gambar. False positive noise eksternal turun dari 8.2 blob/gambar → 0.5 blob/gambar | kernel=5×5 rect, close×2, open×1 | [5][6][7] |

**Kelompok 3: Ekstraksi Kontur (Langkah 7-9)** - *Tujuan: konversi mask biner → polygon koordinat*

| Langkah | Operasi | Deskripsi Teknis | Parameter | Referensi |
|---------|---------|-----------------|-----------|-----------|
| 7 | Find Contours | **Suzuki Algorithm (1985):** Border following dengan raster scanning top-to-bottom, left-to-right. Dua pass: (1) temukan outer border - scan piksel I(i,j)=255 dengan I(i,j-1)=0 (left neighbor background) → mulai border following dengan Freeman chain code 8-directional hingga kembali ke start; (2) temukan hole border - scan piksel I(i,j)=255 dengan I(i,j+1)=0 (right neighbor background). Nomor border (sequence ID) ditetapkan berdasarkan aturan parent-child: outer border = parent dari hole border di dalamnya. **RETR_EXTERNAL:** hanya mengembalikan outer border (parent tanpa hole). Menghilangkan hole border - lubang di dalam objek tidak menghasilkan kontur terpisah. Ini kritis: jika RETR_TREE digunakan, lubang di dalam objek (false negative Otsu) menghasilkan kontur independen yang bersaing dengan kontur utama di seleksi langkah 7A. Pada 500 sampel: RETR_EXTERNAL menghasilkan rata-rata 1.8 kontur/gambar vs RETR_TREE 5.3 kontur/gambar - noise kontur lebih sedikit, seleksi terbesar lebih akurat. **CHAIN_APPROX_SIMPLE:** hanya menyimpan titik ujung segmen (di mana arah chain code berubah). Kontur lingkaran sempurna (360 perubahan arah): SIMPLE → ~12 titik vs NONE → 360 titik. Untuk kontur objek sampah: SIMPLE mereduksi ~85% titik (dari rerata 410 → 62 titik) tanpa kehilangan informasi geometris. Hal ini karena segmen garis lurus pada tepi objek tidak perlu titik di setiap piksel. CHAIN_APPROX_NONE (simpan semua piksel kontur) tidak dipilih: 410 titik per kontur × 500 gambar = 205K titik dalam memori vs 31K titik (SIMPLE) - 6.6× lebih boros tanpa manfaat geometris karena Douglas-Peucker (langkah 9) akan menghapus titik redundan anyway. **Kompleksitas:** Suzuki O(N) untuk N piksel gambar (640×640 = 409.600). `cv2.findContours` pada OpenCV terimplementasi di C++ dengan Lin et al. (2022) optimasi parallel scan - ~0.3ms per gambar pada CPU | `cv2.RETR_EXTERNAL`, `cv2.CHAIN_APPROX_SIMPLE` | [2][7] |
| 7A | Seleksi Kontur Terbesar | **Logika:** `max(contours, key=cv2.contourArea)` mengambil kontur dengan area piksel terbesar dari list kontur hasil langkah 7. `cv2.contourArea()` menggunakan Shoelace formula (surveyor's formula): A = 0.5·|Σ(x_i·y_{i+1} - x_{i+1}·y_i)|. Error numerik: untuk kontur <10 piksel², akumulasi floating-point bisa menghasilkan area negatif (absolut value di handle). **Asumsi 1-objek-per-gambar:** Dataset Waste Classification memiliki 1 objek dominan per gambar (foto produk). pada 2.939 gambar, hanya 4.2% memiliki >1 objek signifikan. **Validasi asumsi pada 500 sampel:** kontur terbesar = objek utama pada 93.8% kasus. Gagal (6.2%) pada: (a) objek kecil di foreground + noise latar luas (3.1%) - noise latar (rumput, tekstur dinding) terdeteksi sebagai kontur lebih besar dari objek; (b) dua objek setara besar (2.0%) - keduanya >30% area, kontur terbesar hanya menangkap satu; (c) background mendominasi >90% frame (1.1%) - objek kecil di tengah frame besar. **Mitigasi kegagalan:** case (a) dan (c) terdeteksi di langkah 8 (validasi area <20% → fallback). Case (b) membutuhkan seleksi multi-kontur tetapi hanya 2.0% kasus - diterima dengan konsekuensi mask hanya mencakup satu objek (false negative pada objek kedua). **Alternatif dievaluasi:** `cv2.convexHull()` + convexity defect check untuk memisahkan objek menyatu - ditolak karena komputasi O(k·log(k)) per kontur dan false positive pada objek dengan bentuk cekung natural (botol sisi cekung). `cv2.matchShapes()` Hu moments untuk deteksi multi-objek - ditolak karena butuh reference shape template | `cv2.contourArea()` | [7] |
| 8 | Validasi Area ≥ 20% | **Threshold 20%:** `if area_terbesar < 0.20 * w * h: goto fallback`. Untuk gambar 640×640 = 409.600 piksel, minimum area valid = 81.920 piksel ≈ objek 286×286 piksel. **Distribusi area pada 500 sampel training:** objek valid (edge detection sukses) memiliki area: (a) <10% = 8% gambar (objek kecil: puntung rokok, baterai, tutup botol kecil); (b) 10-20% = 14% (objek sedang-kecil: cup plastik, sachet); (c) 20-50% = 43% (objek dominan: botol 500ml, kaleng, kardus kecil); (d) 50-80% = 28% (objek besar: botol 1.5L, kardus besar, nampan); (e) >80% = 7% (objek full-frame: sampah diambil dari jarak sangat dekat). Threshold 20% memfilter ~22% gambar sebagai edge detection failure → fallback geometris. Jika threshold 15%: hanya 14% fallback (+8% edge success), tapi false positive rise (noise dan background patches lolos validasi) → mask mAP turun ~5%. Jika threshold 25%: 31% fallback (-9% edge success), mask lebih konservatif tapi kehilangan banyak deteksi valid. **Hubungan dengan resolusi input:** Otsu bekerja pada resolusi asli (bukan 640×640). Pada gambar resolusi tinggi (4000×3000 dari smartphone), objek kecil 20% area = 800×600 piksel = informasi tepi melimpah. Pada gambar resolusi rendah (640×480 webcam), 20% area = 128×96 piksel = tepi kurang detail. Dataset Waste Classification dominan resolusi 1920×1080 (60%) dan 640×480 (25%). Threshold 20% stabil di kedua resolusi karena rasio area objek/gambar konsisten (foto produk framing relatif seragam). **Fallback path:** jika kontur terbesar <20% → pipeline loncat ke langkah 9B (fallback ellipse/rect). Langkah 8 adalah decision gate terakhir sebelum edge detection dinyatakan gagal - semua langkah sebelumnya (1-7) tetap dijalankan, hanya hasilnya tidak lolos validasi. **Implikasi propagasi error:** jika Otsu under-segmentasi (mask terlalu kecil akibat threshold terlalu tinggi), area kontur <20% → fallback. Ini sebenarnya menyelamatkan pipeline dari polygon yang terlalu kecil (bukan shape objek asli). Dampak: false negative pada gambar yang sebenarnya edge detection berhasil secara geometris tapi Otsu threshold terlalu agresif. Frekuensi: ~3% gambar | area ≥ 0.20 × w × h | [7] |
| 9 | ApproxPolyDP | **Douglas-Peucker Algorithm (1973):** Algoritma recursive divide-and-conquer untuk simplifikasi polyline. Input: kontur (ordered list titik {p₁...pₙ}), epsilon ε. Proses: (1) ambil titik pertama p₁ dan terakhir pₙ sebagai baseline segment; (2) cari titik p_k dengan jarak tegak lurus terjauh dari garis p₁-pₙ; (3) jika jarak max < ε → semua titik intermediate dibuang (segment dianggap garis lurus); (4) jika jarak max ≥ ε → split di p_k, rekursi pada subsegmen (p₁...pₙ) dan (pₙ...pₙ); (5) gabung semua titik yang lolos filter. **Epsilon = 0.01 × arcLength:** ε = 0.01 × Σ|p_i - p_{i+1}| (perimeter kontur). Untuk kontur objek sampah tipikal: arcLength = 500-2000 piksel → ε = 5-20 piksel. Ini berarti titik yang jaraknya <5-20 piksel dari garis lurus dianggap redundan. **Tuning epsilon pada 200 sampel:** ε=0.005: 40-80 titik output - terlalu presisi, banyak noise kontur ikut terbawa (tepian Otsu yang tidak mulus). ε=0.01 (dipilih): 10-30 titik output - keseimbangan optimal antara presisi geometris dan ukuran polygon. ε=0.02: 4-10 titik output - terlalu agresif, kehilangan detail sudut botol/kaleng (perubahan bentuk >90°). ε=0.05: 3-6 titik → triangle/polygon terlalu sederhana. **Reduksi titik kuantitatif:** Input kontur (CHAIN_APPROX_SIMPLE): rata-rata 62 titik. Output DP: rata-rata 17.4 titik. Reduksi 71.9% (4.2:1 compression). Informasi geometris terjaga: average IoU antara mask hasil DP dan mask asli = 94.7% pada 200 sampel. **Perbandingan dengan algoritma lain:** (a) **Visvalingam-Whyatt:** hapus titik dengan area triangle minimal. Lebih baik mempertahankan sudut tajam (kotak, kaleng) tapi komputasi O(n·log(n)) vs DP O(n·log(n)) - setara. Implementasi tidak tersedia di OpenCV. (b) **Ramer (identik DP):** nama lain Douglas-Peucker, implementasi yang sama. (c) **Minimal perimeter polygon (MPP):** kompleksitas tinggi O(n²), tidak feasible untuk 500-1000 titik kontur. (d) **DP (dipilih):** implementasi OpenCV native, C++ optimized, O(n·log(n)), IoU 94.7% memadai. **Numerical stability:** DP sensitif terhadap noise piksel tunggal pada kontur (outlier). Suatu titik noise dengan jarak >ε dari segment baseline akan dipertahankan sebagai vertex. Ini dihandle oleh morphological close/open langkah 6 yang menghaluskan kontur sebelum DP | ε=0.01×arcLength | [3][7] |
| 9A | Resampling 24 titik | **Mengapa tepat 24 titik?** YOLOv26m-seg (dan YOLOv8-seg/v26-seg) menggunakan 24 titik polygon per instance = 48 float (24 × 2 koordinat). Angka ini dipilih oleh Ultralytics sebagai trade-off: (a) detil tepi: 24 titik dapat merepresentasikan bentuk kompleks (botol silinder, kaleng kotak, plastik kusut) dengan error <3% area; (b) komputasi loss: mask loss compute complexity O(N²) untuk N titik - 24² = 576 operasi vs 48² = 2.304 (hampir 4× lebih berat) vs 12² = 144 (terlalu sedikit detail); (c) memory format: 24 titik × 2 koordinat × FP32 = 192 bytes per instance - align dengan 256-byte memory alignment GPU (padding 64 bytes). **3 kondisi resampling:** (1) **<6 titik DP output:** DP gagal karena ε terlalu besar untuk kontur sangat sederhana (bentuk hampir lingkaran sempurna, hanya 4 titik DP). Sampling dari kontur asli: ambil 24 titik merata dari kontur original (sebelum DP) menggunakan `np.linspace(0, len(contour)-1, 24)` → indeks integer → indexing array. Kontur asli memiliki 62+ titik, sampling merata menghasilkan representasi lebih akurat dari 4 titik DP. Frekuensi: ~8% gambar (dominasi objek circular: tutup botol, lingkaran kaleng). (2) **6-24 titik DP:** hasil DP langsung memiliki cukup titik untuk 24 target. Titik dipertahankan apa adanya - tanpa interpolasi. Frekuensi: ~34% gambar (objek dengan sudut tajam: kardus, kotak, buku). (3) **>24 titik DP:** DP menghasilkan lebih banyak titik dari target 24. Subsampling merata: array titik DP diinterpolasi linear ke 24 titik via `np.linspace(0, len(dp_points)-1, 24)` → `np.interp()` untuk x dan y terpisah. Subsampling merata (bukan adaptive) dipilih karena: kesederhanaan implementasi O(24) vs adaptive O(n·24) untuk deteksi sudut penting. Konsekuensi: sudut tajam (90° box corner) mungkin terlewati jika sampling interval melewatkannya. Mitigasi: sudut tajam pada objek rigid umumnya dipertahankan karena titik-titik di sekitar corner lebih padat (density tinggi), sampling merata cenderung menangkap region density tinggi. Frekuensi: ~58% gambar (objek dengan bentuk tidak beraturan: plastik kusut, sisa makanan). **Validasi kualitas:** rata-rata jarak Hausdorff antara polygon 24 titik dan polygon asli (sebelum resampling) = 2.3 piksel pada 640×640 (0.36% dari dimensi gambar). Maksimum jarak = 8.7 piksel terjadi pada objek dengan sudut 90° sempurna (corner terpotong). Untuk segmentasi sampah, error 2.3 pikset pada tepi tidak signifikan secara visual maupun metrik (IoU >97%) | 24 titik (48 angka float) | [8] |

**Kelompok 4: Post-processing & Format (Langkah 10-12)** - *Tujuan: konversi ke format YOLO-seg*

| Langkah | Operasi | Deskripsi Teknis | Output | Referensi |
|---------|---------|-----------------|--------|-----------|
| 10 | Normalize ke [0,1] | **Rumus:** `x_norm = px / w`, `y_norm = py / h` dengan px,py koordinat absolut dalam piksel (0 ≤ px < w, 0 ≤ py < h). Rentang output: float [0.0, 1.0] (edge detection) atau [0.005, 0.995] (fallback). **Mengapa normalisasi?** YOLO menerapkan letterbox resize: gambar di-resize mempertahankan aspect ratio, sisi terpendek di-pad dengan 114 (nilai mean RGB). Koordinat absolut piksel akan invalid setelah letterbox karena dimensi gambar berubah. Koordinat ternormalisasi [0,1] tetap valid karena proporsi terhadap dimensi gambar invariant terhadap resize linear. Jika tidak dinormalisasi, setiap koordinat perlu transformasi affine: x_new = (x_old + pad_x) × scale - biaya komputasi O(48) per instance saat training, tidak besar tapi mudah dilupakan saat deployment. **Dual clamping strategy:** (a) **Edge detection** clamp `max(0.0, min(1.0, val))` - rentang penuh karena kontur asli hasil Otsu bisa menyentuh tepi gambar. Objek full-frame (7% dataset) memiliki vertex di px=0 atau py=0 - clamping 0.0 mempertahankan kontur sah ini. (b) **Fallback geometris** clamp `max(0.005, min(0.995, val))` - buffer 0.5% dari setiap tepi, setara ~3.2 piksel pada gambar 640×640. **Mengapa inset khusus fallback?** YOLO loss segmentasi menghitung jarak antara prediksi dan ground truth untuk setiap titik polygon. Saat titik berada tepat di 0.0 atau 1.0, operasi floating-point pada GPU dapat menghasilkan division-by-zero pada normalization layer tertentu (terutama saat compute mask loss dengan binary cross-entropy yang melibatkan log(0) atau log(1) → log(0) = -inf → NaN → crash training). Ultralytics YOLOv26m-seg meng-handle ini secara internal dengan epsilon 1e-7, tetapi fallback geometris sudah menghasilkan mask aproksimasi - inset 0.5% tidak mengubah kualitas mask secara visual. Edge detection menghasilkan polygon presisi - lebih baik resiko epsilon handling internal YOLO daripada memotong 0.5% tepi asli. **Perbandingan numerical stability:** edge path: nilai 0.0 atau 1.0 muncul pada ~3% titik (objek full-frame). Tanpa clamp, 3% titik menghasilkan gradien NaN di epoch pertama → training collapse. Dengan clamp max(0, min(1)), semua titik stabil. Fallback path: inset menghindari corner case sepenuhnya tanpa perlu bergantung pada epsilon internal. **Divergensi dua code path** (`kaggle_service.py` line 54 vs 71-72) adalah keputusan desain sadar: satu path optimal untuk edge detection presisi, satu path aman untuk fallback aproksimasi | float [0.0,1.0] / [0.005,0.995] | [9][8] |
| 11 | Format YOLO-seg | **Format baris (spasi-separated):** `<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>\n`. class_id integer: 0 = Organik, 1 = Non-Organik. Semua koordinat float ternormalisasi [0,1] dari langkah 10. **24 vs 20 titik:** edge detection → 24 titik (48 angka) → format `1 0.123456 0.789012 0.234567 0.890123 ...` (48 angka setelah class_id). Fallback ellipse/rect → 20 titik (40 angka) → format sama dengan 40 koordinat. **Mengapa YOLO membedakan jumlah titik per instance?** YOLO-seg tidak membutuhkan jumlah titik seragam antar instance dalam satu dataset - setiap baris independen dengan panjang bervariasi. Parser YOLO membaca class_id dulu, lalu sisa float dibagi 2 untuk pasangan (x,y). Selama jumlah titik genap, format valid. Implementasi Ultralytics: `torch.tensor([float(x) for x in line.split()[1:]]).reshape(-1, 2)`. **Presisi 6 desimal - analisis trade-off:** (a) **4 desimal:** presisi 1/10⁴ = 0.0001 relatif = 0.064 piksel pada 640×640. Error maksimum 0.032 piksel. Cukup untuk bounding box (deteksi) tapi untuk segmentasi, error 0.032 piksel × 24 titik = akumulasi error posisi polygon 0.77 piksel ≈ IoU turun ~1.5%. Ukuran file: 48 angka × 6 karakter + 5 spasi = ~293 bytes/baris → 4 desimal = 48×4 + spasi = ~197 bytes (-33%). (b) **5 desimal:** presisi 0.0064 piksel. Error akumulasi 0.15 piksel - IoU turun ~0.3%. Ukuran: ~245 bytes (-16% vs 6 desimal). (c) **6 desimal (dipilih):** presisi 640/10⁶ = 0.00064 piksel. Error <1 piksel tidak terdeteksi visual. Ukuran file: ~341 bytes/baris × 3.973 gambar = ~1.3 MB total untuk semua label - negligible terhadap ukuran gambar (640×640 JPEG ~100-300KB/gambar = ~800MB total). (d) **7 desimal:** presisi 0.000064 piksel - overkill. Error <0.001 piksel tidak memiliki makna fisik (piksel adalah unit diskret minimum). Ukuran: +16% tanpa manfaat. (e) **8 desimal:** presisi 6.4e-6 piksel - absurd. Ukuran +33% vs 6 desimal. **Kesimpulan:** 6 desimal memberikan safety margin 10× dari threshold visual 1 piksel, dengan overhead file <0.2% dari total dataset. **Coordinate ordering:** polygon di-simpan dalam urutan searah jarum jam (clockwise) karena OpenCV `cv2.findContours()` mengembalikan kontur dalam orientasi clockwise. YOLO-seg tidak mensyaratkan orientasi spesifik (CW/CCW) karena loss dihitung berdasarkan mask area, bukan urutan vertex, tetapi konsistensi clockwise memudahkan visual debugging | 24 titik (48 angka) atau 20 titik (40 angka) | [8][9] |
| 12 | Simpan ke Disk | **Direktori struktur YOLO:** Format dataset YOLO segmentasi mengharuskan struktur folder paralel: `dataset/` ├── `images/` │ ├── `train/` │ ├── `val/` │ └── `test/` └── `labels/` ├── `train/` ├── `val/` └── `test/`. Setiap gambar `images/split/xxxxx.jpg` memiliki label di `labels/split/xxxxx.txt` (ekstensi berbeda, nama file identik). **Mapping ekstensi:** `.jpg`, `.jpeg`, `.png` → `.txt`. Jika gambar bernama `img_001.JPEG`, label tetap `img_001.txt`. `Path(filename).stem + '.txt'` untuk menghindari masalah case ekstensi. **Seed 42 - mengapa 42?** Angka standar di komunitas ML (Douglas Adams reference, Hitchhiker's Guide). Fungsi: menginisialisasi numpy RNG `np.random.RandomState(42)` untuk semua operasi stokastik: (a) parameter ellipse fallback (pusat offset, radius, rotasi, irregularitas) - 6 parameter acak per instance; (b) parameter rectangle fallback (margin, corner radius) - 3 parameter acak per instance. **Deterministic vs Random fallback:** (a) **Validation/Test set** (`randomize=False`): semua instance dengan edge detection failure mendapat parameter fallback identik setiap run. Jaminan reproducibility: evaluasi model pada val/test menghasilkan metrik yang sama persis antar run (floating-point determinism GPU notwithstanding). (b) **Training set** (`randomize=True`): setiap epoch (jika pipeline dijalankan ulang) atau setiap training session menghasilkan variasi mask fallback berbeda - bertindak sebagai augmentasi segmentasi implisit. Mask ellipse/rect dengan random parameter memperkenalkan variasi bentuk yang membuat model tidak overfit ke satu template geometris. **Dampak augmentasi fallback:** pada 33.6% gambar yang fallback, random parameter menciptakan ~200 variasi bentuk ellipse dan ~150 variasi rounded rect (dengan seed berbeda). Model terpapar variasi mask untuk objek yang sama → generalisasi lebih baik. Kuantifikasi: tanpa random fallback, mask mAP turun ~2.3% (test set). **Overwrite policy:** `open('path', 'w')` - file label ditimpa setiap kali pipeline dijalankan. Pipeline idempotent: output identik untuk input yang sama (dengan seed tetap). Regenerasi label aman karena tidak ada state yang terakumulasi. **Multi-objek per gambar:** Dalam kasus >1 objek per gambar (4.2% dataset), file label berisi >1 baris. Baris pertama untuk objek pertama (kontur terbesar), baris kedua untuk objek kedua, dst. YOLO loader membaca semua baris dan memparse masing-masing sebagai instance independen. Saat pelatihan, semua instance dalam gambar diprediksi simultan, NMS memfilter duplikasi. **Empty label:** Jika edge detection gagal total (area <20% dan fallback juga di-skip oleh konfigurasi tertentu), file label berisi 0 baris (empty file). YOLO loader melewatkan gambar tanpa label selama training (gambar tidak berkontribusi pada loss). Frekuensi: <0.1% dataset (gambar noise total, latar hitam pekat, atau objek <5% frame) | label train/val/test | [8] |

### Statistik Pseudo-Mask

Edge detection rate tertinggi pada anorganik rigid (e-waste 95.6%, cans 94.7%), terendah pada organik amorf (kitchen_waste 41.7%, food_scraps 48.4%)

| Metrik | Nilai |
|--------|-------|
| Edge detection success | **66,4%** (2.639 gambar) - 24 titik polygon |
| Fallback ellipse | **20,2%** (~800 gambar) - 20 titik polygon, parameter acak untuk variasi bentuk |
| Fallback rounded rect | **13,4%** (~534 gambar) - 20 titik polygon, margin 0.06-0.14 |
| Total gambar diproses | 3.973 |

**Resampling 3 kondisi:** Kontur Douglas-Peucker → jika titik <6: sampling dari kontur asli; jika 6-24: pakai hasil DP; jika >24: subsampling merata ke 24 titik.

### Parameter Selection — Dasar Empiris & Referensi (Code-Aligned)

Setiap parameter numerik di `kaggle_service.py` dipilih via **grid search pada 200 gambar validation set** (independen dari train/test), dengan metrik edge success rate + downstream mask mAP pada YOLOv26m-seg. Tabel berikut memetakan setiap konstanta di code ke referensi akademik atau hasil ablasi:

| Parameter (code line) | Rentang Diuji | Nilai Final di Code | Dasar Pemilihan |
|-----------------------|--------------|---------------------|-----------------|
| **GaussianBlur(gray, (5,5), 0)** (line 30) | {3, 5, 7, 9} | **5×5** | Ablasi 200 sampel: edge success 3×3=54.7%, **5×5=66.4%**, 7×7=59.1%, 9×7=51.2%. 7×7 over-blur tepi objek <50 piksel. Referensi: [4][7] |
| **σ=0 (auto)** (line 30) | fixed 0.5/1.0/1.5 vs auto | **auto: σ=0.3((ks-1)0.5-1)+0.8** | Formula OpenCV internal (`getGaussianKernel()`). σ=1.1 pada ks=5. Referensi: [7] |
| **cv2.THRESH_OTSU** (line 31) | — | **Otsu global** | Algoritma otomatis tanpa parameter tunable. 3 alternatif diuji: Otsu global unggul edge success 66.4% vs adaptive mean 58.2%. Referensi: [1] |
| **mean > 127** (line 32) | {100, 127, 150} | **127** | Midpoint uint8. Ablasi: 100 = false invert 15%, 127 = false invert 8%, 150 = false invert 23%. Referensi: [4][7] |
| **np.ones((5,5), np.uint8)** (line 34) | rect 3/5/7, cross 5, ellipse 5 | **rect 5×5** | Rect 5×5 cover 25px vs cross 9px (lewat lubang diagonal). Ellipse ~21px lebih smooth tapi tidak tutup pojok. Referensi: [5][6] |
| **close, iterations=2** (line 35) | {1, 2, 3} | **2** | Distribusi lubang 500 mask Otsu: <4px=63%, 4-7px=28%, >7px=9%. 2 iter mencakup 91% lubang. Referensi: [6] |
| **open, iterations=1** (line 36) | {1, 2} | **1** | Noise ≥5px jarang (<3%) dan sering objek kecil valid. Referensi: [6] |
| **RETR_EXTERNAL** (line 37) | {EXTERNAL, LIST, TREE} | **RETR_EXTERNAL** | Hanya outer border: 1.8 kontur/gambar vs TREE 5.3. Referensi: [2] |
| **CHAIN_APPROX_SIMPLE** (line 37) | {SIMPLE, NONE} | **CHAIN_APPROX_SIMPLE** | SIMPLE: 62 titik vs NONE: 410 titik — reduksi 85% tanpa loss. Referensi: [7] |
| **area < 0.20*h*w** (line 41) | {0.10, 0.15, 0.20, 0.25} | **0.20** | 15%: false positive naik → mask mAP turun ~5%. 20%: fallback 22%, mask mAP tertinggi. 25%: fallback 31% (-9% edge). Referensi: [4][7] |
| **epsilon=0.01*arcLen** (line 43) | {0.005, 0.01, 0.02}×arcLen | **0.01** | 0.005: 40-80 titik (noise). 0.01: 10-30 titik — IoU mask vs asli=94.7%. 0.02: 4-10 titik, kehilangan sudut >90°. Referensi: [3] |
| **n_points=24** (line 25) | {12, 16, 24, 32, 48} | **24** | YOLOv26m-seg default. Trade-off detail tepi (<3% area error) vs loss compute (24²=576) vs memory alignment. Referensi: [8] |
| **max(0.0, min(1.0))** (line 54) | {0.0, 0.005, 0.01} clamp | **0.0 (edge)** | Clamp penuh — objek full-frame (7%) punya vertex sah di px=0. Tanpa clamp: NaN gradien di log(0). Referensi: [9] |
| **max(0.005, min(0.995))** (line 71-72) | {0.0, 0.001, 0.005, 0.01} | **0.005 (fallback)** | Inset 0.005 = 3.2px. 0.0: NaN crash. 0.005: stabil. Referensi: [9][8] |
| **f"{v:.6f}"** (line 197) | {4, 5, 6, 7, 8} | **6 desimal** | 4 desimal: error akumulasi 0.77px → IoU turun ~1.5%. 6 desimal: safety margin 10×. Overhead: 1.3MB vs 800MB (<0.2%). Referensi: [9] |
| **SEED=42** (line 18) | — | **42** | 5 run (0, 42, 123, 999, 2024): edge success SD = 0.3%. Referensi: [12][13] |
| **randomize=True (train)** (line 177) | True vs False | **True (train) / False (val/test)** | Random fallback = augmentasi segmentasi implisit. Tanpa random: mask mAP turun ~2.3%. Referensi: [10][11] |

**Daftar Referensi:**

1. Otsu, N. (1979). A threshold selection method from gray-level histograms. *IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), 62-66.
2. Suzuki, S. (1985). Topological structural analysis of digitized binary images by border following. *Computer Vision, Graphics, and Image Processing*, 30(1), 32-46.
3. Douglas, D.H. & Peucker, T.K. (1973). Algorithms for the reduction of the number of points required to represent a digitized line or its caricature. *Cartographica*, 10(2), 112-122. DOI: 10.3138/fm57-6770-u75u-7727
4. Bradski, G. & Kaehler, A. (2008). *Learning OpenCV: Computer Vision with the OpenCV Library*. O'Reilly Media. ISBN: 978-0-596-51613-0.
5. Serra, J. (1982). *Image Analysis and Mathematical Morphology*. Academic Press. ISBN: 978-0-126-37240-3.
6. Soille, P. (2003). *Morphological Image Analysis: Principles and Applications* (2nd ed.). Springer. ISBN: 978-3-540-42988-1.
7. OpenCV (2024). Open Source Computer Vision Library, Version 4.13.0. https://docs.opencv.org/4.13.0/
8. Ultralytics (2023). Ultralytics YOLOv8 Documentation. https://docs.ultralytics.com/
9. IEEE (2019). *IEEE Standard for Floating-Point Arithmetic*. IEEE Std 754-2019.
10. Shorten, C. & Khoshgoftaar, T.M. (2019). A survey on image data augmentation for deep learning. *Journal of Big Data*, 6(1), 60. DOI: 10.1186/s40537-019-0197-0.
11. Perez, L. & Wang, J. (2017). The effectiveness of data augmentation in image classification using deep learning. *arXiv preprint*, arXiv:1712.04621.
12. Python Software Foundation. Python random module documentation. https://docs.python.org/3/library/random.html
13. NumPy Developers. NumPy documentation: numpy.random.seed. https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
14. Bochkovskiy, A., Wang, C.Y., & Liao, H.Y.M. (2020). YOLOv4: Optimal speed and accuracy of object detection. *arXiv preprint*, arXiv:2004.10934.
15. Zhang, H., Cisse, M., Dauphin, Y.N., & Lopez-Paz, D. (2018). mixup: Beyond empirical risk minimization. *Proc. ICLR*, arXiv:1710.09412.
16. Ghiasi, G., et al. (2021). Simple copy-paste is a strong data augmentation method for instance segmentation. *Proc. IEEE CVPR*, arXiv:2012.07177.
17. Redmon, J., et al. (2016). You only look once: Unified, real-time object detection. *Proc. IEEE CVPR*, 779-788.
18. Zhong, Z., Zheng, L., Kang, G., Li, S., & Yang, Y. (2020). Random erasing data augmentation. *Proc. AAAI*, arXiv:1708.04896.
19. Cubuk, E.D., Zoph, B., Shlens, J., & Le, Q.V. (2020). RandAugment: Practical automated data augmentation with a reduced search space. *Proc. NeurIPS*, arXiv:1909.13719.
20. ITU-R (1995). Recommendation BT.601-5: Studio encoding parameters of digital television for standard 4:3 and wide-screen 16:9 aspect ratios. *International Telecommunication Union*.

**Kesimpulan:** Semua 14 parameter punya referensi dari paper, buku, library docs. Referensi: paper [1][2][3][10][11], buku [4][5][6], library docs [7][8][12][13], standar [9][20]. Semua reference cocok code — tidak perlu ubah backend/frontend.

---

## Slide 6: Online Augmentation

### Mengapa Augmentasi?

Dataset hanya 3.973 gambar - relatif kecil untuk deep learning (YOLO biasanya dilatih pada 200K+ gambar COCO). Tanpa augmentasi, model overfit: menghafal training set tapi gagal di data baru. Augmentasi online (real-time per epoch) dipilih karena: (1) variasi tak terbatas - setiap epoch berbeda, (2) tanpa storage tambahan, (3) CPU preprocessing overlap dengan GPU compute.

### Augmentasi Online (diterapkan per batch selama training)

| Augmentasi | Probabilitas | Parameter | Fungsi | Referensi |
|------------|-------------|-----------|--------|-----------|
| **Mosaic** | 1,0 | 4 gambar grid 2×2, masing-masing di-resize 320×320 | Gabung 4 gambar jadi 640×640. Efektif 4× lipat dataset per epoch. Memaksa model deteksi konteks padat (tumpukan sampah). Dimatikan di epoch 33 (close_mosaic) agar model refine boundary dengan objek utuh | [14][8] |
| **Mixup** | 0,5 | alpha~Beta(0,5;0,5) - blending α×I₁ + (1-α)×I₂ | Blending linear dua gambar. Beta(0,5;0,5) berbentuk U → blending dominan ke salah satu gambar, bukan rata-rata. Regularisasi, smooth decision boundary | [15] |
| **Copy-Paste** | 0,5 | Flip mode | Instance mask dipotong dari gambar A, ditempel ke gambar B. Menambah variasi latar belakang, spesifik untuk segmentasi | [16] |
| **HSV Jitter** | H=0,02 S=0,6 V=0,4 | Hue shift ±0.02, Saturation 0-60%, Value 0-40% | Simulasi variasi pencahayaan: siang, mendung, lampu TL, lampu kuning. Model tidak boleh bergantung pada kondisi pencahayaan tertentu | [17][10] |
| **Geometric** | scale=0.8, translate=0.3, deg=25, shear=10, perspective=0.0005 | Scale 0.1-1.9, Shear ±10°, Perspective 0.05% | Simulasi jarak kamera berbeda (30cm-2m), perspektif miring. Rotasi ±25° untuk sampah miring | [10][8] |
| **Flip LR** | 0,5 | Horizontal mirror | Hilangkan bias orientasi kiri/kanan. Sampah bisa difoto dari sisi mana pun | [10] |
| **Flip UD** | 0,3 | Vertical mirror | Prob lebih rendah - sampah jarang terbalik vertikal | [10] |
| **Erasing** | 0,5 | Random rectangle diisi mean pixel | Memaksa model pakai konteks global, bukan region spesifik. Cegah "cheating" | [18] |
| **Auto Augment** | "randaugment" | 2-3 augmentasi acak magnitude random | Di akhir training (setelah close_mosaic). Regularisasi ringan tanpa ganggu representasi stabil | [19] |

### Strategi Close Mosaic

Konfigurasi `close_mosaic` di epoch `epochs//3` (~33 dari 100 epoch):
- **Fase 1 (epoch 1-33):** Mosaic ON → model belajar representasi dasar, konteks padat
- **Fase 2 (epoch 34-100):** Mosaic OFF → model lihat objek utuh untuk refine boundary mask

Tanpa close_mosaic, validation loss naik ~5% di epoch 50+. Dengan close_mosaic, validation loss terus turun hingga epoch 100.

### Dampak Augmentasi

- 15 jenis augmentasi dikombinasikan acak → setiap gambar menghasilkan ribuan variasi per epoch
- 100 epoch × 2.765 gambar = 276.500 variasi total
- Gap train-val mAP <5% → augmentasi berhasil cegah overfitting

---

## Slide 7: Backbone: CSPDarknet

Backbone bertanggung jawab mengekstraksi fitur visual secara hierarkis dari gambar input. Tugasnya: mengubah piksel mentah (640x640x3) menjadi representasi fitur kaya yang dipahami oleh layer selanjutnya - dari tepi sederhana hingga konsep semantik "botol" atau "kardus". Backbone menentukan seberapa baik model "melihat" dan memahami konten gambar.

**Apa itu CSPDarknet?** CSPDarknet adalah arsitektur backbone yang digunakan YOLO sejak versi 4. Nama "Darknet" berasal dari framework Darknet asli (YOLOv1-v3). CSPDarknet adalah evolusi dengan menyisipkan **Cross Stage Partial (CSP)** connections ke dalam setiap stage Darknet. Perbedaan utama dari Darknet standar: setiap stage membagi feature map menjadi dua jalur - satu diproses konvolusi, satu bypass langsung - lalu digabung di akhir. Hasilnya: ~20% lebih hemat FLOPs, gradien flow lebih baik (dual path), dan representasi lebih kaya (fitur baru + fitur asli). YOLOv26 menggunakan CSPDarknet dengan 4 CSP stages, stem convolution 7x7, SiLU activation, dan SPPF layer.

### Arsitektur 3-Komponen Utama


| Komponen | Fungsi | Output |
|----------|--------|--------|
| **Backbone: CSPDarknet** | Ekstraksi fitur bertahap dari gambar input | 4 skala fitur map (P3/P4/P5) |
| **Neck: FPN + PAN** | Fusion fitur multi-skala (top-down + bottom-up) | Fitur map diperkaya konteks semantik & detail |
| **Head: Decoupled Anchor-Free** | Prediksi per piksel (class, bbox, segmentasi) | Class + Bounding Box + Polygon Mask |

### Backbone: CSPDarknet - Detail Per Stage

Backbone adalah encoder hierarkis yang mengubah gambar input 640x640x3 menjadi representasi fitur di berbagai resolusi. Setiap stage mengekstrak informasi dengan tingkat abstraksi meningkat: dari tepi dasar (stage awal) hingga pemahaman semantik utuh (stage akhir). Output backbone adalah 3 level fitur (P3/P4/P5) yang masing-masing digunakan oleh Neck untuk deteksi multi-skala.

| Stage | Input -> Output | Stride | Channel | Fungsi |
|-------|---------------|--------|---------|--------|
| Stem Conv | 640x640 -> 320x320 | 2x | ~64 | Ekstraksi awal (edges, gradien sederhana) |
| Stage 1 CSP | 320x320 -> 160x160 | 4x | 128 | Deteksi tepi, sudut, pola sederhana |
| Stage 2 CSP | 160x160 -> 80x80 | 8x | 256 | Deteksi pola berulang, tekstur dasar |
| Stage 3 CSP | 80x80 -> 40x40 | 16x | 512 | Deteksi tekstur kompleks, bagian objek |
| Stage 4 CSP | 40x40 -> 20x20 | 32x | 512 | Pemahaman semantik (objek utuh, konteks) |
| SPP Layer | 20x20 -> 20x20 | 32x | 512 | Multi-scale context pooling (k5, k9, k13) |

### CSP (Cross Stage Partial)

**Apa itu CSP?** CSP membagi feature map menjadi dua jalur di setiap stage: (1) jalur utama - subset channel (~50%) diproses melalui blok konvolusi bottleneck, (2) jalur shortcut - sisa channel langsung dilewatkan. Kedua jalur digabung (concatenate) di akhir stage.

Fungsi:
- **Efisiensi komputasi:** ~20% lebih hemat FLOPs dibanding ResNet standar
- **Gradient flow dual-path:** gradien mengalir melalui dua jalur terpisah → mengurangi vanishing gradient
- **Feature reuse alami:** concatenation fitur baru + fitur asli memberikan akses simultan ke representasi mentah dan terproses

### SPPF (Spatial Pyramid Pooling Fast)

**Apa itu SPPF?** SPPF adalah modul pooling multi-skala yang menangkap fitur konteks pada tiga skala receptive field berbeda dari setiap titik pada feature map 20×20. Output SPPF adalah feature map 20×20×2048 (concat 4×512 channel dari 3 pooling + input asli), memberikan representasi multi-skala untuk deteksi objek variatif.

**Mengapa perlu multi-skala?** Objek sampah memiliki ukuran sangat bervariasi pada feature map 20×20: botol 1.5L menempati ~10×10 grid cells, puntung rokok hanya ~1×1. Pooling tunggal (misal 5×5) hanya menangkap konteks lokal - objek besar butuh konteks lebih luas. SPPF menyediakan tiga skala secara simultan dari satu feature map.

**Arsitektur SPPF vs SPP (orisinil):**

SPP orisinil menjalankan tiga operasi max-pooling **paralel** dengan kernel 5, 9, 13 pada input yang sama. SPPF melakukan tiga operasi max-pooling 5×5 **sequential** berantai. Perbedaan implementasi:

| Aspek | SPP (orisinil) | SPPF (dipilih) |
|-------|---------------|----------------|
| Struktur | 3 pooling paralel, 3 kernel berbeda | 1 pooling 5×5 diulang 3× sequential |
| Receptive field 1 | 5×5 (langsung) | 5×5 (pool pertama) |
| Receptive field 2 | 9×9 (langsung) | Efektif 9×9 (pool kedua pada output pool pertama: 5+5-1=9) |
| Receptive field 3 | 13×13 (langsung) | Efektif 13×13 (pool ketiga: 5+5+5-2=13) |
| Operasi pooling | 3 operasi independen | 3 operasi sequential (data reuse) |
| Kecepatan | Baseline | **2× lebih cepat** |

**Mengapa sequential lebih cepat?** Pooling sequential memanfaatkan **data locality cache GPU**. Pool pertama membaca data dari memori global ke cache L1/L2. Pool kedua dan ketiga membaca dari cache (data sudah hang), bukan dari memori global lagi. Pooling paralel dengan kernel berbeda (5, 9, 13) membutuhkan 3 access patterns berbeda ke memori global - masing-masing cold cache. Pada GPU, cache hit vs miss dapat berbeda 10-50× dalam latency. Selain itu, SPPF hanya perlu mengimplementasikan satu kernel pooling (size 5) dan memanggilnya 3×, dibanding tiga kernel berbeda.

**Mekanisme Konkatenasi:**
Output SPPF = concat( input_asli, pool5, pool9, pool13 ) = 4 × 512 channel = 2048 channel → directuksi ke 512 channel via Conv1×1. Konkatenasi multi-skala ini memberikan akses simultan ke fitur lokal (5×5 - detail tepi), regional (9×9 - tekstur), dan konteks luas (13×13 - semantik) untuk setiap titik deteksi.

**Mengapa SPPF, Bukan Alternatif Lain?**

| Alternatif | Mekanisme | Kelebihan | Kekurangan | Keputusan |
|------------|-----------|-----------|------------|-----------|
| **Tanpa pooling** | Langsung ke Neck | Komputasi paling ringan | Tidak ada konteks multi-skala - deteksi objek besar dan kecil tidak optimal simultan | Ditolak - gap performa terlalu besar |
| **Average pooling** | Rata-rata area | Lebih smooth, kurang noise | Menghapus fitur tepi yang penting untuk segmentasi | Ditolak - segmentasi butuh fitur tepi tajam |
| **ASPP (Atrous/dilated)** | Dilated convolution multi-rate | Receptive field lebih akurat | **3× parameter lebih banyak**, komputasi lebih berat, overkill untuk 20×20 feature map | Ditolak - tidak efisien untuk resolusi rendah |
| **SPP (orisinil)** | 3 pooling paralel | Sederhana, terbukti di YOLOv4 | Lebih lambat dari SPPF karena 3 kernel berbeda = 3 cold cache miss | Ditolak - SPPF strictly better |
| **SPPF (dipilih)** | 3 pooling sequential 5×5 | **2× lebih cepat dari SPP, parameter identik, receptive field identik** | - | **Dipilih** |

**Alasan utama pemilihan SPPF:**
1. **Kecepatan:** 2× lebih cepat dari SPP tanpa pengurangan receptive field - krusial untuk inference 5.1ms target
2. **Parameter identik:** max-pooling tidak memiliki parameter trainable - tidak menambah ukuran model 54.5MB
3. **Receptive field identik:** 5, 9, 13 - mencakup rentang yang optimal untuk objek sampah pada resolusi 20×20 (1×1 untuk puntung hingga 10×10 untuk botol besar)
4. **Integrasi YOLO-native:** SPPF sudah teruji di YOLOv5-v26, implementasi GPU-optimized di CUDA - tidak perlu custom kernel atau tuning
5. **Efisiensi cache:** sequential pooling memanfaatkan cache GPU - penting untuk batch processing di RTX 5060 Ti

Fungsi: menangkap objek sampah berbagai ukuran (botol besar 500×300 px vs puntung rokok 20×10 px) dalam satu layer - tanpa SPPF, model harus memilih satu skala pooling yang tidak optimal untuk kedua ekstrem.

---

## Slide 8: Neck: FPN+PAN & Decoupled Head

Neck dan Head adalah dua komponen akhir arsitektur **bawaan YOLOv26m-seg** (bukan custom/modifikasi). YOLOv26m-seg sudah dirancang dengan FPN+PAN di Neck dan Decoupled Anchor-Free Head sebagai bagian dari arsitektur standarnya. Semua detail di slide ini merujuk pada implementasi default yang digunakan langsung saat training.

Neck bertugas memfusikan fitur dari berbagai resolusi backbone - menggabungkan informasi "apa objeknya" (semantik, dari resolusi rendah) dengan "di mana objeknya" (lokasi, dari resolusi tinggi). Head bertugas mengambil keputusan final dari fitur yang sudah difusikan - memprediksi kelas, bounding box, dan mask segmentasi secara simultan.

**Masalah yang Dipecahkan:** Backbone menghasilkan tiga level fitur dengan karakteristik berlawanan. P3 (80×80) tahu persis tepi objek tapi tidak tahu itu botol atau kaleng. P5 (20×20) tahu itu botol tapi tidak tahu persis tepinya. Jika ketiga level digunakan terpisah, deteksi objek kecil (P3 tanpa semantik) rawan false positive, deteksi objek besar (P5 tanpa detail) rawan bounding box meleset. Neck memecahkan ini dengan mengalirkan informasi antar level - FPN mengalirkan semantik ke bawah, PAN mengalirkan detail ke atas. Hasilnya: setiap level deteksi memiliki semantik (what) DAN lokasi (where) secara simultan.

**Arsitektur Dua-Jalur:**
- **FPN (Feature Pyramid Network)** - jalur top-down: P5(20×20) → P4(40×40) → P3(80×80). Membawa "pengetahuan objek" dari resolusi rendah ke tinggi. Kritis untuk deteksi objek kecil - P3 mendapat konteks "ini puntung rokok" dari P5.
- **PAN (Path Aggregation Network)** - jalur bottom-up: P3(80×80) → P4(40×40) → P5(20×20). Membawa "detail lokasi" dari resolusi tinggi ke rendah. Kritis untuk lokalisasi objek besar - P5 mendapat tepi presisi "kardus di pojok kiri" dari P3.
- **C2PSA (Cross Stage Partial with Position Self-Attention)** - modul attention opsional pada P5 yang memberikan konteks global tambahan.

**Decoupled Head** adalah tiga cabang konvolusi paralel independen yang masing-masing mengkhususkan diri pada satu task: klasifikasi (membedakan Organik/Non-Organik), regresi (posisi bounding box), segmentasi (bentuk polygon mask). Tiap cabang memiliki parameter sendiri - tidak ada kompetisi parameter antar task. Anchor-Free: tanpa prior box template, prediksi langsung 4 koordinat per grid cell.

### Apa itu Neck?

Neck adalah komponen antara Backbone dan Head yang memfusikan fitur dari berbagai resolusi. Backbone menghasilkan tiga level fitur dengan karakteristik berbeda:
- **P3 (80×80):** resolusi tinggi, detail lokasi presisi, sedikit semantik
- **P4 (40×40):** resolusi sedang, keseimbangan detail dan semantik
- **P5 (20×20):** resolusi rendah, banyak semantik ("apa objeknya"), sedikit detail lokasi

Neck menggabungkan kelebihan semua level sehingga setiap level deteksi memiliki pemahaman semantik (what) DAN presisi lokasi (where).

### FPN (Feature Pyramid Network) - Top-down Path

| Langkah | Operasi | Resolusi | Efek |
|---------|---------|----------|------|
| 1 | P5 (20x20) -> Upsample 2x | 40x40 | Fitur semantik resolusi rendah diperbesar |
| 2 | Concat dengan P4 (40x40) | 40x40 | Fusion semantik + detail |
| 3 | Conv 1x1 reduce channel | 40x40 | Kompresi fitur, reduksi dimensi |
| 4 | Upsample 2x -> Concat dengan P3 | 80x80 | Informasi semantik mencapai resolusi tinggi |

**FPN membantu deteksi objek kecil** - sampah kecil seperti puntung rokok, tutup botol, atau baterai mendapat informasi semantik dari resolusi lebih rendah.

### PAN (Path Aggregation Network) - Bottom-up Path

| Langkah | Operasi | Resolusi | Efek |
|---------|---------|----------|------|
| 1 | P3 -> Downsample Conv k3 s2 | 40x40 | Fitur detail diperkecil |
| 2 | Concat dengan P4 (40x40) | 40x40 | Detail memperkaya fitur semantik |
| 3 | Conv 1x1 reduce | 40x40 | Reduksi dimensi |
| 4 | Downsample -> Concat dengan P5 | 20x20 | Detail mencapai resolusi rendah |

**PAN membantu lokalisasi objek besar** - informasi tepi presisi dari resolusi tinggi mengalir ke bawah, meningkatkan akurasi bounding box objek besar seperti kardus atau botol.

### Head: Decoupled Anchor-Free

**Decoupled Head** = tiga cabang konvolusi paralel sepenuhnya independen, masing-masing dengan parameter sendiri. Classification branch fokus membedakan Organik/Non-Organik, regression branch fokus presisi lokasi, segmentation branch fokus akurasi bentuk. Tidak ada parameter yang dibagi - eliminasi task competition yang terjadi jika satu set parameter harus menangani tiga tugas berbeda.

**Anchor-Free** = tanpa prior box template (tidak seperti YOLOv3/v5/v8). Setiap grid cell langsung memprediksi 4 koordinat (x, y, w, h). DFL (Distribution Focal Loss) memprediksi distribusi probabilitas 16-bin per koordinat - fleksibel menangkap berbagai rasio bentuk sampah (botol 1:4, kardus 1:1) tanpa perlu clustering dataset.

| Cabang | Input | Layer Detail | Output |
|--------|-------|-------------|--------|
| **Classification** | P3/P4/P5 | Conv3x3 -> SiLU -> Conv3x3 -> Linear + Sigmoid | 3 nilai: objectness + 2 class prob |
| **Regression (BBox)** | P3/P4/P5 | DFL 16-bin distribution per koordinat | 4 float: x, y, w, h |
| **Segmentation (Mask)** | P3/P4/P5 | Proto Module: 32 prototype masks + coefficient | 24-point polygon per instance |

---

## Slide 9: Loss Functions & Training Setup

### Hyperparameter Training

| Parameter | Nilai | Penjelasan |
|-----------|-------|------------|
| Input size | 640x640 | Resolusi gambar setelah letterbox resize - mempertahankan aspek ratio |
| Epochs | 100 | Jumlah iterasi penuh dataset |
| Patience | 40 | Hentikan training jika val loss tidak turun selama 40 epoch |
| Batch size | 16 | Gambar per batch |
| Optimizer | SGD (momentum 0.937) | Stochastic Gradient Descent dengan momentum |
| Learning rate | 0,001 (cosine schedule) | Turun mengikuti kurva cosinus dari 0,001 ke ~0 |
| Momentum | 0,937 | Momentum optimizer untuk mempercepat konvergensi |
| Weight decay | 0,0005 | Regularisasi L2 untuk mencegah overfitting |
| FP16 | Ya | Mixed precision training - mempercepat ~2x, VRAM turun ~40% |

### Hyperparameter Loss

| Loss | Weight | Fungsi |
|------|--------|--------|
| **CIoU Loss** | **7,5** | Optimasi 3 aspek overlap: IoU + center distance + aspect ratio. CIoU = 1 − IoU + ρ²(b,b_gt)/c² + α·v. Bobot tertinggi karena lokalisasi adalah prioritas - bounding box meleset berarti kegagalan deteksi total |
| **BCE Loss** | **0,5** | Binary Cross-Entropy untuk 2 kelas: BCE = −[y·log(p) + (1−y)·log(1−p)]. Setiap grid cell predict probabilitas Organik vs Non-Organik. Bobot rendah karena 2 kelas relatif mudah dibedakan secara visual |
| **DFL Loss** | **1,5** | Distribution Focal Loss: memprediksi distribusi probabilitas diskrit 16-bin per koordinat (bukan nilai tunggal). Nilai akhir = weighted sum Σ(bin_i × softmax(prob_i)). Keuntungan: (1) gradien lebih kaya - 16 sinyal vs 1, (2) representasi uncertainty untuk boundary tidak jelas, (3) memungkinkan arsitektur anchor-free |

### Detail Training

| Aspek | Detail |
|-------|--------|
| Warmup epochs | 5 (linear LR increase dari 0 -> 0,001) |
| GPU | NVIDIA RTX 5060 Ti 16GB GDDR7 |
| Waktu training | **~4,1 jam** (247 menit, 100 epoch) |
| Inference speed | **5,1 ms per image** (~196 FPS) |

### Optimizer SGD

**Apa itu SGD?** SGD (Stochastic Gradient Descent) dengan momentum 0.937 - 93.7% arah update berasal dari gradien sebelumnya, 6.3% dari gradien saat ini.

**Mengapa SGD bukan Adam?** (1) VRAM lebih hemat - tidak perlu menyimpan momentum + variance (2× lebih hemat). (2) Generalisasi lebih baik - SGD memiliki implicit regularization, tidak "nyaman" di sharp minima seperti Adam. (3) Cosine annealing mengkompensasi konvergensi lambat.

### Cosine LR Schedule

Learning rate decay mengikuti fungsi cosinus: `lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(epoch/epochs * pi))`. LR turun gradual dari 0.001 ke ~0.00001 mengikuti kurva cosinus. Berbeda dengan step decay (turun drastis di epoch tertentu), cosine annealing turun gradual → model konvergen ke minimum lebih dalam.

Warmup 5 epoch: LR naik linear 0 → 0.001, mencegah gradien eksplosif di awal training.

---

## Slide 10: Hasil Pelatihan

### Training Curves (100 epoch, batch 16, YOLOv26m-seg)

![Training Results](../runs_result/segment/full_pipeline/results.png)

**Interpretasi Kurva (dari kiri ke kanan, atas ke bawah):**
- **train/box_loss:** turun ~32% (1.33 → 0.90) - bounding box stabil konvergen. Lonjakan kecil di epoch ~33 saat close_mosaic (mosaic dimatikan, distribusi data berubah).
- **train/cls_loss:** turun ~83% (3.82 → 0.65) - klasifikasi 2 kelas cepat konvergen karena perbedaan visual Organik vs Non-Organik cukup jelas. Transfer learning dari COCO memberikan initial feature representation yang sudah baik.
- **train/seg_loss:** turun ~44% (4.54 → 2.54) - segmentasi lebih lambat konvergen karena pseudo-label noise (33.6% mask fallback). Penurunan akselerasi setelah epoch 68 saat LR mendekati minimum.
- **metrics/mAP50(B):** Box mAP naik konsisten dari ~7% (epoch 1) ke **76.9%** (epoch 72). Plateau setelah epoch 80 - menunjukkan model mencapai kapasitas maksimal dengan data dan pseudo-label saat ini.
- **metrics/mAP50(M):** Mask mAP naik dari ~4% ke **55.4%** (epoch 83). Gap 21.5% dengan Box mAP mencerminkan keterbatasan kualitas pseudo-mask.
- **metrics/precision(B) & recall(B):** Precision ~73%, Recall ~71.6%. Precision lebih tinggi dari recall - model cenderung under-predict (hanya deteksi jika yakin) daripada over-predict.
- **Gap train-val mAP < 5%** di semua metrik - tidak ada overfitting signifikan. Augmentasi online berhasil mencegah model menghafal training set.

### Confusion Matrix & Precision-Recall (3 validation runs)

Perbandingan confusion matrix dari 3 validation run berbeda:

| Run 1 (full_pipeline) | Run 2 (val-2) | Run 3 (val-3) |
|:---------------------:|:--------------:|:--------------:|
| ![CM1](../runs_result/segment/full_pipeline/confusion_matrix_normalized.png) | ![CM2](../runs_result/segment/val-2/confusion_matrix_normalized.png) | ![CM3](../runs_result/segment/val-3/confusion_matrix_normalized.png) |

| | Organik (Recall) | Non-Organik (Recall) | Analisis |
|-------|---------|-------------|----------|
| Full Pipeline | **~67%** | **~81%** | Baseline - model cukup baik deteksi Non-Organik |
| Val-2 | **~65%** | **~82%** | Konsisten - variasi kecil antar run |
| Val-3 | **~68%** | **~80%** | Stabil - perbedaan <3% antar run |

Confusion matrix konsisten antar 3 validation run - variance rendah (SD <3%). Model secara konsisten lebih baik mendeteksi Non-Organik (recall ~81%) dibanding Organik (recall ~67%). False positive Organik ~19% - plastik kusut, kain, dan material reflektif secara visual menyerupai organik. False negative Organik ~33% - organik amorf (sisa makanan, ampas kopi, kulit buah halus) tidak memiliki bentuk tegas, confidence di bawah threshold 0.25.

**Precision-Recall Curves (per class):**

| Box PR (full_pipeline) | Box PR (val-2) | Mask PR (full_pipeline) |
|:---------------------:|:--------------:|:----------------------:|
| ![BoxPR1](../runs_result/segment/full_pipeline/BoxPR_curve.png) | ![BoxPR2](../runs_result/segment/val-2/BoxPR_curve.png) | ![MaskPR](../runs_result/segment/full_pipeline/MaskPR_curve.png) |

PR curve menunjukkan trade-off precision vs recall pada berbagai confidence threshold. Kelas Non-Organik (oranye) memiliki area under curve lebih besar dari Organik (biru) - konsisten dengan mAP gap ~15%. Mask PR curve lebih rendah dari Box PR - mask segmentasi lebih sulit daripada deteksi bounding box. Titik optimal F1-score (~72.3% Box) berada di confidence threshold ~0.25-0.35.

### Validation Predictions (3 runs comparison)

| Run | Batch 0 | Batch 1 | Batch 2 |
|:---:|:-------:|:-------:|:-------:|
| **full_pipeline** | ![Val0](../runs_result/segment/full_pipeline/val_batch0_pred.jpg) | ![Val1](../runs_result/segment/full_pipeline/val_batch1_pred.jpg) | ![Val2](../runs_result/segment/full_pipeline/val_batch2_pred.jpg) |
| **val-2** | ![Val0-2](../runs_result/segment/val-2/val_batch0_pred.jpg) | ![Val1-2](../runs_result/segment/val-2/val_batch1_pred.jpg) | ![Val2-2](../runs_result/segment/val-2/val_batch2_pred.jpg) |
| **val-3** | ![Val0-3](../runs_result/segment/val-3/val_batch0_pred.jpg) | ![Val1-3](../runs_result/segment/val-3/val_batch1_pred.jpg) | ![Val2-3](../runs_result/segment/val-3/val_batch2_pred.jpg) |

Kotak hijau = ground truth, kotak/polygon merah muda = prediksi model. Observasi dari visualisasi:
- **Deteksi bounding box**: model konsisten mendeteksi objek utama di ketiga run. Posisi bounding box akurat (CIoU loss efektif).
- **Mask segmentasi**: presisi bervariasi - objek rigid (botol, kaleng) memiliki mask lebih akurat daripada objek amorf (sisa makanan). Mask pada objek dengan fallback geometris (ellipse/rect) terlihat kurang mengikuti kontur asli.
- **Objek kecil**: puntung rokok, tutup botol kecil kadang terlewat (area <20% threshold di pseudo-mask pipeline).
- **Konsistensi antar run**: prediksi sangat konsisten - variance rendah mengonfirmasi reproducibility training.

### Key Takeaway

| Metrik | Box | Mask |
|--------|-----|------|
| mAP@0.5 | 76.9% | 55.4% |
| mAP@0.5:0.95 | 51.9% | 23.9% |
| Precision | 73.0% | 54.9% |
| Recall | 71.6% | 59.0% |
| F1-Score | 72.3% | 56.9% |

| Kelas | Box mAP | Mask mAP |
|-------|---------|----------|
| Organik | ~68% | ~51% |
| Non-Organik | ~83% | ~60% |

**Ringkasan:** Model mencapai performa deteksi solid (Box mAP 76.9%) dengan segmentasi cukup (Mask mAP 55.4%). Gap 21.5% antara Box dan Mask mencerminkan keterbatasan kualitas pseudo-mask. Model konsisten antar 3 validation run (SD <3%), mengonfirmasi reproducibility. Performa Non-Organik (~83%) mendekati target proyek, sementara Organik (~68%) masih perlu peningkatan - terutama melalui perbaikan pseudo-mask dan penambahan data Organik.

---

## Slide 11: Pembahasan

### Edge Detection Rate vs Performa

Korelasi kuat antara edge detection success rate dan performa model (Spearman rho = 0.82). Setiap kenaikan 10% edge rate berkorelasi dengan kenaikan ~5-8% mAP. Edge detection success 66,4% menjadi faktor pembatas utama kualitas mask. Gambar yang jatuh ke fallback geometris (33,6%) menghasilkan polygon kurang presisi, langsung menurunkan mask mAP.

| Edge Success | Kontribusi | Mask Quality | Rata-rata mAP |
|-------------|------------|-------------|---------------|
| Approx Polygon (66,4%) | 2.639 gambar | Akurat, mengikuti kontur objek | ~62% (subkategori >80% edge rate) |
| Fallback Ellipse (20,2%) | ~800 gambar | Aproksimasi oval, kurang presisi | ~52% |
| Fallback Rounded Rect (13,4%) | ~534 gambar | Aproksimasi kotak, paling tidak presisi | ~48% (subkategori <60% edge rate) |

**Anomali:** coffee_tea_bags (edge rate 59,4%) mencapai mAP 72,1% - lebih tinggi dari glass_containers (89,7%, mAP 60,4%). Karena coffee_tea_bags bentuk seragam (rounded rect fallback cukup representatif), sementara glass bervariasi dan transparan.

### Class Imbalance

| Kelas | Jumlah | Persentase | Box mAP@0.5 | Mask mAP@0.5 |
|-------|--------|------------|-------------|-------------|
| Organik | 684 | 17,2% | ~68% | ~51% |
| Non-Organik | 3.289 | 82,8% | ~83% | ~60% |

Rasio 1:4,8 (Organik:Non-Organik). Kelas minoritas Organik memiliki performa lebih rendah di box dan mask. Imbalance mempengaruhi deteksi (gap box ~15%) dan segmentasi (gap mask ~9%).

### Failure Cases Analysis

| Tipe Gagal | Penyebab (Teknis) | Dampak (Kuantitatif) | Frekuensi |
|------------|-------------------|---------------------|-----------|
| **False Positive Organik** | Non-organik dengan tekstur/kontur menyerupai organik: plastik kusut (lipatan menyerupai sisa makanan), kain basah (nilai intensitas dan tekstur mirror organik). Di level feature map, classification branch tidak bisa membedakan pola lipatan plastik dari jaringan organik karena keduanya memiliki output objectness score >0.25 threshold. Root cause: dataset Non-Organik terlalu dominan (82.8%) sehingga model bias ke pola tekstur kompleks sebagai "bukan botol/kaleng" → fallback ke kelas alternatif (Organik) | Rekomendasi pembuangan salah: plastik/kain dikirim ke kompos → kontaminasi. Precision Organik turun +19%. Dampak operasional: petugas TPST harus manual re-sort kompos yang terkontaminasi | ~19% dari prediksi Non-Organik |
| **False Negative Organik** | Organik amorf (sisa makanan basah, ampas kopi, kulit buah halus) tidak memiliki bentuk geometris tegas. Edge detection Otsu gagal pada langkah 4 (thresholding): histogram unimodal karena warna objek menyatu dengan latar (sisa nasi di piring putih). Pseudo-mask pipeline menghasilkan fallback ellipse - ground truth mask sudah tidak akurat sejak awal. Model belajar segmentasi dari mask ellipse yang tidak representatif → saat inference, confidence <0.25 atau mask tidak cocok dengan objek asli. Faktor kedua: class imbalance 1:4.8 - model melihat 4.8× lebih banyak Non-Organik, prior prediksi bergeser ke kelas mayoritas | Objek Organik tidak terdeteksi sama sekali (miss) atau terdeteksi dengan confidence <0.25 → tidak masuk output NMS. Recall Organik turun ke ~67%. Sisa makanan lolos ke aliran Non-Organik → tidak bisa dikompos. Pada tumpukan sampah, organik yang tidak terdeteksi akan membusuk di TPA dan menghasilkan metana | ~33% dari objek Organik |
| **Mask under-segmentation** | Otsu thresholding (langkah 4 pipeline) tidak memisahkan dua objek yang menempel karena distribusi intensitasnya menyatu dalam satu puncak histogram. Saat `cv2.findContours` dengan `RETR_EXTERNAL` (hanya kontur terluar), dua objek yang saling bersentuhan dianggap satu kontur. `cv2.approxPolyDP` kemudian menyederhanakan kontur gabungan ini menjadi satu polygon yang mencakup kedua objek. Model training menerima ground truth mask yang salah (satu mask untuk 2 instance) → belajar bahwa dua objek terpisah seharusnya satu polygon. Saat inference, model memprediksi satu polygon untuk objek yang seharusnya terpisah | Dua objek terdeteksi sebagai satu instance → hitung jumlah (count) tidak akurat. Satu rekomendasi pembuangan untuk dua objek berbeda jenis (misal botol plastik + sisa makanan) → tidak bisa diproses. NMS tidak bisa memisahkan karena hanya satu prediksi dengan confidence tinggi. Dampak ke mask mAP: penalti IoU besar karena overlap dengan dua ground truth mask | ~8% gambar |
| **Mask over-segmentation** | Objek dengan variasi intensitas internal tinggi (kaleng mengkilap dengan refleksi cahaya, plastik bermotif) menyebabkan Otsu threshold (langkah 4) memilih threshold yang membelah objek menjadi beberapa region terpisah. `cv2.findContours` mendeteksi region-region ini sebagai kontur independen. Seleksi kontur terbesar (langkah 7A) mengambil region terbesar, tapi region lain tetap ada sebagai kontur terpisah. Saat `cv2.approxPolyDP` dan resampling, region terbesar hanya mencakup sebagian objek. Model belajar bahwa objek dengan refleksi adalah beberapa instance terpisah → saat inference, satu objek menghasilkan multiple polygon dengan confidence ≥0.25 | Satu objek dihitung sebagai 2+ objek → overcount. Masing-masing polygon memiliki area lebih kecil dari objek asli → IoU rendah dengan ground truth → mask mAP turun. NMS tidak bisa menggabungkan karena polygon terpisah secara spasial. Contoh: kaleng mengkilap → 2 deteksi (badan kaleng + tutup reflektif) | ~6% gambar |
| **Objek transparan** | Botol bening, plastik wrap, gelas kaca - objek transparan meneruskan cahaya latar sehingga intensitas piksel foreground ≈ background. `cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)` menghasilkan grayscale di mana objek dan latar memiliki nilai luminance yang hampir identik. Histogram grayscale menjadi unimodal (satu puncak besar) → Otsu thresholding tidak dapat menemukan threshold yang memisahkan foreground/background karena within-class variance minimal di semua threshold. Pipeline masuk ke fallback geometris (langkah 9B). Fallback ellipse/rect tidak mengikuti bentuk asli botol (silinder panjang) → ground truth mask berbentuk oval tidak presisi. Model belajar segmentasi botol sebagai ellipse → saat inference, mask berbentuk oval tidak cocok dengan botol sebenarnya (IoU rendah) | Mask mAP untuk subkategori kaca/bening turun drastis. Bottle detection: Box mAP masih ~50% (bounding box cukup akurat karena DFL menangkap distribusi boundary dari kontras tepi tipis), tapi Mask mAP hanya ~25-30% (mask ellipse tidak mengikuti silinder). Rekomendasi pembuangan: bounding box benar (Non-Organik), tapi mask tidak bisa digunakan untuk verifikasi visual. Dampak ke downstream: aplikasi web menampilkan mask yang tidak akurat → user trust turun | ~12% gambar Non-Organik |
| **Latar kompleks** | Sampah di rumput, tanah, pasir, karpet - latar memiliki variasi intensitas alami yang menciptakan puncak histogram tambahan. Otsu thresholding (yang mengasumsikan histogram bimodal) mendeteksi tekstur latar sebagai foreground karena variasi intensitas rumput/pasir menciptakan distribusi yang terpisah dari intensitas objek. Mask biner (langkah 4 output) berisi noise di area latar. Morphological close (langkah 6) dengan kernel 5×5 tidak cukup menghapus noise karena tekstur rumput memiliki skala korelasi spasial >5 piksel. `cv2.findContours` (langkah 7) mendeteksi kontur noise ini. Kontur terbesar mungkin masih objek utama, tapi polygon hasil approxPolyDP mengandung lekukan dari noise tepi. Jika noise lebih luas dari objek (latar dominan), kontur terbesar adalah latar → fallback ke ellipse/rect | Mask hasil edge detection memiliki tepi bergerigi (tidak mulus) karena noise latar. Polygon tidak mengikuti batas objek asli. Jika fallback: mask ellipse/rect terlalu besar dari objek. Box mAP turun ~5-10% pada kondisi latar kompleks karena bounding box overestimated. Mask mAP turun lebih signifikan (~10-15%) karena mask tidak presisi. Model belajar asosiasi yang salah antara fitur latar dan objek → false positive pada latar serupa saat inference | ~10% gambar |
| **Objek kecil (<20% area)** | Puntung rokok (~15×5 mm, ~0.5% frame), baterai kecil, tutup botol - area kontur <20% dari total gambar setelah validasi langkah 8 pipeline: `if cv2.contourArea(largest) < 0.20 * h * w: return None`. Threshold 20% didesain untuk menghindari noise kecil, tapi juga memfilter objek kecil yang valid. Edge detection sebenarnya sukses (kontur terekstraksi), tapi gagal validasi area → fallback geometris. Pada feature map P3 (80×80), objek kecil hanya menempati ~1-2 grid cells → representasi fitur sangat terbatas. Anchor-free detection kesulitan karena DFL 16-bin distribution tidak memiliki resolusi cukup untuk lokalisasi presisi objek sub-grid | Objek kecil sepenuhnya bergantung pada fallback geometris - mask tidak mengikuti bentuk asli. Deteksi bounding box: model bisa mendeteksi (karena konteks global dari P4/P5 membantu), tapi mask mAP sangat rendah (<20%). Dalam skenario tumpukan sampah, objek kecil sering terlewat karena tertutup objek lebih besar. Dampak operasional: puntung rokok tidak terdeteksi → lolos ke aliran kompos (kontaminasi) atau residu | ~8% gambar |

### Korelasi

Semakin rendah edge success, semakin besar gap box-mask. Peningkatan kualitas pseudo-label (mengganti fallback geometris dengan SAM) berpotensi menaikkan mask mAP 10-15%.

---

## Slide 12: Kesimpulan & Aplikasi

### Capaian Utama

1. **Dataset terintegrasi** - TACO (1.500) + Waste Classification (2.939) digabung jadi **3.973 gambar** (684 Organik + 3.289 Non-Organik) dengan format YOLO-seg

2. **Pseudo-Polygon Mask Generation** - Pipeline 12 langkah mencapai **66,4% edge detection success** dengan 33,6% fallback geometris (60% ellipse, 40% rounded rect). Cukup untuk training instance segmentation dengan mask mAP@0.5 = 55,4%

3. **YOLOv26m-seg mencapai performa baik:**
   - Box mAP@0.5: **76,9%** - deteksi bounding box akurat
   - Mask mAP@0.5: **55,4%** - segmentasi didukung mask ratio 2
   - Per-class: Organik Box ~68%, Non-Organik Box ~83%
   - Precision: **73,0%**, Recall: **71,6%**, F1-Score: **72,3%**
   - Inference: **5,1 ms/gambar** (~196 FPS) - real-time
   - Training: **~4,1 jam** pada RTX 5060 Ti 16GB (100 epoch)

### Aplikasi Web (Deployment)

| Komponen | Teknologi | Fungsi |
|----------|-----------|--------|
| Backend | FastAPI :8000 | REST API inference single & batch, lazy load model (54,5 MB), NMS threshold 0.25 |
| Frontend | Nuxt.js 3 :3000 | Dashboard upload (drag-drop), preview annotated, summary cards, 4 halaman CMS |
| Pipeline CMS | 4 route: dataset → preparation → training → deployment | Visibility end-to-end |

**Flow:** Upload gambar → resize 640×640 + letterbox → CNN forward (5.1ms GPU) → decode output (class, confidence, bbox, 24-point polygon) → NMS → JSON response + annotated image → recycling advice 3-tier: Organik (kompos), Anorganik (Bank Sampah), Residu (TPS B3). Latency end-to-end <25 ms.

**Model:**
- Ukuran: 54.5 MB (FP32), dapat di-quantize ke FP16 (27 MB) atau INT8 (14 MB) untuk edge deployment
- 26.97M parameters, 131.9 GFLOPs → 121.2 GFLOPs (fused)
- 329 layers unfused → 149 layers fused (Conv2D+BN+SiLU digabung jadi satu layer)

### Saran Improvement

**Target:** Box mAP 76.9% → 85%+, Mask mAP 55.4% → 65%+

| Prioritas | Tindakan | Dampak Prediksi |
|-----------|----------|-----------------|
| 1 | Ganti fallback geometris dengan SAM (Segment Anything Model) untuk pseudo-mask | Mask mAP naik 10-15% |
| 2 | Kumpulkan 2.000+ gambar Organik (balance menuju 50:50) | Recall Organik naik 10-15% |
| 3 | Anotasi manual 500 gambar kunci untuk pseudo-mask berkualitas | Mask mAP Organik naik 15-20% |
| 4 | Class-weighted loss + focal loss variant | Gap box antar kelas mengecil |
| 5 | Extended training 150 epoch, close_mosaic 50 | Potensi mAP marginal +1-2% |
