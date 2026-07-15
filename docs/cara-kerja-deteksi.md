# Cara Kerja Deteksi — Panduan Junior Programmer

Dokumen ini menjelaskan bagaimana satu baris di Bounding Box Report bisa muncul. Bahasa sederhana, tanpa rumus rumit.

---

## Contoh Satu Baris

```
#  | Label        | Category    | Confidence | X1 | Y1 | X2   | Y2  | W    | H   | YOLO Format
1  | Non-Organik  | Non-Organik | 83.6%      | 0  | 7  | 1200 | 800 | 1200 | 793 | 1 0.0 7.0 1200.0 793.0
```

Artinya: YOLO melihat sebuah gambar 1200×800 piksel, dan bilang "Saya 83.6% yakin ada objek **Non-Organik** di area (0,7) sampai (1200,800) — hampir seluruh gambar."

---

## Step 1: User Upload Gambar

User pilih file gambar (misal `sampah.jpg` ukuran 1200×800 piksel), lalu klik tombol Detect.

**Yang terjadi:** File dikirim dari browser ke server backend (FastAPI) melalui HTTP POST ke endpoint `/api/detect`.

**Di kode:** `frontend/composables/useDetection.ts` — fungsi `detect()` mengambil file dari form, bungkus sebagai FormData, kirim ke backend.

---

## Step 2: Backend Terima & Validasi File

Server backend menerima file, cek:
- Apa benar ini gambar? (cek ekstensi: .jpg, .png, .jpeg)
- Ukuran file tidak terlalu besar? (maks 200MB)
- Simpan file sementara di folder `uploads/`

**Di kode:** `backend/app/routes/detect.py` — endpoint POST `/api/detect` panggil `save_upload()` dari `file_utils.py`.

**Penjelasan sederhana:** Backend kayak resepsionis — terima kiriman, pastikan isinya bener, catat, terus serahkan ke bagian pemrosesan.

---

## Step 3: Gambar Disiapkan — Resize & Letterbox

### 3.1 Masalah: YOLO Cuma Bisa Terima 640×640 Persegi

YOLO (CNN) butuh input **fixed-size persegi** — 640×640. Tapi gambar user bisa bermacam-macam ukuran. Contoh kita: 1200×800 (landscape, tidak persegi).

### 3.2 Ada 3 Cara Bikin Gambar Jadi 640×640

| Metode | Cara | Hasil | Masalah |
|--------|------|-------|---------|
| **Stretch** | Paksa resize jadi 640×640 | ❌ Objek gepeng/melebar | Botol jadi pendek-gemuk — model bingung |
| **Crop** | Potong tengah 640×640 | ❌ Objek di tepi hilang | Sampah di pinggir foto tidak terdeteksi |
| **Letterbox** ✅ | Resize proporsional + padding hitam | ✅ Objek utuh, proporsi terjaga | Ada area hitam (tapi tidak masalah) |

**Visual perbandingan:**

```
Gambar asli 1200×800:
┌────────────────────────────────────────────────────────────┐
│                                                            │
│                   □□ BOTOL PLASTIK □□                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

STRETCH (paksa 640×640):
┌──────────────────┐
│░░░░░░░░░░░░░░░░░░│  ← botol jadi gepeng!
│░░░░ BOTOL ░░░░░░│     model tidak bisa belajar
│░░░░ PLASTIK ░░░░│     karena proporsi tidak natural
│░░░░░░░░░░░░░░░░░░│
└──────────────────┘

CROP (ambil tengah):
┌──────────────────┐
│                  │
│   □□ BOTOL □□   │  ← botol utuh tapi
│                  │     sampah di pinggir
└──────────────────┘     hilang tidak terdeteksi

LETTERBOX (dipakai YOLO):
┌──────────────────┐
│██████████████████│  ← padding hitam 107px
│░░░░░░░░░░░░░░░░░░│
│░░░░ BOTOL ░░░░░░│  ← gambar 640×427
│░░░░ PLASTIK ░░░░│     (proporsi terjaga sempurna)
│░░░░░░░░░░░░░░░░░░│
│██████████████████│  ← padding hitam 106px
└──────────────────┘
640×640
```

### 3.3 Cara Hitung Letterbox — Step by Step

```
Input gambar: 1200 × 800
Target:        640 × 640

Langkah 1 — Hitung skala:
  skala_lebar  = 640 / 1200 = 0.533
  skala_tinggi = 640 / 800  = 0.8
  skala = min(0.533, 0.8) = 0.533
  → ambil yang terkecil biar tidak ada yang kepotong

Langkah 2 — Resize gambar dengan skala:
  lebar_baru  = 1200 × 0.533 = 640   (pas!)
  tinggi_baru = 800 × 0.533  = 427   (masih kurang 213px)

Langkah 3 — Hitung padding:
  total_padding = 640 - 427 = 213
  padding_atas  = 213 ÷ 2 = 106 (dibulatkan ke bawah)
  padding_bawah = 213 - 106 = 107

Langkah 4 — Hasil akhir:
  ┌──────────────────┐
  │████ 106px hitam ██│
  │░░ 427px gambar ░░│
  │████ 107px hitam ██│
  └──────────────────┘
  Tensor input: (640, 640, 3)
  Nilai pixel:  0-255 (BGR)
  Area hitam:   [0, 0, 0]
```

### 3.4 Apa yang Terjadi pada Koordinat?

Karena ada padding, posisi objek bergeser:

```
Titik (0, 7) di gambar asli 1200×800:
  x_frame = 0 × 0.533          = 0
  y_frame = 7 × 0.533 + 106    = 3.7 + 106 = 109.7

Titik (1200, 800) di gambar asli:
  x_frame = 1200 × 0.533        = 640
  y_frame = 800 × 0.533 + 106   = 426.4 + 106 = 532.4
```

**Nanti setelah model prediksi,** Ultralytics otomatis scale balik:

```
x_asli = (x_frame - padding_kiri) / skala
y_asli = (y_frame - padding_atas) / skala

Contoh untuk box hasil model [0, 109.7, 640, 532.4] (frame 640):
  x1 = (0 - 0) / 0.533      = 0
  y1 = (109.7 - 106) / 0.533 = 7.0   ← kembali ke koordinat asli!
  x2 = (640 - 0) / 0.533    = 1200
  y2 = (532.4 - 106) / 0.533 = 800
```

**Presisi terjaga** — tidak ada informasi yang hilang.

### 3.5 Apakah Padding Hitam Mengganggu Model?

**Tidak.** Empat alasan:

1. **Model sudah dilatih dengan letterbox** — saat training, gambar juga di-letterbox dulu. Model terbiasa melihat area hitam.
2. **Padding tidak mengandung objek** — area hitam = 0. Neuron di sana tidak aktif. Confidence rendah, otomatis difilter.
3. **Objek tetap di tengah** — area padding hanya di pinggir, objek tetap di area gambar asli.
4. **Backpropagation tidak terpengaruh** — gradien di area hitam = 0, tidak mengubah weight.

### 3.6 Kapan Letterbox Bisa Jadi Masalah?

| Situasi | Masalah | Solusi |
|---------|---------|--------|
| Objek sangat kecil | Setelah resize makin kecil | Pakai input lebih besar (imgsz=1280) |
| Objek persis di tepi | Bisa kena potong letterbox | Foto objek di tengah frame |
| Rasio ekstrim (2000×100) | Banyak area hitam terbuang | Crop manual sebelum upload |
| Video portrait | Samping kiri-kanan hitam | Tidak masalah — tetap jalan |

### 3.7 Kenapa 640×640? (Bukan 320, Bukan 1280, Bukan 1024?)

Angka 640×640 tidak asal pilih. Ini hasil **keseimbangan (tradeoff)** antara 3 faktor:

#### Faktor 1: Akurasi vs Kecepatan

| imgsz | Grid Total Sel | Kecepatan Relatif | Akurasi Objek Kecil | VRAM |
|-------|----------------|-------------------|---------------------|------|
| **320** | 2.100 | 🟢🟢🟢🟢🟢 5× | ❌ Banyak lewat | ~1 GB |
| **480** | 4.725 | 🟢🟢🟢🟢 | 🟡 Cukup | ~2.5 GB |
| **640** ✅ | **8.400** | 🟢🟢🟢 | 🟢 Baik | ~4 GB |
| **960** | 18.900 | 🟢🟢 | 🟢🟢 Lebih baik | ~8 GB |
| **1280** | 33.600 | 🟢 | 🟢🟢🟢 Terbaik | ~12 GB |

**640 adalah sweet spot** — deteksi cukup akurat tanpa perlu GPU mahal.

#### Faktor 2: Dibagi Habis 32

Stride terbesar YOLO adalah **32** (P5). Agar grid di P5 bulat (tidak pecahan):

```
640 ÷ 32 = 20   ← bulat, tidak ada sisa
```

Kalau imgsz=650:
```
650 ÷ 32 = 20.3125   ← pecahan! Grid jadi tidak presisi
```

Angka yang habis dibagi 32: **32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280**.

Dari daftar itu, yang umum dipakai: **320 (cepat), 640 (seimbang), 1280 (akurat)**.

#### Faktor 3: Warisan Dataset & Pretrained Model

Model YOLO pretrained (seperti `yolo26m-seg.pt`) dilatih oleh Ultralytics dengan **imgsz=640** di dataset COCO. Kalau kita ubah imgsz terlalu jauh, model harus beradaptasi — bisa turun akurasi.

| Pakai imgsz | Efek |
|-------------|------|
| **640** (sama dengan pretrained) | Akurasi optimal, langsung jalan |
| 320 | Lebih cepat, tapi objek kecil hilang |
| 1280 | Lebih lambat, perlu fine-tuning biar optimal |

#### Kok Bukan 512 atau 1024?

| Angka | Habis ÷32? | Dipakai? | Kenapa? |
|-------|------------|----------|---------|
| **512** | ✅ (512÷32=16) | Jarang | Terlalu kecil untuk deteksi objek ukuran normal |
| **640** | ✅ (640÷32=20) | ✅ **Paling umum** | Standar industri |
| **1024** | ✅ (1024÷32=32) | Kadang | Untuk objek sangat kecil, butuh GPU besar |

**Kesimpulan:**
- 640 dipakai karena **cukup besar untuk deteksi akurat, cukup kecil untuk GPU konsumen**
- 640 habis dibagi 32 → grid bulat
- 640 adalah standar Ultralytics pretrained
- Kalau GPU kuat dan butuh lebih detail → naik ke 1280
- Kalau GPU lemah dan deteksi kasar → turun ke 320

#### Di Project Ini

Kita set `imgsz=640` di:

```python
# backend/app/cli/train.py:114  — training
parser.add_argument("--imgsz", type=int, default=640)

# backend/app/cli/test.py:15   — evaluasi
parser.add_argument("--imgsz", type=int, default=640)

# backend/app/services/yolo_service.py:46  — inference
model(img, imgsz=640)
```

### 3.8 Di Kode

Semua penanganan letterbox dilakukan otomatis oleh Ultralytics YOLO. Kita hanya set:

```python
model.predict(gambar, imgsz=640)  # ← Ultralytics urus letterbox sendiri
```

Backend (`detector.py`) dan frontend tidak perlu urus padding — model sudah handle scale balik.

**Yang perlu diingat:** Model melihat 640×640 dengan padding. Tapi output `box.xyxy` sudah dalam koordinat gambar asli 1200×800. Seolah-olah model melihat gambar asli langsung.

---

## Step 4: Model YOLO Memproses — Gambar Dipecah Jadi Grid Kecil

YOLO membayangkan gambar sebagai grid kotak-kotak, seperti papan catur raksasa.

Bayangkan:

```
┌──┬──┬──┬──┬──┬──┬──┬──┐
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
├──┼──┼──┼──┼──┼──┼──┼──┤
└──┴──┴──┴──┴──┴──┴──┴──┘
20×20 grid (setiap kotak = 32×32 piksel)
```

Tapi YOLO pakai **3 grid sekaligus** dengan ukuran beda:

| Nama Grid | Ukuran | Setiap Kotak Mewakili | Cocok Untuk |
|-----------|--------|----------------------|-------------|
| P3 | 80×80 | 8×8 piksel asli | Objek kecil (botol jauh, piring kecil) |
| P4 | 40×40 | 16×16 piksel asli | Objek sedang (gelas, botol dekat) |
| P5 | 20×20 | 32×32 piksel asli | **Objek besar** (kardus, tumpukan sampah) |

**Contoh kita:** Box `[0,7,1200,800]` hampir sebesar gambar. Ini objek **besar**, paling mungkin terdeteksi di **grid P5 (20×20)** — karena setiap kotaknya sudah mewakili area yang luas.

---

### Penjelasan Detail: Dari Mana Angka 80×80, 40×40, 20×20?

Angka-angka ini **tidak di-hardcode** di kode SmartBin. Mereka **turun (derive) otomatis** dari 2 hal:

```
Grid Size = imgsz ÷ Stride

imgsz = 640  (dari config)
Stride = faktor downsampling backbone CNN

P3: 640 ÷ 8  = 80   → 80×80 grid
P4: 640 ÷ 16 = 40   → 40×40 grid
P5: 640 ÷ 32 = 20   → 20×20 grid
```

#### Dari Mana Stride 8, 16, 32?

Stride adalah **faktor pengecilan ukuran** setelah gambar melewati lapisan CNN.

```
Gambar masuk 640×640
    │
    ├─ Conv stride=2  → 320×320   (setengah)
    ├─ Conv stride=2  → 160×160   (seperempat)
    ├─ Conv stride=2  →  80×80    (seperdelapan)  ← stride 8×, P3
    ├─ Conv stride=2  →  40×40    (seperenambelas) ← stride 16×, P4
    └─ Conv stride=2  →  20×20    (sepertigadua)   ← stride 32×, P5
```

Setiap kali CNN melakukan konvolusi dengan **stride=2**, ukuran feature map turun setengah. Setelah 3 kali stride=2: 640 ÷ 2³ = 80. Setelah 4 kali: 40. Setelah 5 kali: 20.

#### Di Kode Mana Ini Ditentukan?

**1. Model YAML** — Arsitektur backbone (`yolo26-seg.yaml` di Ultralytics):

```yaml
# Dari file: ultralytics/cfg/models/26/yolo26-seg.yaml
- [-1, 1, Conv, [256, 3, 2]]    # 3-P3/8     ← stride 8
- [-1, 1, Conv, [512, 3, 2]]    # 5-P4/16    ← stride 16
- [-1, 1, Conv, [1024, 3, 2]]   # 7-P5/32    ← stride 32
```

Setiap `Conv [filter, kernel_size=3, stride=2]` mengecilkan ukuran setengah. Komentar `P3/8` artinya "ini output P3 dengan total stride 8×".

Tidak ada angka 80, 40, 20 di file ini. Hanya stride.

**2. Runtime Computation** — Ultralytics hitung stride otomatis saat model dimuat:

```python
# Dari file: ultralytics/nn/tasks.py line 428
m.stride = torch.tensor([s / x.shape[-2] for x in _forward(dummy_input)])
```

Code ini: kirim dummy input 256×256 lewat backbone, ukur berapa ukuran outputnya, lalu stride = 256 / ukuran_output. Hasilnya tensor([8, 16, 32]).

**3. Grid Dibuat Per-Inference** — Setiap kali deteksi:

```python
# Dari file: ultralytics/utils/tal.py line 406-413
for i in range(len(feats)):
    h, w = feats[i].shape[2:]   # ambil tinggi & lebar feature map
    # buat grid points dari h×w
```

Untuk imgsz=640:
- Feature map P3 shape = (1, 256, **80**, 80) → grid 80×80
- Feature map P4 shape = (1, 512, **40**, 40) → grid 40×40
- Feature map P5 shape = (1, 1024, **20**, 20) → grid 20×20

#### Apakah Bisa Diubah?

**Ya.** Ganti `imgsz` → grid ikut berubah:

| imgsz | P3 | P4 | P5 | Total Sel |
|-------|----|----|----|-----------|
| 640 | 80×80 (6.400) | 40×40 (1.600) | 20×20 (400) | **8.400** |
| 1280 | 160×160 (25.600) | 80×80 (6.400) | 40×40 (1.600) | **33.600** |
| 320 | 40×40 (1.600) | 20×20 (400) | 10×10 (100) | **2.100** |

**Konsekuensi:**
- imgsz lebih besar → grid lebih rapat → deteksi objek kecil lebih baik → komputasi lebih lambat
- imgsz lebih kecil → grid lebih jarang → objek kecil bisa terlewat → komputasi lebih cepat

Di kode kita, `imgsz=640` diset di:

```python
# backend/app/cli/train.py:114
parser.add_argument("--imgsz", type=int, default=640)

# backend/app/cli/test.py:15
parser.add_argument("--imgsz", type=int, default=640)

# backend/app/services/yolo_service.py:46
model(img, imgsz=640)
```

#### Ringkasan

```
Model YAML           stride       imgsz        Grid
(yolo26-seg.yaml)    (dihitung)   (config)     (otomatis)
────────────────────────────────────────────────────────
P3/8                 ÷8           640          640÷8=80   → 80×80
P4/16                ÷16          640          640÷16=40  → 40×40
P5/32                ÷32          640          640÷32=20  → 20×20
```

Tidak ada angka ajaib. Ini matematika sederhana: **ukuran input dibagi stride backbone**.

---

## Step 5: Model Prediksi — Setiap Kotak Grid "Nebak"

Bayangkan setiap kotak di grid bertanya 3 hal:

### Pertanyaan 1: "Ada objek di sini?"

Model lihat area di dalam kotak, lalu kasih skor 0 sampai 1:
- 0.0 = tidak ada apa-apa
- 0.9 = yakin ada objek

### Pertanyaan 2: "Kalau ada, objek apa?"

Model kasih 2 skor:
- Skor Organik: 0.01 (yakin bukan organik)
- Skor Non-Organik: 0.92 (yakin ini non-organik)

Yang dipilih = skor tertinggi: **Non-Organik**.

### Pertanyaan 3: "Di mana persisnya batas objek?"

Model nebak 4 angka:
- Seberapa jauh batas kiri dari pusat kotak?
- Seberapa jauh batas kanan?
- Seberapa jauh batas atas?
- Seberapa jauh batas bawah?

Dari 4 angka ini, model menggambar persegi panjang (bounding box) di sekitar objek.

**Ilustrasi:**

```
Suatu kotak di grid P5 (20×20):
        
        ←── l ──→←── r ──→
        ┌──────────────────┐  ↑
        │                  │  t
        │    ★ (pusat)     │  ↓
        │                  │
        └──────────────────┘  ↑
        ←────── bw ──────→   b
                              ↓
        
l = 6.8  (batas kiri, dalam satuan 32 piksel)
r = 5.2  (batas kanan)
t = 6.1  (batas atas)
b = 6.4  (batas bawah)

bx = (posisi_kotak_x - l) × 32
by = (posisi_kotak_y - t) × 32
bw = (l + r) × 32
bh = (t + b) × 32
```

---

## Step 6: Model Juga Kasih Skor Keyakinan (Confidence)

Confidence bukan cuma "seberapa yakin ada objek". Ini gabungan 2 hal:

```
Confidence = "apa bener ada objek?" × "apa box-nya pas?"
```

Makanya nilai akhir 83.6%, bukan 92% (yang tadi dari klasifikasi).

**Analogi:** 
- Klasifikasi bilang 92% = "Saya yakin ini Non-Organik"
- Tapi box-nya tidak sempurna — mungkin kurang pas 1-2 piksel
- Jadi confidence diturunkan jadi 83.6% = "Yakin Non-Organik, tapi box agak meleset"

---

## Step 7: Model Mengecek Ulang — NMS (Non-Max Suppression)

Masalah: Satu objek sering "dilihat" oleh banyak kotak sekaligus. Bayangkan 5 kotak mengklaim objek yang sama:

```
Sebelum NMS (5 box untuk 1 objek):
┌──┐
│┌─┼─┐
└┼─┼─┘
 └─┘
 
Setelah NMS (1 box terbaik):
┌──────┐
│      │
│      │
└──────┘
```

Cara kerja NMS:
1. Urutkan semua box dari yang paling yakin ke paling ragu
2. Ambil box paling yakin, simpan sebagai hasil
3. Hapus box lain yang hampir sama posisinya (terlalu tumpang tindih)
4. Ulangi sampai semua box unik

**Ibarat pemilu:** Banyak kandidat (box), dipilih yang paling populer (confidence tinggi), kandidat mirip disuruh mundur.

**Batas "terlalu mirip":**
- Saat deteksi: 45% tumpang tindih → anggap duplikat, hapus
- Saat validasi: 70% tumpang tindih → lebih longgar (biar akurasi terukur lebih tinggi)

---

## Step 8: Model Skal Balik ke Ukuran Asli

Ingat Step 3? Kita perkecil gambar jadi 640×640 dengan padding. Sekarang hasilnya harus dikembalikan ke ukuran asli.

Model memprediksi box dalam koordinat 640×640 (dengan padding). Ultralytics (library YOLO) secara otomatis:

1. Kurangi padding hitam
2. Scale sesuai rasio ukuran asli

```
Dari: box.xyxy = [102.4, 124.8, 486.4, 524.8]  (dalam 640×640)
Ke:   box.xyxy = [0, 7, 1200, 800]              (dalam 1200×800 asli)
```

**Yang disimpan model ke memori:**

```
box.cls  = [1]          → Non-Organik (kelas 1)
box.conf = [0.836]      → Yakin 83.6%
box.xyxy = [[0, 7, 1200, 800]]  → Pojok kiri atas (0,7), pojok kanan bawah (1200,800)
```

---

## Step 9: Backend Baca Hasil dari Model

File `backend/app/services/detector.py`:

```python
# Ambil data dari hasil YOLO
cls_id = int(box.cls[0])        # 1 → Non-Organik
conf = float(box.conf[0])       # 0.836
x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  
                                # x1=0, y1=7, x2=1200, y2=800

# Hitung lebar dan tinggi
w, h = x2 - x1, y2 - y1
# w = 1200, h = 793

# Cari label dari ID kelas
label = CATEGORY_MAP[1]  # "Non-Organik"
```

**CATEGORY_MAP** adalah kamus sederhana:
```python
{
    0: ("Organik", "Organik"),
    1: ("Non-Organik", "Non-Organik"),
}
```

**Apa yang backend lakukan?** Hanya menyalin angka dari output model ke format yang rapi. Tidak ada perubahan atau perhitungan ulang.

**Kenapa H = 793, bukan 800?** Model memprediksi Y1 = 7, artinya objek mulai 7 piksel dari tepi atas, bukan persis di tepi. Mungkin ada batas putih 7 piksel, atau model sengaja memberi margin. YOLO tidak selalu presisi sempurna di tepi gambar.

Backend menyusun response JSON:

```json
{
  "label": "Non-Organik",
  "category": "Non-Organik",
  "confidence": 0.836,
  "bbox": [0.0, 7.0, 1200.0, 800.0]
}
```

---

## Step 10: Data Dikirim ke Frontend

Backend mengirim response HTTP ke browser. Isinya:

```json
{
  "success": true,
  "file_type": "image",
  "filename": "sampah.jpg",
  "detected_objects": [
    {
      "label": "Non-Organik",
      "category": "Non-Organik",
      "confidence": 0.836,
      "bbox": [0.0, 7.0, 1200.0, 800.0]
    }
  ],
  "summary": {
    "total": 1,
    "organik": 0,
    "non_organik": 1
  },
  "result_url": "/static/result/annotated_sampah.jpg"
}
```

---

## Step 11: Frontend Tampilkan di Tabel

File `frontend/components/BoundingBoxReport.vue` menerima data ini dan mengolahnya:

### Confidence → Persen

```javascript
// Dari 0.836 jadi "83.6%"
(0.836 * 100).toFixed(1) + "%"
// 83.6 + "%" = "83.6%"
```

**Cara baca:** Kalikan 100, bulatkan 1 desimal, tambah tanda persen.

### X1, Y1, X2, Y2

```javascript
// Ambil dari array bbox[0 sampai 3]
bbox = [0.0, 7.0, 1200.0, 800.0]

X1 = bbox[0].toFixed(0)   // 0.0 → "0"
Y1 = bbox[1].toFixed(0)   // 7.0 → "7"
X2 = bbox[2].toFixed(0)   // 1200.0 → "1200"
Y2 = bbox[3].toFixed(0)   // 800.0 → "800"
```

**Cara baca:** Ambil angka dari array, buang desimal (`.toFixed(0)`).

### W (Lebar) dan H (Tinggi)

```javascript
// Dihitung dari selisih X2-X1 dan Y2-Y1
W = (bbox[2] - bbox[0]).toFixed(0)  // 1200 - 0 = "1200"
H = (bbox[3] - bbox[1]).toFixed(0)  // 800 - 7 = "793"
```

**Cara baca:** Lebar = ujung kanan dikurang ujung kiri. Tinggi = ujung bawah dikurang ujung atas.

### YOLO Format (Custom)

```javascript
// Format: class_id x1 y1 w h
clsId("Non-Organik") = 1  (karena Organik=0, Non-Organik=1)
x1 = 0.0
y1 = 7.0
w  = 1200.0
h  = 793.0

Hasil: "1 0.0 7.0 1200.0 793.0"
```

**Cara baca:** Ini format khusus — bukan YOLO standar. Berisi: nomor kelas, lalu koordinat X1, Y1, lebar, tinggi (semua dalam piksel asli, bukan dipersentasekan).

---

## Step 12: Tabel Dirender di HTML

Vue.js mengubah data JavaScript menjadi baris tabel HTML:

```html
<tr>
  <td>1</td>                                          <!-- nomor urut -->
  <td>Non-Organik</td>                                 <!-- label -->
  <td><span class="bg-blue...">Non-Organik</span></td> <!-- category dengan badge warna -->
  <td>83.6%</td>                                       <!-- confidence dalam persen -->
  <td>0</td>                                           <!-- X1 -->
  <td>7</td>                                           <!-- Y1 -->
  <td>1200</td>                                        <!-- X2 -->
  <td>800</td>                                         <!-- Y2 -->
  <td>1200</td>                                        <!-- W = X2 - X1 -->
  <td>793</td>                                         <!-- H = Y2 - Y1 -->
  <td><code>1 0.0 7.0 1200.0 793.0</code></td>        <!-- YOLO format -->
</tr>
```

---

## Ringkasan: Siapa Kerja Apa?

```
┌──────────────┐
│   USER       │ Upload gambar 1200×800
└──────┬───────┘
       ↓
┌──────────────┐
│   BACKEND    │ Terima, validasi, simpan file
└──────┬───────┘
       ↓
┌──────────────┐
│   PREPROSES  │ Resize 640×427, letterbox 640×640
└──────┬───────┘
       ↓
┌──────────────┐
│   YOLO       │ Proses dengan grid (P3,P4,P5)
│   MODEL      │ Prediksi box, kelas, confidence
│              │ NMS, scale balik ke ukuran asli
│              │ Output: cls=1, conf=0.836, xyxy=[0,7,1200,800]
└──────┬───────┘
       ↓
┌──────────────┐
│   BACKEND    │ Baca output model
│              │ Hitung w=1200, h=793
│              │ Cari label "Non-Organik"
│              │ Kirim JSON ke frontend
└──────┬───────┘
       ↓
┌──────────────┐
│   FRONTEND   │ Terima JSON
│              │ Ubah conf 0.836 → "83.6%"
│              │ Ambil bbox[0..3] → X1,Y1,X2,Y2
│              │ Hitung W = X2-X1, H = Y2-Y1
│              │ Format YOLO = clsId + x1 + y1 + w + h
│              │ Render tabel HTML
└──────┬───────┘
       ↓
┌──────────────┐
│   USER       │ Lihat Bounding Box Report
└──────────────┘
```

---

## Cara Menjelaskan ke Orang Lain (Script Presentasi)

"Jadi begini cara kerja deteksi sampah kita:

1. **User upload foto** sampah — bisa botol, plastik, sisa makanan, apa aja.

2. **Gambar diperkecil** jadi 640×640 — seperti bikin thumbnail, tapi proporsinya dijaga (ada pinggiran hitam biar tidak gepeng).

3. **YOLO memecah gambar** jadi ribuan kotak kecil — bayangkan kertas milimeter block. Setiap kotak "melihat" bagian gambarnya sendiri.

4. **Setiap kotak nebak**: 'Ini ada objek gak ya? Kalau ada, organik atau non-organik? Terus batasnya di mana?'

5. **Tapi banyak kotak** nebak objek yang sama. NMS kayak pemilu — pilih satu kandidat terbaik, sisanya disuruh mundur.

6. **Hasilnya dikembalikan** ke ukuran gambar asli, dikirim ke server, lalu server kirim ke browser.

7. **Browser tampilkan tabel** — confidence diubah ke persen, koordinat box ditampilkan, lebar dan tinggi dihitung dari selisih X2-X1 dan Y2-Y1.

8. **Voila!** User lihat: 'Non-Organik, 83.6%, dari pojok kiri atas ke kanan bawah.'"

---

## Konsep Penting CNN (Convolutional Neural Network)

CNN adalah "otak" di belakang YOLO. Semua konsep di bawah adalah cara CNN memahami gambar.

---

### 1. Kernel (Filter)

**Apa itu?** Kernel adalah pola kecil — seperti stempel — yang digeser di atas gambar untuk mencari pola tertentu.

**Analogi:** Bayangkan punya stempel berbentuk lingkaran hitam. Kamu geser stempel itu di atas foto. Setiap kali stempel cocok dengan lingkaran di foto, kamu kasih tanda. Itulah cara kerja kernel.

**Ukuran kernel yang kita pakai:**
- **3×3**: Paling umum — melihat tetangga terdekat
- **1×1**: Melihat satu titik — berguna untuk menggabungkan informasi tanpa melihat tetangga

**Jumlah kernel:** Dalam satu lapisan bisa ada 128, 256, atau 512 kernel. Masing-masing mencari pola berbeda:
- Kernel 1: mencari garis vertikal
- Kernel 2: mencari garis horizontal
- Kernel 3: mencari lingkaran
- Kernel 4: mencari tekstur kotak-kotak
- ... dan seterusnya

**Contoh nyata di kode:** Di head YOLO, ada Conv 1×1 yang mengubah 256 channel jadi 2 channel (kelas Organik/Non-Organik). Setiap channel pakai kernel berbeda.

---

### 2. Dot Product (Perkalian Titik)

**Apa itu?** Dot product adalah operasi matematika untuk mengukur **seberapa cocok** dua hal. CNN menggunakannya untuk mengukur seberapa cocok kernel dengan bagian gambar.

**Cara kerja:**
```
Kernel (3×3):
[1  0  -1]
[2  0  -2]
[1  0  -1]

Bagian gambar (3×3):
[10  12   9]
[11  13  10]
[ 9  11  10]

Dot product = (1×10)+(0×12)+(-1×9)+(2×11)+(0×13)+(-2×10)+(1×9)+(0×11)+(-1×10)
            = 10 + 0 - 9 + 22 + 0 - 20 + 9 + 0 - 10
            = 2
```

**Analogi:** Seperti mencocokkan kunci dengan lubangnya. Kalau cocok (angka besar), berarti pola yang dicari ketemu. Kalau tidak cocok (angka kecil atau negatif), pola tidak ada.

**Kenapa penting?** Setiap kali CNN "melihat" gambar, dia melakukan ribuan dot product per detik — mencocokkan kernel dengan setiap bagian gambar.

---

### 3. Padding

**Apa itu?** Padding adalah nambah baris/kolom pixel di tepi gambar (biasanya hitam/0) sebelum proses konvolusi.

**Visual:**
```
Tanpa padding:
[1 2 3]    Setelah conv 3×3:    [a b]
[4 5 6]    ukuran mengecil      [c d]
[7 8 9]    3×3 → 2×2

Dengan padding 1:
[0 0 0 0 0]
[0 1 2 3 0]    Setelah conv 3×3:    [a b c]
[0 4 5 6 0]    ukuran tetap          [d e f]
[0 7 8 9 0]    5×5 → 3×3            [g h i]
[0 0 0 0 0]
```

**Analogi:** Seperti waktu kita foto keluarga — orang di pinggir juga harus masuk frame. Padding memastikan tepi gambar tidak terpotong.

**Di kode kita:** Ultralytics menangani padding secara otomatis. Yang kita atur cuma `imgsz=640`.

---

### 4. Stride

**Apa itu?** Stride adalah seberapa jauh kernel melompat setiap kali geser.

**Visual:**
```
Stride = 1 (geser 1 piksel):
┌──┐
│A │  Hasil: kotak rapat, detail halus
└──┘─┐
   ┌─┼──┐
   │ │B │
   └─┴──┘

Stride = 2 (lompat 2 piksel):
┌──┐      ┌──┐
│A │      │B │  Hasil: setengah ukuran,
└──┘      └──┘  lebih cepat, lebih kasar
```

#### Apa Fungsi Stride? (Kenapa Stride Ada?)

Stride punya **3 fungsi utama** dalam CNN:

**Fungsi 1: Mengecilkan Ukuran (Downsampling)**

Bayangkan gambar 640×640. Kalau stride selalu 1, setiap lapisan outputnya tetap 640×640 — tidak ada pengecilan. Akibatnya:
- Komputasi membengkak (semakin dalam, channel makin banyak → makin berat)
- Tidak ada hierarki skala (semua neuron lihat area sama kecil)

Dengan stride=2, di setiap lapisan ukuran turun setengah:
```
640 → 320 → 160 → 80 → 40 → 20
```
Ini yang memungkinkan CNN punya **pandangan bertahap** — dari detail ke全局.

**Fungsi 2: Memperbesar Receptive Field Tanpa Menambah Ukuran Kernel**

Kernel 3×3 dengan stride=1 hanya lihat 3×3 piksel. Tapi setelah stride=2, di lapisan berikutnya kernel 3×3 yang sama bisa "melihat" 7×7 piksel gambar asli — karena setiap langkahnya melompat 2 piksek.

```
Stride=1 terus:  RF tumbuh lambat
Layer 1: 3×3
Layer 2: 5×5
Layer 3: 7×7

Stride=2 di layer 2:  RF tumbuh cepat!
Layer 1: 3×3
Layer 2: 7×7   (lompat 2 → jangkauan lebih luas)
Layer 3: 15×15
```

**Fungsi 3: Menghemat Komputasi**

```
Stride=1:
  640×640 × 256 filter → 640×640×256 = 104 juta operasi per lapisan

Stride=2:
  640×640 × 256 filter → 320×320×256 = 26 juta operasi per lapisan
  → 4× lebih cepat!
```

#### Ibarat Stride dalam Kehidupan Sehari-hari

| Stride | Analogi | Akibat |
|--------|---------|--------|
| **Stride=1** | Baca buku kata per kata | Detail maksimal, lambat |
| **Stride=2** | Baca buku lompat 1 kata tiap langkah | Cepat, paham inti, kadang lewat detail |
| **Stride=4** | Baca buku scan per paragraf | Super cepat, cuma dapet garis besar |

YOLO pakai stride **bertahap**: stride=1 di awal (tangkap detail), lalu stride=2 berulang (perbesar pandangan, hemat komputasi).

#### Hubungan Stride dengan Grid P3/P4/P5

```
imgsz=640
    │
    ├─ Stride 8×  (3× stride=2)  → 640÷8  = 80   → P3 (80×80)  untuk objek kecil
    ├─ Stride 16× (4× stride=2)  → 640÷16 = 40   → P4 (40×40)  untuk objek sedang
    └─ Stride 32× (5× stride=2)  → 640÷32 = 20   → P5 (20×20)  untuk objek besar
```

Stride menentukan **seberapa rapat grid**. Makin besar stride → makin jarang grid → setiap sel lihat area makin luas → cocok untuk objek besar.

#### Di Kode

Stride=2 di-set di model YAML Ultralytics:

```yaml
# ultralytics/cfg/models/26/yolo26-seg.yaml
- [-1, 1, Conv, [256, 3, 2]]    # stride=2
```

Angka `2` di `[256, 3, 2]` artinya stride=2. Kalau diganti 1, ukuran tidak mengecil. Tapi arsitektur sudah tetap — tidak disarankan diubah.

**Ringkasan:** Stride adalah alat untuk **mengecilkan ukuran, memperluas pandangan, dan menghemat komputasi** secara bersamaan. Tanpa stride, CNN akan kebanyakan parameter dan lambat.

---

### 5. Receptive Field (Lapangan Pandang)

**Apa itu?** Receptive field adalah area piksel di gambar asli yang "dilihat" oleh satu neuron di lapisan dalam.

**Ilustrasi:**

```
Layer 1 (conv 3×3):
    Setiap neuron lihat 3×3 piksel gambar asli
    ┌───┐
    │ ◎ │  ← neuron ini lihat 9 piksel
    └───┘

Layer 2 (conv 3×3):
    Setiap neuron lihat 5×5 piksel karena menggabungkan informasi
    dari 9 neuron sebelumnya
    ┌─────┐
    │  ◎  │  ← neuron ini lihat 25 piksel
    └─────┘

Layer 3 (conv 3×3, stride=2):
    Setiap neuron lihat 7×7 piksel (makin lebar)
    ┌───────┐
    │   ◎   │  ← neuron ini lihat 49 piksel
    └───────┘
```

**Kenapa penting?**
- Lapisan awal: RF kecil → melihat detail halus (tepi, sudut)
- Lapisan tengah: RF sedang → melihat bagian objek (mata, roda, tutup botol)
- Lapisan akhir: RF besar → melihat objek utuh (wajah, mobil, botol)

**Hubungan dengan multi-scale grid YOLO:**
- P3 (80×80): RF kecil → objek kecil
- P4 (40×40): RF sedang → objek sedang
- P5 (20×20): RF besar → **objek besar** (contoh kita: box 1200×793)

**Analogi:** Seperti melihat foto dari jarak berbeda:
- Dekat: lihat detail (tekstur botol)
- Sedang: lihat bagian (tutup botol, label)
- Jauh: lihat keseluruhan (botol utuh)

---

### 6. Activation Function — SiLU

**Apa itu?** Activation function adalah "gerbang" yang menentukan apakah informasi dari neuron diteruskan atau tidak.

**Yang kita pakai: SiLU (Sigmoid Linear Unit)**

```
Rumus: f(x) = x × sigmoid(x)

Grafiknya:
f(x)
↑
│      ╱
│     ╱
│    ╱
│   ╱
│  ╱
│ ╱
╱
└──────────────────→ x
```

**Cara kerja sederhana:**
- Input positif besar → output hampir sama (tidak diubah)
- Input positif kecil → output sedikit dikecilkan
- Input negatif → output negatif kecil (bukan nol!)

**Kenapa SiLU, bukan ReLU?**

| Fungsi | ReLU | SiLU (punya kita) |
|--------|------|-------------------|
| Input negatif | Output 0 (mati total) | Output negatif kecil (masih hidup) |
| Gradien negatif | 0 (tidak belajar) | Ada (masih bisa belajar) |
| Bentuk | Patah di 0 | Mulus (smooth) |

**Analogi:**
- **ReLU** seperti saklar ON/OFF — kalau negatif, mati total
- **SiLU** seperti dimmer — kalau negatif, tidak mati total, masih ada sedikit aliran

**Kenapa SiLU lebih baik untuk YOLO?** Karena saat training, banyak neuron yang menerima sinyal negatif. Dengan ReLU, neuron itu mati total selamanya (dying ReLU). Dengan SiLU, neuron tetap hidup dan bisa belajar lagi.

**Di kode:** SiLU dipakai setelah setiap Conv + BatchNorm di seluruh backbone YOLO. Tidak perlu kita setting — sudah bawaan model Ultralytics.

---

### Kaitan Semua Konsep — Satu Cerita Utuh

Bayangkan CNN seperti **pabrik pemeriksaan**:

```
Gambar masuk
    ↓
┌── KONVOLUSI (Stempel + Dot Product) ──┐
│  "Cocokkan stempel (kernel) dengan     │
│   setiap bagian gambar. Ukur            │
│   kecocokan pakai dot product."         │
└────────────────────────────────────────┘
    ↓
┌── PADDING ───────────────────────────┐
│  "Tambah bingkai biar tepi gambar     │
│   tidak kepotong."                     │
└────────────────────────────────────────┘
    ↓
┌── STRIDE ────────────────────────────┐
│  "Lompat 2 langkah biar ukuran       │
│   berkurang dan proses lebih cepat."   │
└────────────────────────────────────────┘
    ↓
┌── AKTIVASI SiLU ────────────────────┐
│  "Filter informasi — yang penting     │
│   diteruskan, yang tidak direm."      │
└────────────────────────────────────────┘
    ↓
┌── LAPISAN BERIKUTNYA ───────────────┐
│  "Sekarang receptive field lebih     │
│   besar — lihat area lebih luas."     │
└────────────────────────────────────────┘
    ↓
(Diulang 4-5 kali, makin ke dalam makin luas RF-nya)
    ↓
Hasil akhir: model paham isi gambar dari detail sampai keseluruhan
```

---

### 7. CSP (Cross-Stage Partial) — Trik Khusus YOLO

**Apa itu?** CSP adalah cara menyusun lapisan CNN agar lebih efisien.

**Cara kerja:**
```
Input (256 channel)
    │
    ├→ 128 channel → konvolusi biasa (proses berat)
    │
    └→ 128 channel → langsung lewat (skip)
    │
    Gabung lagi → 256 channel → konvolusi ringan
```

**Kenapa?**
- Separuh channel tidak diproses berat → lebih cepat
- Gradien punya jalur pendek → tidak vanishing
- Informasi asli tetap terjaga (tidak berubah)

**Analogi:** Seperti fotokopi dokumen — daripada 256 halaman difotokopi semua (lama), cukup 128 halaman, 128 halaman sisanya langsung digabung.

---

### 8. Letterbox — Membuat Gambar Persegi Tanpa Merusak Proporsi

**Apa itu?** Letterbox adalah teknik menambahkan padding (biasanya hitam) di sisi pendek gambar agar menjadi persegi, **tanpa mengubah proporsi objek di dalamnya**.

**Mengapa YOLO butuh letterbox?** YOLO (dan CNN pada umumnya) hanya bisa menerima input **fixed-size persegi** — dalam kasus kita 640×640. Tapi gambar dari user bisa bermacam-macam ukuran: landscape (1200×800), portrait (800×1200), persegi (1000×1000), dll.

**Ada 3 cara membuat gambar jadi 640×640:**

| Metode | Cara | Hasil | Masalah |
|--------|------|-------|---------|
| **Stretch** | Paksa resize jadi 640×640 | ❌ Objek gepeng/melebar | Botol jadi pendek-gemuk — model bingung |
| **Crop** | Potong tengahnya saja 640×640 | ❌ Objek di tepi hilang | Sampah di pinggir foto tidak terdeteksi |
| **Letterbox** | Resize proporsional + padding hitam | ✅ Objek utuh, proporsi terjaga | Ada area hitam (tapi tidak masalah) |

**Visual perbandingan:**

```
Gambar asli 1200×800 (landscape):
┌────────────────────────────────────────────────────────────┐
│                                                            │
│                   □□ BOTOL PLASTIK □□                       │
│                                                            │
└────────────────────────────────────────────────────────────┘

STRETCH (paksa 640×640):
┌──────────────────┐
│░░░░░░░░░░░░░░░░░░│  ← botol jadi gepeng!
│░░░░ BOTOL ░░░░░░│
│░░░░ PLASTIK ░░░░│
│░░░░░░░░░░░░░░░░░░│
└──────────────────┘

CROP (ambil tengah):
┌──────────────────┐
│                  │
│   □□ BOTOL □□   │  ← botol utuh tapi
│                  │     sisi kiri kanan hilang
└──────────────────┘

LETTERBOX (resize 640×427 + padding hitam):
┌──────────────────┐
│██████████████████│  ← padding hitam 107px
│░░░░░░░░░░░░░░░░░░│
│░░░░ BOTOL ░░░░░░│  ← gambar 640×427
│░░░░ PLASTIK ░░░░│     (proporsi terjaga)
│░░░░░░░░░░░░░░░░░░│
│██████████████████│  ← padding hitam 106px
└──────────────────┘
640×640
```

**Cara menghitung letterbox:**

```
Gambar input: 1200×800
Target: 640×640

Step 1 — Hitung scale factor:
  scale = min(target_width / input_width, target_height / input_height)
        = min(640 / 1200, 640 / 800)
        = min(0.533, 0.8)
        = 0.533   ← ambil yang terkecil

Step 2 — Resize gambar proporsional:
  new_width  = 1200 × 0.533 = 640   (pas!)
  new_height = 800 × 0.533  = 427   (kurang dari 640)

Step 3 — Hitung padding:
  total_padding = 640 - 427 = 213
  padding_atas  = floor(213 / 2) = 106
  padding_bawah = 213 - 106 = 107

Step 4 — Tambah padding:
  ┌──────────────────┐
  │████ 106px hitam ██│
  │░░ 427px gambar ░░│
  │████ 107px hitam ██│
  └──────────────────┘
  640×640
```

**Apa yang terjadi pada koordinat saat letterbox?**

```
Titik (0, 0) di gambar asli → berubah karena ada padding.

Rumus konversi gambar asli → frame model:
  x_frame = x_asli × scale
  y_frame = y_asli × scale + padding_atas

Contoh:
  Titik (0, 7) di gambar asli 1200×800:
    x_frame = 0 × 0.533 = 0
    y_frame = 7 × 0.533 + 106 = 3.7 + 106 = 109.7

Nanti setelah model prediksi, Ultralytics otomatis scale balik:
  x_asli = (x_frame - padding_kiri) / scale
  y_asli = (y_frame - padding_atas) / scale

Contoh untuk box hasil model:
  x1 = (0 - 0) / 0.533 = 0
  y1 = (109.7 - 106) / 0.533 = 7.0  ← kembali ke nilai asli!
```

**Apakah padding hitam mengganggu model?** Tidak. Beberapa alasan:

1. **Model sudah dilatih dengan letterbox** — saat training, gambar juga di-letterbox dulu. Model terbiasa melihat area hitam di pinggir.
2. **Padding tidak mengandung objek** — neuron di area hitam akan mendapat input 0 (hitam = 0). Tidak ada pola yang terdeteksi, confidence-nya rendah, otomatis difilter.
3. **Objek tetap di tengah** — model fokus ke area tengah yang berisi gambar asli.

**Kapan letterbox bisa jadi masalah?**

| Situasi | Masalah | Solusi |
|---------|---------|--------|
| Objek sangat kecil | Setelah resize, objek makin kecil | Pakai input lebih besar (imgsz=1280) |
| Objek persis di tepi | Bisa kena potong letterbox | Pastikan foto objek di tengah |
| Rasio sangat ekstrim (misal 2000×200) | Banyak area hitam terbuang | Crop dulu sebelum upload |

**Di kode kita:** Semua penanganan letterbox dilakukan otomatis oleh library Ultralytics YOLO. Kita hanya perlu set `imgsz=640` di config. Backend dan frontend tidak perlu urus padding — model sudah handle sendiri.

**Hubungan dengan Bounding Box Report:**
- Tanpa letterbox: model tidak bisa memproses gambar 1200×800
- Dengan letterbox: model bisa predict, lalu scale balik ke koordinat asli
- Hasil akhir: `X1=0, Y1=7, X2=1200, Y2=800` — seolah-olah model melihat gambar 1200×800 langsung

---

### Ringkasan Cepat

| Konsep | Definisi 5 Kata | Analogi |
|--------|----------------|---------|
| **Kernel** | Pola kecil yang dicocokkan | Stempel |
| **Dot Product** | Ukur kecocokan dua hal | Kunci dengan lubang |
| **Padding** | Tambah pixel di tepi | Bingkai foto |
| **Stride** | Lompatan geser kernel | Langkah waktu jalan |
| **Receptive Field** | Area yang dilihat neuron | Jarak lihat foto |
| **SiLU** | Gerbang informasi neuron | Dimmer (bukan saklar) |
| **CSP** | Separuh diproses, separuh skip | Fotokopi selektif |

---

## Glossary (Istilah Penting)

| Istilah | Arti Sederhana |
|---------|----------------|
| **YOLO** | Program yang bisa lihat gambar dan kasih tahu ada apa di dalamnya |
| **Bounding Box** | Kotak persegi yang mengelilingi objek yang terdeteksi |
| **Confidence** | Skor keyakinan model (0-100%) — seberapa yakin deteksinya benar |
| **Label** | Nama objek yang terdeteksi (Organik atau Non-Organik) |
| **Category** | Sama seperti label, untuk grouping |
| **X1, Y1** | Posisi sudut kiri-atas kotak (dalam piksel) |
| **X2, Y2** | Posisi sudut kanan-bawah kotak (dalam piksel) |
| **W, H** | Lebar dan tinggi kotak (dalam piksel) |
| **Letterbox** | Nambah padding hitam biar gambar jadi persegi tanpa merusak proporsi |
| **NMS** | Proses milih box terbaik dari banyak box yang tumpang tindih |
| **Backend** | Server yang proses gambar pakai Python |
| **Frontend** | Browser yang tampilkan hasil ke user |
| **Grid** | Kotak-kotak bayangan yang YOLO pakai untuk menganalisis gambar |
| **DFL** | Cara YOLO nebak posisi box — pakai distribusi probabilitas (tebak 16 kemungkinan, ambil rata-rata) |
