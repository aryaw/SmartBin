# Presentation Detail Discussion
## Panduan Presentasi untuk Tim SmartBin
### Gaya: Professor mengajar mahasiswa -- Sederhana, mendalam, dan menyenangkan

---

## Slide 1: Judul
**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

### Cara Menyampaikan (Narasi)

"Selamat pagi/siang, teman-teman. Hari ini kita akan membahas proyek SmartBin -- sebuah sistem yang bisa mendeteksi dan mengklasifikasikan sampah secara otomatis menggunakan kecerdasan buatan (AI). Bayangkan sebuah tempat sampah pintar yang bisa tahu: 'Oh, ini sampah organik, buang ke komposter!' atau 'Ini botol plastik, masuk ke daur ulang!'. Itulah yang kita bangun."

"Judul penelitian kita panjang, tapi mari kita bedah satu-satu:"

**Deteksi dan Klasifikasi Sampah** -- Kita membuat komputer bisa melihat sampah dan tahu jenisnya.

**Menggunakan YOLOv26m-seg** -- Ini arsitektur AI terbaru untuk computer vision. YOLO singkatan dari 'You Only Look Once' -- sekali lihat langsung tahu. Bayangkan kalau kamu melihat foto, dalam sekejap kamu bisa tahu ada apa saja di foto itu. YOLO melakukan hal yang sama!

**Instance Segmentation** -- Ini lebih canggih dari sekedar deteksi. Kalau deteksi biasa cuma bikin kotak pembatas (bounding box), segmentasi bisa membuat 'topeng' (mask) presisi yang mengikuti bentuk objek. Seperti mewarnai objek persis mengikuti bentuknya.

**2 Kelas: Organik/Non-Organik** -- Kita sederhanakan jadi 2 kelompok saja. Sampah organik (sisa makanan, daun) dan non-organik (plastik, kaca, logam, dll).

### Diagram Flow -- Penjelasan per Bagian

Diagram besar di slide ini adalah peta jalan seluruh proyek kita. Mari kita jelajahi dari kiri ke kanan.

**Source 1: TACO Dataset (Kotak Biru Pertama)**

TACO singkatan dari Trash Annotations in Context. Ini dataset publik yang berisi 1.500 foto sampah dari berbagai tempat. Yang istimewa: setiap foto sudah memiliki anotasi lengkap -- ada polygon yang menggambar bentuk setiap objek sampah, dan ada kotak pembatas (bounding box). Ada 60 kategori sampah berbeda, dari aluminium foil, baterai, botol plastik, sampai puntung rokok.

Tapi kita hanya butuh 2 kelas. Maka kita petakan: kategori 25 (Food waste / sisa makanan) menjadi Organik. 59 kategori lainnya menjadi Non-Organik.

"Anggap saja kita punya lemari file raksasa dengan 60 laci. Kita ambil semua isi laci, lalu kita kelompokkan ulang cuma ke 2 kotak besar: Organik dan Non-Organik."

**Source 2: Waste Classification Dataset (Kotak Biru Kedua)**

Dataset kedua berasal dari folder-folder yang sudah dikelompokkan secara manual. Ada 2.939 foto yang tersimpan dalam 18 subfolder. Misalnya folder 'coffee_tea_bags' berisi foto ampas kopi dan teh celup. Folder 'egg_shells' berisi foto kulit telur. Ini semua masuk Organik.

Sebaliknya, folder 'e-waste' (sampah elektronik), 'cans_all_type' (kaleng), 'glass_containers' (botol kaca) masuk Non-Organik.

"Bayangkan teman-teman kita sudah memilah ribuan foto sampah ke dalam ember-ember kecil. Tugas kita adalah menggabungkan semua ember kecil ini ke 2 ember besar."

**Merge & Classify (Kotak Tengah)**

Di sinilah penggabungan terjadi. Ada script Python di `backend/app/services/kaggle_service.py` fungsi `prepare_from_local()`. Script ini:

1. Membaca semua folder TACO dan Waste Classification
2. Untuk setiap foto, tentukan: apakah ini Organik (kelas 0) atau Non-Organik (kelas 1)?
3. Salin foto ke folder tujuan: `backend/dataset/raw/Organik/` atau `Non-Organik/`

Hasil akhir: 684 foto Organik, 3.289 foto Non-Organik. Total 3.973 foto.

"Perhatikan: ada ketidakseimbangan. Non-Organik hampir 5 kali lipat dari Organik. Ini tantangan yang akan kita bahas nanti."

**Pseudo-Polygon Mask Generation (Kotak Proses Besar)**

"Nah, ini bagian yang paling kreatif dari proyek kita."

Masalahnya: dataset kita hanya berupa foto biasa tanpa anotasi mask. Padahal model YOLO butuh mask untuk belajar segmentasi. Solusi kita: buat mask palsu (pseudo-mask) secara otomatis!

Caranya:

*Langkah 1: Edge Detection Otsu (66.4% kasus)*

1. Foto berwarna (RGB) diubah jadi hitam-putih (grayscale)
2. Diblur supaya noise berkurang (Gaussian Blur 5x5)
3. Otsu thresholding: komputer secara otomatis menentukan nilai batas (threshold) untuk memisahkan objek dari latar belakang. Hasilnya: gambar biner hitam-putih
4. Jika rata-rata piksel > 127, kita balik warnanya (invert) -- ini untuk foto dengan latar belakang putih
5. Operasi morphological: close (menutup lubang kecil) dan open (menghilangkan titik noise)
6. Cari kontur (garis tepi) objek, ambil yang terbesar
7. Jika luas kontur < 20% total gambar, berarti gagal -- fallback ke metode lain
8. Kontur disederhanakan menjadi polygon (24 titik)
9. Koordinat dinormalisasi ke [0, 1]

"Presentasi penuh: bayangkan kamu menggambar bentuk daun dengan 24 titik penghubung. Makin banyak titik, makin presisi bentuknya."

*Langkah 2: Fallback Geometris (33.6% kasus)*

Untuk foto dimana edge detection gagal (biasanya karena objek tidak kontras dengan latar), kita pakai bentuk geometris sederhana:
- 60%: bentuk elips (seperti telur)
- 40%: bentuk persegi panjang dengan sudut membulat (seperti kertas)

"Tentu ini tidak seakurat edge detection, tapi lebih baik daripada tidak ada mask sama sekali."

**Stratified Split 70/15/15**

Dataset harus dibagi 3 untuk training:
- Train (70% = 2.765 foto): untuk belajar
- Val (15% = 593 foto): untuk evaluasi selama training
- Test (15% = 593 foto): untuk evaluasi akhir

"Stratified" artinya proporsi Organik/Non-Organik dijaga sama di setiap bagian. Kalau total ada 17% Organik, maka train, val, dan test masing-masing juga punya ~17% Organik. Ini penting supaya evaluasi adil.

**Online Augmentation**

"Ini seperti memberi kacamata berbeda pada model kita setiap kali belajar."

Karena data terbatas (hanya 3.973 foto), kita perlu membuat variasi buatan:
- Mosaic (1.0): gabung 4 foto jadi 1. Model belajar mendeteksi objek dalam konteks yang kaya
- Mixup (0.2): campur 2 foto secara transparan. Model belajar dari kombinasi
- Copy-Paste (0.15): salin objek dari satu foto ke foto lain
- HSV Jitter: ubah warna (Hue), saturasi, dan kecerahan (Value) secara acak
- Geometric: putar (rotate), perbesar/perkecil (scale), miringkan (shear)
- Flip: balik kiri-kanan (horizontal flip)

"Anggap saja model kita punya 3.973 foto asli. Tapi setiap epoch (satu putaran belajar penuh), model bisa melihat variasi berbeda dari setiap foto. Efeknya: model tidak mudah overfit (hafal) dan lebih generalis."

**Backbone: CSPDarknet**

Backbone adalah 'mata' dari model kita. Tugasnya: mengekstrak fitur dari gambar input.

"Bayangkan kamu melihat foto botol plastik. Pertama kamu lihat tepi botol, tekstur plastik, bayangan, dan seterusnya. Semakin dalam kamu memperhatikan, semakin detail informasi yang kamu dapat. Backbone melakukan hal yang sama."

Prosesnya:
1. Input: 640x640 pixel (3 channel warna RGB)
2. Stem Conv: convolution layer pertama mereduksi ukuran sambil mengekstrak fitur dasar
3. Stage 1-4: setiap stage mereduksi ukuran setengahnya, tapi menggandakan 'kedalaman' informasi
   - Stage 1: 320x320 -> 160x160 (deteksi tepi dasar)
   - Stage 2: 160x160 -> 80x80 (deteksi bentuk sederhana)
   - Stage 3: 80x80 -> 40x40 (deteksi tekstur dan pola)
   - Stage 4: 40x40 -> 20x20 (deteksi konteks dan semantik)
4. SPP Layer: Spatial Pyramid Pooling dengan 3 ukuran pooling (5, 9, 13). "Seperti melihat objek dengan 3 kaca pembesar berbeda secara bersamaan."

CSPNet (Cross Stage Partial Network) adalah teknik khusus yang membagi feature map jadi 2 jalur. Satu jalur diproses, satu jalur langsung diteruskan. Hasilnya: lebih efisien (~20% lebih hemat komputasi) dan gradien mengalir lebih baik saat training.

**Neck: FPN + PAN**

"Backbone sudah mengekstrak fitur, tapi fitur di level bawah (20x20) tahu 'apa' objeknya tapi tidak tahu 'di mana' persisnya. Fitur di level atas (80x80) tahu 'di mana' tapi tidak tahu 'apa'. Neck menggabungkan keduanya."

FPN (Feature Pyramid Network): dari atas ke bawah. Bawa informasi semantik dari level rendah ke level tinggi.

PAN (Path Aggregation Network): dari bawah ke atas. Bawa informasi lokasi presisi dari level tinggi ke level rendah.

Hasilnya: setiap level deteksi (P3=80x80, P4=40x40, P5=20x20) memiliki INFORMASI LENGKAP: tahu apa objeknya DAN di mana letaknya.

"Seperti kamu punya 3 detektif: Detektif P3 ahli objek kecil (tutup botol), Detektif P4 ahli objek sedang (kaleng), Detektif P5 ahli objek besar (kardus). Mereka saling berbagi informasi untuk hasil terbaik."

**Decoupled Head**

Head adalah 'otak' yang membuat keputusan akhir. YOLOv26 menggunakan 'decoupled head': setiap tugas (klasifikasi, regresi, segmentasi) punya jalur saraf sendiri.

1. Classification Branch: "Ini objek apa?" -- output 2 angka: probabilitas Organik dan Non-Organik
2. Regression Branch: "Di mana objeknya?" -- output 4 koordinat bounding box (x_center, y_center, width, height)
3. Segmentation Branch: "Bagaimana bentuk objek?" -- output mask polygon 24 titik

Anchor-Free: YOLO26 tidak menggunakan 'anchor box' (template bentuk) seperti YOLO versi lama. Setiap grid cell (kotak kecil di gambar) bertanggung jawab mendeteksi objek yang pusatnya ada di dalam cell tersebut.

"Bayangkan gambar 640x640 dibagi jadi 8.400 kotak kecil (grid). Setiap kotak bisa mendeteksi SATU objek. Jika pusat objek ada di kotak itu, kotak itu yang bertanggung jawab."

**Loss Functions**

Loss function adalah 'pengukur kesalahan'. Semakin kecil loss, semakin baik model.

1. CIoU Loss (weight 7.5): mengukur seberapa baik bounding box prediksi dibandingkan dengan yang sebenarnya. CIoU = Complete IoU, mempertimbangkan:
   - IoU (Intersection over Union): seberapa besar tumpang tindih
   - Center Distance: jarak antara pusat prediksi dengan pusat sebenarnya
   - Aspect Ratio: perbedaan bentuk (lebar/tinggi)

2. BCE Loss (weight 0.5): Binary Cross-Entropy untuk klasifikasi. 'Seberapa yakin model bahwa ini organik vs non-organik?'

3. DFL Loss (weight 1.5): Distribution Focal Loss untuk regresi posisi. Uniknya, DFL memprediksi DISTRIBUSI probabilitas posisi, bukan nilai tunggal. "Model tidak dipaksa memilih satu titik yang mungkin salah. Model bisa bilang 'tepi botol kemungkinan ada di rentang ini'."

Total Loss = 7.5 x CIoU + 0.5 x BCE + 1.5 x DFL

Bobot berbeda karena setiap tugas punya skala berbeda. Box loss (CIoU) paling penting untuk deteksi presisi.

**Training**

"Setelah semua siap, kita mulai training. Model kita (YOLOv26m-seg) sudah punya pengetahuan dasar dari COCO dataset (Common Objects in Context -- 80 kelas objek umum). Kita fine-tuning (menyesuaikan) pengetahuan itu ke domain sampah."

Proses training:
1. Model melihat batch 16 foto
2. Forward pass: model memprediksi kelas, bbox, mask
3. Hitung loss antara prediksi dan ground truth
4. Backward pass: hitung gradien (arah perbaikan)
5. Optimizer (SGD/MuSGD): update bobot model berdasarkan gradien
6. Ulangi sampai 80 epoch (80 kali lihat seluruh dataset)

FP16 mixed precision: training menggunakan format angka 16-bit (setengah presisi) untuk sebagian operasi. Ini menghemat VRAM ~44% tanpa mengorbankan akurasi secara signifikan.

Early stopping patience=40: jika dalam 40 epoch tidak ada peningkatan, training berhenti lebih awal.

**Evaluation Results**

Setelah training, model dievaluasi pada test set (593 foto yang tidak pernah dilihat model selama training).

Box mAP@0.5: 80.4%. Artinya, dengan IoU threshold 0.5, model mencapai Average Precision 80.4%. "Dari 100 objek, model mendeteksi ~80 dengan benar dan presisi."

Mask mAP@0.5: 49.7%. Lebih rendah karena mask butuh prediksi boundary presisi, sementara ground truth mask kita adalah pseudo-mask (tidak sempurna).

Perhatian: Non-Organik (83.6%) lebih baik dari Organik (77.2%). Kenapa? Karena data Non-Organik ~5x lebih banyak dan bentuknya lebih seragam (botol, kaleng, kaca -- bentuk rigid).

**Backend & Frontend**

"Terakhir, model kita harus bisa dipakai oleh pengguna. Kita buat API (Application Programming Interface) dengan FastAPI dan website dengan Nuxt.js."

POST /api/detect: endpoint utama. Upload foto -> model infer -> return hasil annotated + JSON.

/dashboard: halaman utama. Upload foto, lihat hasil deteksi, jumlah Organik/Non-Organik.

"Flow: Buka website -> Upload foto sampah -> Klik Deteksi -> Lihat hasil annotasi + rekomendasi."

**Model Deployment**

Model yang sudah dilatih disimpan di `runs/segment/full_pipeline/weights/best.pt` (ukuran 54.5 MB). Pipeline menyalinnya ke `backend/models/best.pt`. Saat ada request deteksi, model dimuat ke GPU, infer dalam 5.3ms, lalu hasilnya dikembalikan.

---

## Slide 2: Outline Presentasi

### Narasi

"Kita akan melewati 7 segmen utama dalam presentasi ini. Bayangkan kita naik gunung bersama:"

1. **Latar Belakang** (Base Camp) -- Mengapa kita melakukan ini? Masalah sampah di Bali.
2. **Rumusan Masalah & Tujuan** (Peta Pendakian) -- Apa yang ingin kita capai?
3. **Dataset** (Bekal) -- Data apa yang kita punya? Dari mana asalnya?
4. **Metode** (Jalur Pendakian) -- Bagaimana cara kita mencapai tujuan?
5. **Hasil** (Puncak) -- Apa yang berhasil kita capai?
6. **Aplikasi Web** (Pos Pendinginan) -- Bagaimana hasil ini bisa dipakai?
7. **Kesimpulan & Saran** (Turun Gunung) -- Apa yang kita pelajari? Langkah selanjutnya?

### Waktu Presentasi
- Total: 15-20 menit
- Per slide: ~1.5-2 menit
- Q&A: 5-10 menit

"Waktu kita terbatas, jadi setiap slide harus padat informasi. Fokus pada 'apa', 'kenapa', dan 'bagaimana'."

---

## Slide 3: Latar Belakang

### Narasi

"Teman-teman, mari kita mulai dengan masalah nyata."

**Krisis Sampah di Bali: Angka-angka yang Mencemaskan**

"Bali menghasilkan 1.340 ton sampah SETIAP HARI. Coba bayangkan: 1 ton sampah = 10 kali berat badan kita. 1.340 ton = berat 10.000 orang dewasa! Setiap hari!"

"Dari mana sampah sebanyak ini? 60% dari pariwisata. Setiap turis menghasilkan ~3.5 kg sampah per hari (lebih banyak dari penduduk lokal)."

"Komposisinya: 60% organik (sisa makanan, daun, sayur busuk), 30% plastik (botol, kantong, kemasan), 10% sisanya (logam, kaca, kertas, dll)."

**Masalah Utama: Pemilahan Manual**

"Siapa di sini yang pernah disuruh memilah sampah? Pasti repot, kan? Apalagi kalau volumenya 1.340 ton per hari. Petugas kebersihan harus memilah manual -- lambat, tidak efisien, dan berbahaya (tertusuk jarum, terkena bahan kimia)."

"Belum lagi kesalahan pemilahan. Sampah organik tercampur non-organik, bikin proses daur ulang jadi kacau. Atau sampah berbahaya (baterai, pestisida) masuk ke tempat sampah biasa."

**Solusi: Pergub No.47/2019 + AI**

"Pemerintah Bali melalui Pergub No.47/2019 menetapkan standar klasifikasi: Organik (kompos), Anorganik (daur ulang), Residu (TPA). Tapi implementasinya masih manual."

"Kita hadirkan solusi: menggunakan AI / deep learning untuk otomatisasi. Komputer bisa 'melihat' sampah dan mengklasifikasikannya secara real-time."

**Kenapa Computer Vision?**

"Manusia bisa mengenali sampah dengan melihat. Komputer juga bisa -- setelah dilatih dengan ribuan contoh. Computer vision adalah cabang AI yang memungkinkan komputer 'melihat' dan 'memahami' gambar."

**Kenapa Instance Segmentation?**

"Ada 3 level kemampuan computer vision:"

1. **Klasifikasi** (paling sederhana): "Ini foto sampah organik." -- Cuma label.
2. **Deteksi / Bounding Box** (sedang): "Ini sampah organik di kotak ini." -- Ada lokasi perkiraan.
3. **Segmentasi / Instance Segmentation** (paling detail): "Ini sampah organik, tepatnya di area ini." -- Bentuk presisi.

"Kita pilih level 3 karena sampah sering bertumpuk dan punya bentuk tidak beraturan. Bounding box tidak cukup presisi untuk memisahkan objek yang saling tumpang tindih."

**Kenapa 2 Kelas Saja?**

"Kita cuma pakai 2 kelas: Organik dan Non-Organik. Kenapa tidak 60 kelas seperti TACO?"

"Jawabannya: validasi. Kalau kita punya 60 kelas, siapa yang bisa memverifikasi hasil deteksi model? Hanya ahli sampah. Tapi dengan 2 kelas, SEMUA ORANG bisa bilang: 'Oh, ini bener organik' atau 'Ini bukan organik'."

"Non-Organik nanti dipetakan lagi ke Anorganik (recyclable) dan Residu (landfill) di backend. Tapi untuk model, 2 kelas sudah cukup."

---

## Slide 4: Dataset - Dua Sumber

### Narasi

"Model AI kita butuh data -- banyak data. Semakin banyak dan beragam data, semakin pintar modelnya. Kita punya 2 sumber data."

**Source 1: TACO (Trash Annotations in Context)**

"TACO adalah dataset publik dari peneliti di Italia. Mereka mengambil 1.500 foto sampah dari berbagai tempat (pantai, taman, jalan, rumah) dan MENGANOTASI setiap objek sampah dengan detail."

"Apa itu anotasi? Bayangkan kamu harus menggambar garis tepi setiap botol, setiap kantong, setiap daun di dalam foto. Terus kasih label: 'ini botol plastik', 'ini daun'. Itu anotasinya."

"TACO punya 60 kategori. Contoh:"
- Kategori 0: Aluminium foil
- Kategori 1: Baterai
- Kategori 4: Botol plastik
- Kategori 5: Botol plastik bening
- Kategori 6: Botol kaca
- Kategori 25: Sisa makanan (FOOD WASTE -- ini satu-satunya yang kita anggap ORGANIK)
- Kategori 58: Sampah tidak terlabeli
- Kategori 59: Puntung rokok

"Bayangkan sebuah lemari dengan 60 laci. Setiap laci berisi foto-foto sampah sejenis."

**Source 2: Waste Classification Dataset (Lokal)**

"Ini dataset kedua, dari koleksi pribadi tim kita. Ada 2.939 foto yang sudah diorganisir dalam folder-folder berdasarkan jenisnya."

"Organik: coffee_tea_bags (157 foto), egg_shells (125 foto), food_scraps (147 foto), kitchen_waste (117 foto), yard_trimmings (131 foto)."

"Non-Organik dibagi 2 subkelas:"
- "Anorganik (bisa didaur ulang): e-waste (544), cans (272), glass (142), paper (121), plastic_bottles (130)"
- "Residu (ke TPA): batteries (114), paints (153), pesticides (139), ceramic (139), diapers (145), plastics_bags (135), sanitary_napkin (110), styrofoam (118)"

"Perhatikan: e-waste (sampah elektronik) paling banyak dengan 544 foto. Ini karena sampah elektronik punya variasi bentuk paling banyak."

**Proses Merge (Penggabungan)**

"Kita gabungkan 2 dataset ini jadi satu."

"Script `prepare_from_local()` di file `backend/app/services/kaggle_service.py` melakukan:"

1. "Baca semua foto di TACO. Untuk setiap foto, cek kategorinya. Jika Food Waste (cat 25) -> Organik. Selain itu -> Non-Organik."
2. "Baca semua foto di Waste Classification. Folder 'coffee_tea_bags', 'egg_shells', dll -> Organik. Selain itu -> Non-Organik."
3. "Salin setiap foto ke folder tujuan: `backend/dataset/raw/Organik/` atau `Non-Organik/`."

"Hasil akhir:"

| Kelas | Jumlah |
|-------|--------|
| Organik | 684 |
| Non-Organik | 3.289 |
| **Total** | **3.973** |

"Lihat perbandingannya: Non-Organik 4.8 kali lebih banyak dari Organik. Ini disebut CLASS IMBALANCE. Akan kita bahas nanti."

---

## Slide 5: Split & Pseudo-Mask

### Narasi

"Dua tantangan: (1) Dataset harus dibagi untuk training/validasi/test. (2) Dataset tidak punya anotasi mask -- kita harus membuatnya."

**Stratified Split 70/15/15**

"Pertama, kita bagi dataset jadi 3 bagian:"

- 70% untuk TRAINING: model belajar dari data ini
- 15% untuk VALIDASI: selama training, kita evaluasi model dengan data ini untuk cek progress
- 15% untuk TEST: setelah training selesai, kita uji model dengan data BARU yang tidak pernah dilihat sebelumnya

"Kenapa harus stratified? Stratified artinya 'berlapis'. Proporsi Organik/Non-Organik di train, val, dan test harus SAMA dengan proporsi di dataset total."

"Contoh: Kalau total Organik 17% dan Non-Organik 83%, maka di train juga 17%/83%, di val juga, di test juga."

"Kenapa penting? Supaya evaluasi ADIL. Kalau train punya 50% Organik tapi test punya 10% Organik, model kelihatan jelek padahal sebenarnya tidak (karena distribusi berbeda)."

**Pseudo-Polygon Mask Generation**

"Tantangan: Dataset kita tidak punya anotasi mask. Tapi YOLO butuh mask untuk belajar segmentasi. Solusi: kita BUAT masknya sendiri."

"Prinsipnya sederhana: ambil objek dari latar belakang (background subtraction), lalu gambar tepinya."

*Metode 1: Edge Detection Otsu (berhasil 66.4%)*

"Langkah-langkahnya:"

1. **RGB ke Grayscale**: Foto berwarna diubah ke hitam-putih. Kenapa? Karena kita hanya butuh informasi terang/gelap untuk membedakan objek dari latar.
2. **Gaussian Blur 5x5**: Foto di-blur (dikaburkan) sedikit. Kenapa? Untuk menghilangkan noise (bintik-bintik acak) yang bisa mengganggu deteksi tepi.
3. **Otsu Thresholding**: Komputer menentukan nilai batas (threshold) secara OTOMATIS. Piksel yang lebih terang dari threshold jadi putih (objek), yang lebih gelap jadi hitam (latar), atau sebaliknya.
4. **Invert if mean > 127**: Jika rata-rata piksel > 127 (kebanyakan putih), balik warnanya. Ini berguna untuk foto dengan latar putih.
5. **Morphological Close + Open**: Operasi untuk 'membersihkan' hasil. Close = menutup lubang kecil di dalam objek. Open = menghilangkan titik-titik putih kecil di luar objek.
6. **Find Contours**: Cari garis tepi objek.
7. **Ambil Largest**: Ambil kontur terbesar (asumsi: ini objek utama).
8. **Cek Area**: Jika luas kontur < 20% total gambar, berarti gagal (mungkin objek terlalu kecil atau tidak terdeteksi dengan baik).
9. **Approximate Polygon**: Sederhanakan kontur menjadi polygon dengan epsilon = 0.01 x arcLength. Ini mengurangi jumlah titik dari ratusan menjadi ~24 titik.
10. **Normalize**: Bagi semua koordinat dengan lebar/tinggi gambar. Hasilnya: koordinat antara 0 dan 1.

"Presentasi analogi: Bayangkan kamu menggambar bentuk daun di kertas transparan. Mula-mula kamu jiplak dengan pensil (deteksi tepi), lalu kamu hapus detail kecil yang tidak perlu (approximate), lalu kamu perbesar/perkecil gambarnya (normalisasi)."

*Metode 2: Fallback Geometris (33.6% kasus)*

"Edge detection gagal kalau objek tidak kontras dengan latar. Contoh: pisang di atas meja kayu (sama-sama coklat). Untuk kasus ini, kita pakai bentuk geometris sederhana:"

- 60%: bentuk ELIPS (mirip telur)
- 40%: bentuk PERSEGI PANJANG dengan sudut membulat (mirip kertas)

"Tidak ideal, tapi lebih baik daripada tidak ada mask sama sekali."

**Format Label YOLO-seg**

"Setelah mask jadi, kita simpan dalam format:"

```
<class_id> x1 y1 x2 y2 x3 y3 ... xn yn
```

"Contoh untuk Organik dengan 3 titik:"
```
0 0.25 0.30 0.45 0.35 0.30 0.50
```

"Artinya: kelas 0 (Organik), dengan polygon 3 titik di koordinat (0.25,0.30), (0.45,0.35), (0.30,0.50)."

"Semua koordinat antara 0 dan 1 (ternormalisasi). Jika gambar 640x640, koordinat 0.5 = 320 pixel."

---

## Slide 6: Arsitektur YOLOv26m-seg

### Narasi

"YOLOv26 adalah generasi terbaru dari keluarga YOLO. YOLO = You Only Look Once. Sekali lihat langsung tahu."

"Bayangkan kamu masuk kamar, dalam sekejap kamu bisa tahu ada meja, kursi, lampu, dan buku. Kamu tidak perlu memindai kamar pixel per pixel. YOLO melakukan hal yang sama pada gambar."

**Varian YOLOv26**

"YOLOv26 punya beberapa varian ukuran:"

| Varian | Parameter | FLOPs | mAP COCO | Kecepatan |
|--------|-----------|-------|----------|-----------|
| YOLO26n (nano) | 2.7M | 18.1G | 40.1% | Tercepat |
| YOLO26s (small) | 9.8M | 62.3G | 47.8% | Cepat |
| **YOLO26m (medium)** | **23.5M** | **121.2G** | **52.5%** | **Seimbang** |
| YOLO26l (large) | 48.0M | 263.8G | 53.5% | Lambat |
| YOLO26x (xlarge) | 98.9M | 520.7G | 54.1% | Terlambat |

"Kita pilih varian MEDIUM. Kenapa? Karena keseimbangan antara akurasi dan kecepatan. Varian nano/small kurang akurat. Varian large/xlarge butuh GPU lebih besar (VRAM > 24GB). Medium paling cocok untuk RTX 5060 Ti 16 GB kita."

"FLOPs = Floating Point Operations per second. Ukuran seberapa banyak komputasi yang dibutuhkan. 121.2 GFLOPs artinya 121.2 milyar operasi per detik."

**Backbone: CSPDarknet (Mata Model)**

"Backbone adalah fondasi model. Tugasnya: mengubah gambar 640x640 pixel menjadi fitur-fitur yang bisa dipahami model."

"Prosesnya bertahap:"

Input: 640x640 pixel, 3 channel (RGB)

Stage 1 (320x320 -> 160x160): Deteksi fitur dasar -- tepi, sudut, gradien warna. "Seperti melihat garis-garis kasar."

Stage 2 (160x160 -> 80x80): Deteksi pola sederhana -- lingkaran, kotak, garis lurus. "Seperti melihat bentuk dasar."

Stage 3 (80x80 -> 40x40): Deteksi tekstur dan pola kompleks -- serat kertas, kilau plastik, pori-pori daun. "Seperti melihat permukaan benda."

Stage 4 (40x40 -> 20x20): Deteksi konteks dan semantik -- 'ini benda buatan pabrik', 'ini benda alam'. "Seperti memahami esensi objek."

SPP Layer: Spatial Pyramid Pooling dengan 3 ukuran kernel (5, 9, 13). "Seperti melihat objek dengan 3 kaca pembesar berbeda secara bersamaan. Detail kecil dan besar tertangkap semua."

"CSPNet adalah teknik yang membuat backbone lebih efisien. Biasanya, setiap layer memproses SEMUA informasi. CSPNet membagi informasi jadi 2 jalur: satu diproses, satu langsung diteruskan. Ibaratnya, daripada semua orang antri di satu pintu, kita buka 2 pintu. Hasilnya: lebih cepat, lebih hemat energi."

**Neck: FPN + PAN (Jembatan Informasi)**

"Backbone memberi kita 3 level fitur dengan karakteristik berbeda:"

- P3 (80x80): Detail lokasi bagus, tapi kurang paham konteks
- P4 (40x40): Sedang-sedang
- P5 (20x20): Paham konteks, tapi detail lokasi kurang

"Neck menggabungkan ketiganya. FPN (Feature Pyramid Network) bekerja dari atas ke bawah: bawa pemahaman konteks dari P5 ke P3 dan P4. PAN (Path Aggregation Network) bekerja dari bawah ke atas: bawa detail lokasi dari P3 ke P4 dan P5."

"Hasilnya? Setiap level sekarang PUNYA SEGALANYA: tahu 'apa' objeknya DAN 'di mana' letaknya."

"Bayangkan 3 detektif: Detektif A jago lihat detail kecil, Detektif B jago lihat sedang, Detektif C jago lihat besar. Mereka saling bertukar informasi. Sekarang Detektif A juga bisa lihat besar, Detektif C juga bisa lihat detail."

**Decoupled Head (Otak Model)**

"Head adalah pengambil keputusan. Dipisah (decoupled) jadi 3 cabang:"

1. **Classification Branch**: Menjawab "Ini sampah apa?"
   - Input: fitur dari Neck
   - Proses: 2 layer konvolusi + 1 layer linear
   - Output: 2 angka (probabilitas Organik, probabilitas Non-Organik)
   - Contoh: [0.92, 0.08] artinya 92% yakin Organik, 8% yakin Non-Organik

2. **Regression Branch**: Menjawab "Di mana letak sampah ini?"
   - Input: fitur dari Neck
   - Proses: 2 layer konvolusi + DFL (Distribution Focal Loss)
   - Output: 4 koordinat (x_center, y_center, width, height)
   - Uniknya, DFL memprediksi DISTRIBUSI, bukan nilai tunggal. "Model bilang: 'tepi kiri kemungkinan di pixel 100-120', bukan 'tepi kiri di pixel 105'."
   - Keuntungan: lebih akurat untuk objek dengan boundary tidak jelas (botol transparan, sampah remuk)

3. **Segmentation Branch**: Menjawab "Bagaimana bentuk sampah ini?"
   - Input: fitur dari Neck (semua level)
   - Proses: Proto Module menggabung semua level, menghasilkan 32 'prototype mask' dasar
   - 32 prototype = 32 bentuk dasar (lingkaran, kotak, segitiga, panjang, dll)
   - Untuk setiap objek, model memilih kombinasi prototype: "Objek ini 80% prototype 8 + 30% prototype 15 + 10% prototype 3"
   - Output: mask polygon 24 titik

**Anchor-Free Detection**

"YOLO versi lama menggunakan 'anchor boxes' -- template bentuk yang sudah ditentukan sebelumnya (misalnya: kotak 50x50, 100x50, 200x100, dll). Model kemudian 'memilih' anchor mana yang paling cocok."

"YOLOv26 TIDAK menggunakan anchor. Setiap grid cell (kotak kecil di gambar) langsung memprediksi bounding box-nya sendiri. Lebih sederhana, lebih cepat, dan tidak perlu tuning anchor per dataset."

"Total grid: 80x80 + 40x40 + 20x20 = 8.400 grid cell. Setiap cell bisa mendeteksi 1 objek."

**Loss Functions (Pengukur Kesalahan)**

"Model butuh cara untuk mengukur seberapa salah prediksinya. Inilah loss function:"

1. **CIoU Loss (bobot 7.5)** -- untuk bounding box
   - IoU: seberapa besar tumpang tindih antara prediksi dan yang sebenarnya
   - Center distance: jarak pusat prediksi vs pusat sebenarnya
   - Aspect ratio: perbedaan bentuk
   - Kenapa bobotnya paling besar (7.5)? Karena mendeteksi LOKASI dengan presisi adalah prioritas utama. "Lebih baik salah label tapi kotaknya tepat, daripada label benar tapi kotaknya meleset."

2. **BCE Loss (bobot 0.5)** -- untuk klasifikasi
   - Binary Cross-Entropy: mengukur seberapa yakin model pada kelas yang benar
   - Bobot kecil (0.5) karena klasifikasi 2 kelas relatif mudah dibanding lokalisasi

3. **DFL Loss (bobot 1.5)** -- untuk regresi posisi
   - Distribution Focal Loss: mengukur kualitas distribusi probabilitas posisi
   - Bobot sedang (1.5) karena penting untuk presisi boundary

"Total Loss = 7.5 x CIoU + 0.5 x BCE + 1.5 x DFL"

**Training Process**

"Model kita sudah punya pengetahuan dasar dari training di COCO dataset (Common Objects in Context) -- 80 kelas objek umum, 330.000 gambar, 2.5 juta anotasi. Ini disebut PRETRAINED MODEL."

"Kita fine-tune (sesuaikan) pengetahuan itu ke dataset sampah kita. Ibaratnya, model sudah lulus SD (tahu bentuk dasar, tepi, tekstur). Sekarang kita latih dia untuk jadi AHLI SAMPAH."

Konfigurasi training:
- Epoch: 80 (80 kali lihat seluruh dataset)
- Batch: 16 (setiap langkah, model lihat 16 gambar sekaligus)
- Image size: 640x640 pixel
- Optimizer: SGD (Stochastic Gradient Descent) dengan MuSGD hybrid
- Learning rate: 0.001 (seberapa besar langkah perbaikan)
- Cosine annealing: LR menurun secara bertahap mengikuti kurva cosinus
- Momentum: 0.937 (mempercepat konvergensi)
- Weight decay: 0.0005 (regularisasi untuk mencegah overfitting)

---

## Slide 7: Hasil Pelatihan

### Narasi

"Setelah 80 epoch training (~2.5 jam), kita dapat hasil yang memuaskan."

**Training Progress**

"Selama training, kita pantau beberapa metrik:"

Loss (kesalahan) menurun:
- Box loss: dari ~2.1 ke ~0.78 (turun 63%)
- Cls loss: dari ~3.5 ke ~0.45 (turun 87%)
- DFL loss: dari ~1.8 ke ~0.92 (turun 49%)

"Loss menurun artinya model semakin pintar."

mAP (akurasi) meningkat:
- Dari ~10% di epoch 1 ke 80.4% di epoch 80

**Best Model Metrics**

"Ini hasil akhir pada test set (593 gambar yang tidak pernah dilihat model):"

**Box Metrics:**

| Metrik | Nilai | Arti |
|--------|-------|------|
| mAP@0.5 | 80.4% | Dengan IoU 0.5, akurasi 80.4% |
| mAP@0.5:0.95 | 52.5% | Rata-rata IoU 0.5-0.95, akurasi 52.5% |
| Precision | 76.7% | Dari 100 deteksi, ~77 benar |
| Recall | 75.6% | Dari 100 objek, ~76 terdeteksi |

"mAP@0.5 = 80.4% artinya: dengan batas IoU 0.5 (standar PASCAL VOC), model mencapai akurasi 80.4%. Kalau IoU ditingkatkan ke 0.95 (sangat ketat), akurasinya 52.5%."

"Precision: dari semua yang model deteksi sebagai 'sampah', 76.7% benar-benar sampah. Recall: dari semua sampah yang ada di gambar, 75.6% berhasil dideteksi."

**Mask Metrics:**

| Metrik | Nilai |
|--------|-------|
| mAP@0.5 | 49.7% |
| mAP@0.5:0.95 | 23.1% |
| Precision | 59.2% |
| Recall | 52.3% |

"Mask lebih rendah dari Box. Ini wajar karena: (1) Mask butuh prediksi boundary presisi, (2) Ground truth mask kita adalah pseudo-mask (buatan, tidak sempurna). Gap antara Box dan Mask ~30%."

**Per-Class Performance:**

"Ini yang menarik. Mari kita lihat per kelas:"

| Kelas | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Organik | 71.8% | 73.3% | 77.2% |
| Non-Organik | 81.6% | 77.9% | 83.6% |

"Non-Organik lebih baik. Kenapa? Dua alasan:"

1. **Data lebih banyak**: Non-Organik 3.289 vs Organik 684 (4.8x lipat). Model punya lebih banyak contoh untuk belajar.
2. **Bentuk lebih seragam**: Non-Organik seperti botol, kaleng, kaca -- bentuk rigid (tetap). Organik seperti sisa makanan, daun, ampas kopi -- bentuk amorf (tidak beraturan, selalu berubah).

"Analogi: lebih mudah mengenali botol Coca-Cola (bentuknya selalu sama) daripada mengenali tumpukan nasi (bentuknya bisa apa saja)."

**Inference Speed**

"Kecepatan model saat memproses 1 gambar:"
- Preprocess: 0.4ms (mempersiapkan gambar)
- Inference: 5.3ms (model memprediksi)
- Postprocess: 0.2ms (memformat output)
- Total: ~6ms per gambar (~166 FPS)

"166 FPS artinya: dalam 1 detik, model bisa memproses ~166 gambar. Ini REAL-TIME. Bahkan video 30 FPS tidak masalah."

---

## Slide 8: Lingkungan Eksperimen

### Narasi

"Untuk mencapai hasil ini, kita butuh 'dapur' yang mumpuni."

**Perangkat Keras (Hardware)**

"Komputer yang kita pakai:"

- **CPU: AMD Ryzen 7 8700F** -- 8 core, 16 thread. Ini 'kepala' yang mengatur semua operasi.
- **RAM: 32 GB DDR5** -- 2 stick 16 GB, kecepatan 5200 MT/s. Ini 'meja kerja' sementara.
- **GPU: NVIDIA GeForce RTX 5060 Ti** -- 16 GB VRAM, 4.608 CUDA cores. Ini 'otot' utama untuk deep learning.

"Kenapa GPU penting? GPU punya ribuan core kecil yang bisa mengerjakan banyak perhitungan matematika SECARA BERSAMAAN. Deep learning butuh perkalian matriks raksasa -- GPU sempurna untuk ini."

**Perangkat Lunak (Software)**

- **OS: Ubuntu 25.10** -- sistem operasi Linux
- **Python 3.12.9** -- bahasa pemrograman
- **PyTorch 2.12.1** -- framework deep learning (yang menjalankan model)
- **CUDA 13.0** + cuDNN -- driver dan library untuk mengakses GPU
- **Ultralytics 8.4.84** -- library YOLO (memudahkan training dan inference)
- **FastAPI** -- framework web untuk backend API
- **Nuxt.js 3 / Vue 3** -- framework frontend

**Kinerja Training**

"Waktu training total: ~2.5 jam untuk 80 epoch."

"Setiap epoch memproses 2.765 gambar (train set) dan mengevaluasi 593 gambar (val set). Dengan batch 16, setiap epoch butuh:"

- Training steps: 2.765 / 16 = ~173 langkah
- Validation steps: 593 / 16 = ~37 langkah
- Total per epoch: ~210 langkah

"80 epoch x 210 langkah = 16.800 langkah. Setiap langkah, model melihat 16 gambar, memprediksi, menghitung loss, dan memperbaiki bobotnya."

"VRAM peak: ~9.2 GB dari 16 GB total (57% utilisasi). Cukup longgar."

"Inference speed: 5.3 ms/gambar. Artinya: dalam 1 detik, ~188 gambar bisa diproses."

---

## Slide 9: Aplikasi Web

### Narasi

"Model sudah pintar. Tapi kalau cuma bisa dipakai lewat kode Python, siapa yang mau pakai? Kita buat aplikasi web supaya SEMUA ORANG bisa menggunakannya."

**Arsitektur: FastAPI + Nuxt.js**

"Dua komponen utama:"

1. **Backend (FastAPI)**: port 8000. Ini 'otak' di belakang layar. Menerima request, menjalankan model, mengembalikan hasil.
2. **Frontend (Nuxt.js 3)**: port 3000. Ini 'wajah' yang dilihat pengguna. Halaman web dengan tombol, gambar, dan teks.

**Flow Deteksi Lengkap:**

"Begini cara kerja deteksi dari awal sampai akhir:"

1. **Buka browser**: user buka `http://localhost:3000/dashboard`
2. **Upload foto**: user drag-drop foto sampah atau klik untuk browse
3. **Klik Deteksi**: frontend mengirim POST request ke `http://localhost:8000/api/detect`
4. **Backend terima**: FastAPI validasi file (tipe, ukuran)
5. **Simpan sementara**: file disimpan di folder `uploads/`
6. **Jalankan model**: YOLO memproses gambar di GPU (5.3ms)
7. **Hasil deteksi**: model mengembalikan daftar objek terdeteksi (kelas, confidence, bbox)
8. **Gambar annotasi**: bounding box digambar di foto + label + confidence
9. **Simpan hasil**: gambar annotated disimpan di `static/result/`
10. **Hapus upload**: file asli dihapus (privasi)
11. **Kirim response**: backend mengembalikan JSON + URL gambar annotated
12. **Tampilkan hasil**: frontend menampilkan gambar annotated + summary cards!

"Total waktu: upload + infer + annotasi + response = < 1 detik."

**Response JSON Contoh:**
```json
{
  "success": true,
  "file_type": "image",
  "filename": "sampah.jpg",
  "detected_objects": [
    {
      "label": "Non-Organik",
      "category": "Non-Organik",
      "confidence": 0.9234,
      "bbox": [120.5, 80.3, 340.2, 450.1]
    }
  ],
  "summary": {
    "organik": 0,
    "non_organik": 1,
    "total": 1
  },
  "result_url": "/static/result/20260706_113253_sample_annotated.jpg",
  "recommendation": "Buang ke Tempat Sampah Non-Organik"
}
```

**Frontend Pages:**

- `/dashboard`: Halaman utama. Upload + deteksi + results + Run Full Pipeline
- `/test`: Upload banyak file sekaligus (batch). Bisa preview gambar sebelum upload
- `/train-eval`: Laporan metrik training (graphs, curves)
- `/val-result`: Hasil validasi
- `/test-result`: Hasil test set
- `/inference-export`: Batch inference + export model

**Sidebar:**
- Main: Dashboard
- Report: Training Eval, Validation, Test Results, Inference

**Recycling Advice:**
| Deteksi | Saran |
|---------|-------|
| Organik | Buang ke Tempat Sampah Organik |
| Non-Organik | Buang ke Tempat Sampah Non-Organik |

**API Endpoints:**

| Endpoint | Method | Fungsi |
|----------|--------|--------|
| /api/detect | POST | Deteksi 1 file |
| /api/detect/bulk | POST | Deteksi banyak file |
| /api/kaggle/pipeline/run-full | POST | Jalankan full training pipeline |
| /health | GET | Cek status server + GPU |

**Model Deployment:**

"Setelah training selesai, model terbaik otomatis disalin dari `runs/segment/full_pipeline/weights/best.pt` ke `backend/models/best.pt`."

"Ukuran model: 54.5 MB. Cukup kecil untuk di-deploy di berbagai platform."

"Saat API pertama kali diakses, model dimuat ke GPU (lazy loading). Inference berikutnya langsung cepat karena model sudah di memori."

---

## Slide 10: Kesimpulan

### Narasi

"Mari kita simpulkan perjalanan kita hari ini."

**1. Dual Dataset Integration -- Berhasil**

"Kita berhasil menggabungkan 2 sumber data yang berbeda: TACO (dataset publik dengan anotasi COCO) dan Waste Classification Dataset (dataset lokal kita)."

"Total: 3.973 gambar (684 Organik + 3.289 Non-Organik)."

"Ini lebih banyak dari dataset asli mana pun. Menggabungkan sumber data adalah strategi efektif untuk meningkatkan jumlah dan variasi data."

**2. YOLOv26m-seg -- Mencapai 80.4% Box mAP**

"Model kita (23.5M parameter, 121.2 GFLOPs) mencapai Box mAP@0.5 = 80.4% dan Mask mAP@0.5 = 49.7%."

"Ini hasil yang SANGAT BAIK untuk dataset dengan pseudo-mask (bukan anotasi manual). Model bisa mendeteksi dan mengklasifikasikan sampah dengan akurasi tinggi."

**3. Non-Organik Mendominasi -- Tapi Itu Wajar**

"Non-Organik (83.6%) lebih baik dari Organik (77.2%). Dua penyebab: (1) Data Non-Organik 4.8x lebih banyak, (2) Bentuk Non-Organik lebih seragam."

"Untuk meningkatkan Organik: kumpulkan lebih banyak data, atau beri bobot lebih pada kelas Organik saat training."

**4. Gap Box vs Mask -- Pekerjaan Rumah**

"Mask mAP (49.7%) lebih rendah dari Box mAP (80.4%). Gap ~30%."

"Ini karena kualitas pseudo-mask yang terbatas. Edge detection (66.4%) cukup baik, fallback geometris (33.6%) kurang akurat."

"Solusi: gunakan model segmentasi yang lebih baik untuk generate mask (misalnya SAM -- Segment Anything Model), atau anotasi manual untuk sampel kunci."

**5. Aplikasi Web -- Siap Pakai**

"Seluruh sistem sudah bisa diakses melalui web browser. Upload gambar -> deteksi otomatis -> lihat hasil + rekomendasi."

"Pipeline training juga terintegrasi: satu klik 'Run Full Pipeline', data disiapkan, model dilatih, dan model siap pakai."

**Saran untuk Pengembangan ke Depan:**

*Data:*
- Kumpulkan 2.000+ gambar Organik tambahan untuk menyeimbangkan dataset
- Anotasi mask manual pada 500 gambar kunci untuk meningkatkan kualitas ground truth
- Gunakan Segment Anything Model (SAM) untuk generate mask yang lebih akurat

*Model:*
- Class-weighted loss: beri bobot lebih pada kelas Organik
- Focal loss: gantikan BCE untuk fokus ke contoh sulit
- Coba YOLO26l atau YOLO26x untuk akurasi lebih tinggi (butuh GPU lebih besar)
- Test-Time Augmentation (TTA) untuk inference yang lebih akurat

*Aplikasi:*
- Real-time webcam detection via WebSocket
- Deployment ke cloud (AWS, GCP, atau Azure)
- Mobile app untuk deteksi via smartphone

---

## Slide 11: Q&A

### Narasi

"Terima kasih atas perhatiannya. Sekarang kita buka sesi tanya jawab."

"Sebelum bertanya, berikut ringkasan cepat:"
- Dataset: 3.973 gambar dari 2 sumber (TACO + lokal)
- Model: YOLOv26m-seg, Box mAP@0.5 = 80.4%
- Aplikasi: Web dashboard untuk deteksi langsung

### Q&A Potensial dengan Jawaban Sederhana

**Q: Kenapa hanya 2 kelas? Kenapa tidak langsung 60 kelas seperti TACO?**

A: "Bayangkan kamu harus mengajari adik kelas 5 SD untuk membedakan 60 jenis sampah. Pasti pusing, kan? Dengan 2 kelas, SEMUA ORANG bisa memverifikasi hasil deteksi. 'Ini organik atau bukan?' -- mudah. Non-Organik nanti dipetakan secara otomatis di backend."

**Q: Model kita akurasinya 80%. Apakah itu sudah cukup bagus?**

A: "80% sudah sangat baik untuk dataset pertama dengan pseudo-mask. Sebagai perbandingan, model YOLO standar di COCO dataset (80 kelas, 330.000 gambar) mencapai ~50% mAP@0.5:0.95. Kita dapat 80.4% mAP@0.5 untuk 2 kelas. Tentu kita ingin lebih tinggi. Target kita berikutnya 85-90%."

**Q: Kenapa Mask mAP lebih rendah dari Box mAP?**

A: "Bayangkan kamu harus menggambar bentuk daun dengan 24 titik penghubung. Sulit, kan? Apalagi kalau kamu hanya melihat garis tepi yang kabur. Mask mAP lebih rendah karena: (1) Mask butuh presisi boundary (tepian), (2) Ground truth mask kita buatan sendiri (tidak sempurna). Box mAP hanya butuh kotak pembatas -- lebih mudah."

**Q: Class imbalance (Organik sedikit) -- bagaimana dampaknya?**

A: "Dampaknya ada, tapi tidak terlalu besar. Organik (77.2%) vs Non-Organik (83.6%) -- gap ~6%. Solusi terbaik: kumpulkan lebih banyak data Organik. Solusi teknis: beri bobot lebih pada kelas Organik saat training (class-weighted loss)."

**Q: Model bisa membedakan Anorganik vs Residu?**

A: "Saat ini model hanya 2 kelas: Organik vs Non-Organik. Tapi di backend, kita punya mapping subkategori. Ketika model mendeteksi 'Non-Organik', backend bisa cek subkategorinya: kalau e-waste/cans/glass -> Anorganik (recyclable). Kalau diapers/styrofoam -> Residu (landfill). Jadi pemetaan Anorganik/Residu terjadi di backend, bukan di model."

**Q: Berapa lama training? Berapa biaya listriknya?**

A: "Training 80 epoch selesai dalam ~2.5 jam di RTX 5060 Ti 16 GB. Konsumsi daya GPU ~200 watt. 2.5 jam x 0.2 kW = 0.5 kWh. Dengan tarif listrik ~Rp1.500/kWh, total biaya listrik ~Rp750 per training. Murah!"

**Q: Apakah model bisa jalan di HP (handphone)?**

A: "Model kita 54.5 MB dan butuh GPU untuk inferensi cepat. HP flagship terbaru punya NPU (Neural Processing Unit) yang bisa menjalankan model kecil. Tapi YOLO26m-seg (23.5M parameter) mungkin terlalu besar. Kita perlu konversi ke format TFLite atau ONNX, dan mungkin pakai varian nano (2.7M parameter) untuk HP. Ini bisa jadi pengembangan ke depan."

**Q: Dataset kita apakah sudah cukup representatif untuk Bali?**

A: "TACO diambil dari berbagai negara. Waste Classification Dataset kita tidak disebutkan lokasi pengambilannya. Idealnya, kita punya dataset spesifik Bali: sampah di pantai, pasar tradisional, rumah makan, pura. Ini bisa jadi penelitian lanjutan."

**Q: Apakah model kita overfit?**

A: "Overfit = model hafal data training tapi tidak bisa generalisasi ke data baru. Indikator: val/test mAP jauh lebih rendah dari train mAP. Kita cek: train mAP ~82%, test mAP ~80.4%. Gap <2%. Artinya TIDAK ADA OVERFIT. Model kita generalis dengan baik."

**Q: Satu lagi, apakah model bisa deteksi video real-time?**

A: "Bisa! Dengan 5.3ms per frame, model kita bisa handle ~188 FPS. Video standar 30 FPS tidak masalah. Bahkan video slow-motion 120 FPS pun masih bisa. Tapi perlu streaming (WebSocket) untuk real-time. Saat ini API kita masih request-response (upload dulu, baru deteksi). Fitur real-time ada di roadmap."

---

## Penutup untuk Presenter

"Teman-teman, presentasi ini adalah perjalanan dari masalah nyata (sampah di Bali) ke solusi teknologi (AI deteksi sampah)."

"Kita sudah belajar:"
- Bagaimana AI 'melihat' dan 'memahami' gambar
- Bagaimana dataset dibuat dan diproses
- Bagaimana model dilatih dan dievaluasi
- Bagaimana hasilnya diakses melalui web

"Yang paling penting: teknologi ini bisa membantu menyelesaikan masalah LINGKUNGAN yang nyata. Pemilahan sampah yang lebih baik = lebih banyak sampah didaur ulang = lebih sedikit sampah ke TPA = lingkungan yang lebih bersih."

"Terima kasih. Selamat bertanya!"
