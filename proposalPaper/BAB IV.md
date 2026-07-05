# BAB IV: HASIL DAN PEMBAHASAN

## 4.1 Hasil Pipeline Data

### 4.1.1 Dataset Statistics

Dataset phenomsg/waste-classification: ~2.917 citra, 18 subkategori, 4 kategori utama.

Distribusi per subkategori:

| Subkategori | Images | Main Category |
|-------------|--------|---------------|
| e-waste | 538 | Hazardous |
| cans_all_type | 272 | Recyclable |
| coffee_tea_bags | 157 | Organic |
| paints | 153 | Hazardous |
| food_scraps | 147 | Organic |
| diapers | 145 | Non-Recyclable |
| glass_containers | 140 | Recyclable |
| pesticides | 138 | Hazardous |
| ceramic_product | 138 | Non-Recyclable |
| platics_bags_wrappers | 134 | Non-Recyclable |
| yard_trimmings | 131 | Organic |
| plastic_bottles | 126 | Recyclable |
| egg_shells | 125 | Organic |
| paper_products | 121 | Recyclable |
| stroform_product | 118 | Non-Recyclable |
| kitchen_waste | 114 | Organic |
| batteries | 110 | Hazardous |
| sanitary_napkin | 110 | Non-Recyclable |

```mermaid
xychart-beta
    title "Distribusi Citra per Subkategori"
    x-axis ["e-waste", "cans_all", "coffee_tea", "paints", "food_scraps", "diapers", "glass", "pesticides", "ceramic", "plastics_bags", "yard_trimmings", "plastic_bottles", "egg_shells", "paper", "stroform", "kitchen_waste", "batteries", "sanitary_napkin"]
    y-axis "Images" 0 --> 600
    bar [538, 272, 157, 153, 147, 145, 140, 138, 138, 134, 131, 126, 125, 121, 118, 114, 110, 110]
```

### 4.1.2 Pseudo-Mask Generation

| Metode | Jumlah | Persentase |
|--------|--------|------------|
| Edge detection | 2.429 | 83.3% |
| Fallback geometris | 488 | 16.7% |

Edge detection berhasil pada 83.3% citra. Fallback terjadi pada citra dengan foreground/background kontras rendah.

```mermaid
pie title Metode Pseudo-Mask Generation
    "Edge Detection Otsu" : 83.3
    "Fallback Geometris" : 16.7
```

### 4.1.3 Split Distribution

| Split | Images |
|-------|--------|
| Train | 2.041 |
| Val | 438 |
| Test | 438 |

## 4.2 Hasil Pelatihan Model

### 4.2.1 Training Progress

Training: 95 epoch (early stopped at 75 best, patience=20). Total ~4 jam pada Tesla T4.

Best model pada epoch 75:

| Metrik | Box | Mask |
|--------|-----|------|
| mAP@0.5 | 48.5% | 35.7% |
| mAP@0.5:0.95 | 33.7% | 18.0% |
| Precision | 53.0% | 39.8% |
| Recall | 48.1% | 39.2% |

```mermaid
xychart-beta
    title "Box vs Mask Metrics (Best Epoch 75)"
    x-axis ["mAP@0.5", "mAP@0.5:0.95", "Precision", "Recall"]
    y-axis "%" 0 --> 60
    bar [48.5, 33.7, 53.0, 48.1]
    bar [35.7, 18.0, 39.8, 39.2]
```

### 4.2.2 Per-Class Mask AP@50 (Test Set)

| Subkategori | Mask AP@50 | Category |
|-------------|------------|----------|
| e-waste | 78.6% | Hazardous |
| platics_bags_wrappers | 75.2% | Non-Recyclable |
| cans_all_type | 60.0% | Recyclable |
| coffee_tea_bags | 62.3% | Organic |
| batteries | 53.8% | Hazardous |
| paints | 45.8% | Hazardous |
| sanitary_napkin | 42.3% | Non-Recyclable |
| stroform_product | 40.6% | Non-Recyclable |
| glass_containers | 39.8% | Recyclable |
| paper_products | 39.7% | Recyclable |
| egg_shells | 37.7% | Organic |
| plastic_bottles | 35.7% | Recyclable |
| food_scraps | 33.3% | Organic |
| yard_trimmings | 21.4% | Organic |
| pesticides | 18.6% | Hazardous |
| diapers | 17.0% | Non-Recyclable |
| ceramic_product | 16.5% | Non-Recyclable |
| kitchen_waste | 11.2% | Organic |

```mermaid
flowchart TD
    subgraph High[High Performance >50%]
        H1[e-waste: 78.6%]
        H2[plastics_bags: 75.2%]
        H3[cans_all: 60.0%]
        H4[coffee_tea: 62.3%]
        H5[batteries: 53.8%]
    end
    
    subgraph Medium[Medium Performance 30-50%]
        M1[paints: 45.8%]
        M2[sanitary_napkin: 42.3%]
        M3[stroform: 40.6%]
        M4[glass: 39.8%]
        M5[paper: 39.7%]
        M6[egg_shells: 37.7%]
        M7[plastic_bottles: 35.7%]
        M8[food_scraps: 33.3%]
    end
    
    subgraph Low[Low Performance <30%]
        L1[yard_trimmings: 21.4%]
        L2[pesticides: 18.6%]
        L3[diapers: 17.0%]
        L4[ceramic: 16.5%]
        L5[kitchen_waste: 11.2%]
    end
    
```

## 4.3 Pembahasan

### 4.3.1 Kinerja per Kategori

Kategori Hazardous (e-waste 78.6%, batteries 53.8%) dan Recyclable (cans 60.0%) menunjukkan performa terbaik karena bentuk objek yang relatif seragam.

Kategori Organic (kitchen_waste 11.2%, food_scraps 33.3%) menunjukkan performa terendah karena variasi bentuk dan tekstur yang tinggi.

### 4.3.2 Pseudo-Mask Quality

Edge detection (83.3%) menghasilkan mask yang cukup baik untuk objek dengan kontras foreground/background jelas. Fallback geometris (16.7%) kurang akurat, terutama pada kitchen_waste dan food_scraps.

### 4.3.3 Class Imbalance

Subkategori dengan jumlah citra sedikit (kitchen_waste 114, batteries 110, sanitary_napkin 110) cenderung memiliki mask AP lebih rendah. e-waste (538 citra) memiliki performa terbaik.

```mermaid
flowchart LR
    subgraph Factors[Faktor yang Mempengaruhi Performa]
        F1[Jumlah Citra per Kelas]
        F2[Kontras Foreground/Background]
        F3[Variasi Bentuk & Tekstur]
        F4[Class Imbalance]
    end
    
    F1 --> E[e-waste: 538 images<br/>AP 78.6%]
    F4 --> L[kitchen_waste: 114 images<br/>AP 11.2%]
    F2 --> EDGE[Edge Detection 83.3%]
    F3 --> ORGANIC[Organic: Variasi Tinggi<br/>AP Rendah]
    
```

### 4.3.4 Referensi Notebook

Pipeline dibagi dalam 4 kelompok di aplikasi web: `/raw/dataset` (Load + Profiling), `/raw/preparation` (Convert + Visualize), `/raw/training` (Train + Results + Evaluate), dan `/raw/deployment` (Inference + Batch + Export + Verify). Dokumentasi teknis tersedia di `walkthrought.md`.
