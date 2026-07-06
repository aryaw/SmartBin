## Slide 1: Judul

**Deteksi dan Klasifikasi Sampah Menggunakan YOLOv26m-seg untuk Instance Segmentation (2 Kelas: Organik/Non-Organik)**

```mermaid
flowchart TD
    classDef default fill:none,stroke:#333,stroke-width:1
    subgraph S1[Source 1: TACO Dataset]
        A1[annotations.json<br/>COCO Format] --> A2[1.500 Images<br/>60 Categories]
        A2 --> A3[Segmentation Polygons<br/>+ Bounding Boxes]
        A3 --> A4[Category Mapping<br/>Food waste → Organik<br/>All other 59 → Non-Organik]
    end
    
    subgraph S2[Source 2: Waste Classification Dataset]
        B1[18 Subcategory Folders<br/>backend/dataset/raw/] --> B2[2.939 Images<br/>JPG/PNG]
        B2 --> B3[Organik Subfolders<br/>coffee_tea_bags, egg_shells<br/>food_scraps, kitchen_waste<br/>yard_trimmings]
        B2 --> B4[Non-Organik Subfolders<br/>e-waste, cans, glass, paper<br/>plastic, batteries, paints<br/>pesticides, ceramic, diapers<br/>plastics_bags, sanitary<br/>styrofoam]
    end
    
    subgraph MG[Merge & Classify]
        C1[Map: Organik = 0<br/>Non-Organik = 1] --> C2[Copy to Flat Folders]
        C2 --> C3[backend/dataset/raw/Organik/<br/>684 Images]
        C2 --> C4[backend/dataset/raw/Non-Organik/<br/>3.289 Images]
        C3 --> C5[Total: 3.973 Images]
        C4 --> C5
    end
    
    A4 --> C1
    B3 --> C1
    B4 --> C1
    
    subgraph PP[Pseudo-Polygon Mask Generation]
        D1[Read Image RGB] --> D2[Convert to Grayscale]
        D2 --> D3[Gaussian Blur 5x5]
        D3 --> D4[Otsu Thresholding]
        D4 --> D5{Mean > 127?}
        D5 -->|Yes| D6[Invert Binary Mask]
        D5 -->|No| D7[Morphological Close 5x5<br/>2 iterations]
        D6 --> D7
        D7 --> D8[Find Contours<br/>Take Largest]
        D8 --> D9{Area >= 20%?}
        D9 -->|Yes 66.4%| D10[Approx Polygon<br/>ε = 0.01 x arcLength]
        D9 -->|No 33.6%| D11[Fallback 60% Ellipse<br/>40% Rounded Rect]
        D10 --> D12[Normalize to [0,1]<br/>24 Points]
        D11 --> D12
        D12 --> D13[YOLO-seg Label<br/>class_id x1 y1 ... xn yn]
    end
    
    C5 --> D1
    
    subgraph SS[Stratified Split 70/15/15]
        E1[Total: 3.973] --> E2[Shuffle + Stratify<br/>by Class]
        E2 --> E3[Train: 2.765]
        E2 --> E4[Val: 593]
        E2 --> E5[Test: 593]
    end
    
    D13 --> E1
    
    subgraph AUG[Online Augmentation]
        F1[Mosaic 1.0<br/>4 images combined] --> F2[Mixup 0.2<br/>Image Blending]
        F2 --> F3[Copy-Paste 0.15<br/>Object Copy]
        F3 --> F4[HSV Jitter<br/>H=0.05 S=0.8 V=0.5]
        F4 --> F5[Geometric<br/>Rotate/Scale/Shear]
        F5 --> F6[Flip LR 50%]
    end

    subgraph BACK[Backbone: CSPDarknet]
        G1[Input 640x640x3] --> G2[Stem Conv k7 s2<br/>320x320]
        G2 --> G3[Stage 1 CSP<br/>160x160 C=128]
        G3 --> G4[Stage 2 CSP<br/>80x80 C=256]
        G4 --> G5[Stage 3 CSP<br/>40x40 C=512]
        G5 --> G6[Stage 4 CSP<br/>20x20 C=512]
        G6 --> G7[SPP Layer<br/>k5, k9, k13 Pool]
    end
    
    subgraph NECK[Neck: FPN + PAN]
        H1[P5 20x20] --> H2[Upsample 2x]
        H2 --> H3[Concat with P4<br/>40x40]
        H3 --> H4[Conv 1x1 Reduce]
        H4 --> H5[Upsample 2x]
        H5 --> H6[Concat with P3<br/>80x80]
        H6 --> H7[Conv 1x1 Reduce]
        H7 --> H8[Downsample Conv k3 s2]
        H8 --> H9[Concat with P4<br/>40x40]
        H9 --> H10[Conv 1x1 Reduce]
        H10 --> H11[Downsample Conv k3 s2]
        H11 --> H12[Concat with P5<br/>20x20]
    end
    
    subgraph HEAD[Decoupled Head <br/>Anchor-Free]
        I1[P3 80x80] --> I2[Classification<br/>2x Conv3x3 + Linear<br/>→ 2 classes + obj]
        I1 --> I3[Regression<br/>DFL 16-bin distrib<br/>→ x, y, w, h]
        I1 --> I4[Segmentation<br/>Proto Module<br/>32 prototypes<br/>→ 24-pt polygon]
        
        I5[P4 40x40] --> I2
        I5 --> I3
        I5 --> I4
        
        I6[P5 20x20] --> I2
        I6 --> I3
        I6 --> I4
    end
    
    E3 --> AUG
    AUG --> G1
    
    subgraph LOSS[Loss Functions]
        J1[CIoU Loss<br/>weight 7.5<br/>IoU + Center Dist<br/>+ Aspect Ratio] --> J4[Total Loss]
        J2[BCE Loss<br/>weight 0.5<br/>Binary Classification] --> J4
        J3[DFL Loss<br/>weight 1.5<br/>Position Distribution] --> J4
        J4 --> J5[SGD Optimizer<br/>MuSGD Hybrid<br/>LR=0.001 Cosine<br/>Momentum=0.937<br/>Weight Decay=0.0005]
    end
    
    I2 --> J2
    I3 --> J1
    I3 --> J3
    J5 --> K1[80 Epochs<br/>Patience=40<br/>Batch=16<br/>FP16 Mixed Precision]
    
    K1 --> L1[Best Model<br/>Epoch Selection]
    
    subgraph EVAL[Evaluation Metrics]
        M1[Box mAP@0.5: 80.4%] --> M4[Per-Class Analysis]
        M2[Box mAP@0.5:0.95: 52.5%] --> M4
        M3[Mask mAP@0.5: 49.7%] --> M4
        M4 --> M5[Organik Box mAP: 77.2%<br/>Non-Organik Box mAP: 83.6%]
        M4 --> M6[Organik Mask mAP: 38.2%<br/>Non-Organik Mask mAP: 61.2%]
        M5 --> M7[Precision: 76.7%<br/>Recall: 75.6%<br/>F1-Score: 76.1%]
    end
    
    L1 --> M1
    L1 --> M2
    L1 --> M3
    
    subgraph API[Backend API FastAPI Port 8000]
        N1[GET /health<br/>GPU Status] --> N2[POST /api/detect<br/>Image/Video Upload]
        N2 --> N3[YOLO Inference<br/>5.3ms per image]
        N3 --> N4[Detected Objects<br/>Class + Confidence + BBox]
        N4 --> N5[Annotated Image<br/>Saved to static/result/]
        N5 --> N6[Response JSON<br/>DetectedObjects + Summary<br/>+ Recommendation]
        N2 --> N7[POST /api/detect/bulk<br/>Multi-file Batch]
        N8[POST /api/kaggle/pipeline/run-full<br/>End-to-end Training Pipeline] --> N9[prepare_from_local<br/>→ Train YOLO<br/>→ Validate<br/>→ Copy best.pt]
    end
    
    M5 --> N2
    
    subgraph FE[Frontend Nuxt.js 3 Port 3000]
        O1[/dashboard<br/>Upload & Detect] --> O2[FileUpload Component<br/>Drag & Drop]
        O2 --> O3[POST /api/detect]
        O3 --> O4[DetectionResult<br/>Annotated Image<br/>Organik/Non-Organik Counts<br/>Recommendation]
        O1 --> O5[Run Full Pipeline Button<br/>POST /api/kaggle/pipeline/run-full]
        O6[/test<br/>Multi-file Upload<br/>Preview Before Upload]
        O7[/train-eval<br/>Training Metrics Report]
        O8[/val-result<br/>Validation Results]
        O9[/test-result<br/>Test Set Results]
        O10[/inference-export<br/>Batch Inference & Export]
    end
    
    N6 --> O4
    N7 --> O6
    
    subgraph DEPLOY[Model Deployment]
        P1[runs/segment/full_pipeline/weights/best.pt<br/>54.5 MB] --> P2[Copy to<br/>backend/models/best.pt]
        P2 --> P3[Detector loads model<br/>on first inference request]
        P3 --> P4[GPU: RTX 5060 Ti 16GB<br/>CUDA 13.0<br/>Inference: 5.3ms/image]
    end
    
    K1 --> P1
    

---

## Slide 2: Outline Presentasi

1. **Latar Belakang** - Krisis sampah global & Indonesia
2. **Rumusan Masalah & Tujuan**
3. **Dataset** - Dua sumber: TACO + Waste Classification
4. **Metode** - Pseudo-mask, Pipeline, Arsitektur
5. **Hasil** - 80.4% Box mAP@0.5, Per-class analysis
6. **Aplikasi Web** - Dashboard deteksi langsung
7. **Kesimpulan & Saran**

---

## Slide 3: Latar Belakang

- Bali menghasilkan **~1.340 ton sampah per hari** (DLHK Bali)
- Pariwisata menyumbang **60% dari total sampah**
- Komposisi: **60% organik, 30% plastik, 10% lainnya**
- Pemilahan manual masih dominan - tidak efisien

Pergub No.47/2019 menetapkan 3 kategori: **Organik**, **Anorganik**, **Residu**.

Pendekatan 2 kelas (Organik/Non-Organik) dipilih karena tidak memerlukan keahlian khusus untuk validasi.

---

## Slide 4: Dataset - Dua Sumber

**Source 1: TACO (Trash Annotations in Context)**
- 1.500 citra dengan anotasi COCO (segmentation polygon, bbox)
- 60 subkategori: Aluminium foil, Battery, Plastic bottle, Food waste, dll
- Dipetakan ke 2 kelas via ORGANIC_CATEGORIES config

**Source 2: Waste Classification Dataset**
- 2.939 citra, 18 subkategori
- Organik: coffee_tea_bags, egg_shells, food_scraps, kitchen_waste, yard_trimmings
- Non-Organik: e-waste, cans, glass, paper, plastic, batteries, dll

| Kelas | Jumlah |
|-------|--------|
| Organik | 684 |
| Non-Organik | 3.289 |
| **Total** | **3.973** |

---

## Slide 5: Split & Pseudo-Mask

**Stratified Split 70/15/15:**

| Split | Images |
|-------|--------|
| Train | 2.765 |
| Val | 593 |
| Test | 593 |

**Pseudo-Polygon Mask Generation:**
- Edge detection Otsu: 66.4%
- Fallback geometris (ellipse/rounded rect): 33.6%

Format YOLO-seg: `<class_id> x1 y1 x2 y2 ... xn yn`

---

## Slide 6: Arsitektur YOLOv26m-seg

| Komponen | Detail |
|----------|--------|
| Model | YOLOv26m-seg (23.5M params) |
| Input | 640x640 |
| Backbone | CSPDarknet (4 stage) |
| Neck | FPN + PAN |
| Head | Decoupled: Classification + Regression + Segmentation |
| Optimizer | SGD (MuSGD hybrid) |
| Pretrained | COCO fine-tuned waste |

**Hyperparameter:**
| Parameter | Nilai |
|-----------|-------|
| Epochs | 80 (patience=40) |
| Batch | 16 |
| Image size | 640 |
| LR | 0.001 (cosine) |

---

## Slide 7: Hasil Pelatihan

**Best Model Metrics (80 epoch):**

| Metrik | Box | Mask |
|--------|-----|------|
| **mAP@0.5** | **80.4%** | **49.7%** |
| **mAP@0.5:0.95** | **52.5%** | **23.1%** |
| Precision | 76.7% | 59.2% |
| Recall | 75.6% | 52.3% |

**Per-Class Box mAP@0.5:**
| Kelas | Precision | Recall | mAP@0.5 |
|-------|-----------|--------|---------|
| Organik | 71.8% | 73.3% | 77.2% |
| Non-Organik | 81.6% | 77.9% | 83.6% |

Training time: ~2.5 jam pada RTX 5060 Ti 16 GB.

---

## Slide 8: Lingkungan Eksperimen

| Komponen | Spesifikasi |
|----------|-------------|
| CPU | AMD Ryzen 7 8700F (8 core) |
| RAM | 32 GB DDR5 |
| GPU | NVIDIA RTX 5060 Ti (16 GB VRAM) |
| CUDA | 13.0, Driver 595.71 |
| Framework | Ultralytics 8.4.84, PyTorch 2.12.1 |
| Backend | FastAPI, Uvicorn |
| Frontend | Nuxt.js 3 (Vue 3) |
| Inference speed | 5.3 ms/image |
| Model size | 54.5 MB |

---

## Slide 9: Aplikasi Web

**Arsitektur:** FastAPI port 8000 + Nuxt.js 3 port 3000

**Flow Deteksi:**
1. User upload image via /dashboard
2. Backend infer dengan YOLO model
3. Output: annotated image + counts Organik/Non-Organik
4. Recommendation: Buang ke Tempat Sampah Organik/Non-Organik

**Sidebar:**
- Main: Dashboard
- Report: Training Eval, Validation, Test Results, Inference

**Recycling Advice:**
| Deteksi | Saran |
|---------|-------|
| Organik | Buang ke Tempat Sampah Organik |
| Non-Organik | Buang ke Tempat Sampah Non-Organik |

---

## Slide 10: Kesimpulan

1. **Dual dataset integration** berhasil: 3.973 images (684 Organik + 3.289 Non-Organik)
2. **YOLOv26m-seg** mencapai Box mAP@0.5 **80.4%**, Mask mAP@0.5 **49.7%**
3. **Performa per kelas:** Non-Organik (83.6%) > Organik (77.2%) karena data 4.8x lebih banyak
4. **Class imbalance** masih jadi tantangan untuk kelas Organik
5. **Aplikasi web** fungsional: upload ke deteksi ke rekomendasi

**Saran:**
- Kumpulkan lebih banyak data Organik
- Implementasi class-weighted loss
- Coba YOLO26l/x untuk akurasi lebih tinggi
- Real-time webcam detection

---

## Slide 11: Q&A

**Terima Kasih**

- Dataset: 3.973 images, 2 sumber (TACO + local)
- Model: YOLOv26m-seg, Box mAP@0.5 80.4%
- Aplikasi: Web dashboard deteksi langsung
