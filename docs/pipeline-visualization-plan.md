# Pipeline Visualization Export Plan

## Goal
Generate step-by-step visualization images for 4 pipeline processes to use in presentations and documentation. Output: `waste_datasource/visualization/`.

---

## Process 1: Pseudo-Polygon Mask Generation

Sample image: `waste_datasource/inferencedata/batch_1_000000.jpg`
Output: `waste_datasource/visualization/pseudo_mask/step_XX_description.jpg`

### Step-by-Step Images

| # | Step | Description | Visual Output | Type |
|---|------|-------------|---------------|------|
| 1 | Read Image RGB | Original image loaded from disk | Full color image, 640x640 | Image |
| 2 | Convert to Grayscale | RGB -> single channel luminance | Grayscale image | Image |
| 3 | Gaussian Blur 5x5 | Smooth noise, kernel size 5 | Slightly blurred grayscale | Image |
| 4 | Otsu Thresholding | Auto-calculate threshold, binarize | Binary (black background, white object) | Image |
| 5 | Mean > 127? | Check if average pixel > 127 | Decision diamond with "Yes/No" branches + mean value text | Diagram |
| 6A | Invert Binary Mask | If mean>127: flip black<->white | Inverted binary (white bg, black object) | Image |
| 6B | Morphological Close 5x5 | Close holes, kernel 5x5, 2 iterations | Binary with filled holes | Image |
| 7 | Find Contours | cv2.findContours, take largest | Original image with GREEN contour outline overlay | Image+Overlay |
| 8 | Area >= 20%? | Check if contour area >= 20% image | Decision diamond with "Contour area: XX% of total" text | Diagram |
| 9A | Edge Success | If >=20%: use edge detection result | Original image with contour filled (semi-transparent green) | Image+Overlay |
| 9B | Fallback: Ellipse (60%) | If <20%: generate elliptical polygon | Original image with ellipse overlay (blue, dashed) | Image+Overlay |
| 9C | Fallback: Rounded Rect (40%) | If <20%: generate rounded rectangle | Original image with rounded rect overlay (orange, dashed) | Image+Overlay |
| 10 | Approx Polygon | cv2.approxPolyDP, epsilon=0.01*arcLength | Original with RED polygon vertices + edges, vertex count text | Image+Overlay |
| 11 | Normalize to [0,1] | Divide coordinates by width/height | Polygon with normalized coordinates displayed | Text+Overlay |
| 12 | YOLO-seg Label | Final label format | Image with label text: "0 0.123 0.456 0.789..." | Text+Image |

### Grid Summary (Collage)

Create a 4x4 grid collage of all 16 steps for quick reference.

---

## Process 2: Stratified Split 70/15/15

Output: `waste_datasource/visualization/stratified_split/`

### Visualizations

| # | Image | Description | Type |
|---|-------|-------------|------|
| 1 | split_bar_chart.jpg | Horizontal bar chart: 3 bars (Train=70%, Val=15%, Test=15%) with counts (2,765 / 593 / 593), color-coded blue/green/orange | Chart |
| 2 | split_pie_chart.jpg | Pie chart: 3 segments, 70/15/15, with image count labels | Chart |
| 3 | class_distribution.jpg | Grouped bar chart: per-split class distribution (Organik vs Non-Organik in each split). Shows stratified balance | Chart |
| 4 | split_table.jpg | Table showing: Split | Total | Organik | Non-Organik | Percentage. Train: 2765 / 476 / 2289 / 70%. Val: 593 / 102 / 491 / 15%. Test: 593 / 102 / 491 / 15% | Table |

### Data Table

| Split | Total | Organik | Non-Organik | % of Total |
|-------|-------|---------|-------------|------------|
| Train | 2,765 | ~476 | ~2,289 | 70% |
| Val | 593 | ~102 | ~491 | 15% |
| Test | 593 | ~102 | ~491 | 15% |

---

## Process 3: Online Augmentation

Sample image: `waste_datasource/inferencedata/batch_1_000000.jpg`
Sample image 2-4: Additional images from inferencedata/
Output: `waste_datasource/visualization/augmentation/`

### Step-by-Step Images

| # | Step | Description | Visual Output |
|---|------|-------------|---------------|
| 1 | Original Image | Base image before any augmentation | Full color image |
| 2 | Mosaic 1.0 | 4 images combined into 1 grid (2x2) with bounding lines | 2x2 grid of 4 different waste images |
| 3 | Mixup 0.2 | 2 images blended with alpha=0.2 | Semi-transparent overlay of 2 images |
| 4 | Copy-Paste 0.15 | Object from one image pasted onto another | Main image + floating object from second image |
| 5A | HSV H=0.05 | Hue shift | Color-shifted version (subtle green/blue shift) |
| 5B | HSV S=0.8 | Saturation boost | More vibrant/desaturated version |
| 5C | HSV V=0.5 | Value/brightness shift | Brighter/darker version |
| 6A | Rotate ±15° | Rotation by 15 degrees | Image tilted 15° with black corners |
| 6B | Scale ±50% | Scale transformation | Zoomed in/out version |
| 6C | Shear 5° | Shear distortion | Image slanted 5° |
| 7 | Flip Horizontal 50% | Left-right mirror | Horizontally flipped image |
| 8 | Augmentation Pipeline | All augmentations combined in sequence | Split panel showing original + each augmentation side-by-side |

### Grid Summary

Create a 3x3 grid showing Original + 8 key augmentations.

---

## Process 4: Backbone: CSPDarknet Architecture

Output: `waste_datasource/visualization/backbone/`

### Visualizations

| # | Image | Description | Type |
|---|-------|-------------|------|
| 1 | backbone_overview.jpg | Full backbone flowchart: Input(640x640x3) -> Stem(320x320xC1) -> Stage1(160x160xC2) -> Stage2(80x80xC3) -> Stage3(40x40xC4) -> Stage4(20x20xC5) -> SPP | Diagram |
| 2 | csp_stage_detail.jpg | CSP stage internals: split -> 2xConv -> concat -> Conv. Show 2 paths (main+branch) merging | Diagram |
| 3 | resolution_progression.jpg | Visual comparison of resolutions: 640x640 (full image) -> 320x320 (1/4) -> 160x160 (1/16) -> 80x80 (1/64) -> 40x40 (1/256) -> 20x20 (1/1024). Use overlaid grid lines to show how resolution shrinks | Image+Grid |
| 4 | feature_map_evolution.jpg | Conceptual visualization of what each stage detects: Stage1=edges, Stage2=patterns, Stage3=textures, Stage4=objects. Use color-coded heatmap style | Conceptual |
| 5 | csp_vs_standard.jpg | Comparison: Standard backbone (sequential) vs CSP (split+merge). Show FLOPs savings ~20% | Diagram |
| 6 | spine_detection_scales.jpg | SPP layer with 3 pooling sizes (k=5,9,13) showing different receptive fields on waste objects | Diagram |
| 7 | full_flow_annotated.jpg | Complete backbone flow with resolution, channel count, stride, and downsampling factor annotated at each stage | Diagram |

### Stage Table for Diagrams

| Stage | Input Resolution | Output Resolution | Channels | Stride | Function |
|-------|-----------------|-------------------|----------|--------|----------|
| Input | 640x640 | - | 3 (RGB) | 1x | Raw pixels |
| Stem | 640x640 | 320x320 | ~64 | 2x | Initial feature extraction |
| Stage 1 | 320x320 | 160x160 | 128 | 4x | Edge detection |
| Stage 2 | 160x160 | 80x80 | 256 | 8x | Pattern detection |
| Stage 3 | 80x80 | 40x40 | 512 | 16x | Texture detection |
| Stage 4 | 40x40 | 20x20 | 512 | 32x | Semantic understanding |
| SPP | 20x20 | 20x20 | 512 | 32x | Multi-scale context |

---

## Output Directory Structure (Each Step Folder Contains ALL Input Images)

```
waste_datasource/visualization/
├── pseudo_mask/
│   ├── 01_original_rgb/
│   │   ├── batch_1_000000.jpg
│   │   ├── batch_1_000001.jpg
│   │   ├── batch_1_000003.jpg
│   │   └── batch_1_000004.jpg
│   ├── 02_grayscale/
│   │   ├── batch_1_000000.jpg
│   │   ├── batch_1_000001.jpg
│   │   ├── batch_1_000003.jpg
│   │   └── batch_1_000004.jpg
│   ├── 03_gaussian_blur_5x5/
│   │   └── ... (same 4 images)
│   ├── 04_otsu_threshold/
│   │   └── ... (same 4 images)
│   ├── 05_mean_check_diagram/
│   │   └── ... (same 4 images, each with its own mean value)
│   ├── 06a_invert_mask/
│   │   └── ... (same 4 images)
│   ├── 06b_morphological_close/
│   │   └── ... (same 4 images)
│   ├── 07_find_contours/
│   │   └── ... (same 4 images, contours drawn)
│   ├── 08_area_check_diagram/
│   │   └── ... (same 4 images)
│   ├── 09a_edge_success_mask/
│   │   └── ... (edge-detected images)
│   ├── 09b_fallback_ellipse/
│   │   └── ... (images that needed ellipse fallback)
│   ├── 09c_fallback_rounded_rect/
│   │   └── ... (images that needed rect fallback)
│   ├── 10_approximate_polygon/
│   │   └── ... (same 4 images, polygon overlay)
│   ├── 11_normalized_coordinates/
│   │   └── ... (same 4 images, coords displayed)
│   ├── 12_yolo_seg_label/
│   │   └── ... (same 4 images, label text)
│   └── summary_collage_4x4/
│       └── step_comparison.jpg  (single collage of all steps)
├── stratified_split/
│   ├── 01_split_bar_chart/
│   │   └── chart.jpg
│   ├── 02_split_pie_chart/
│   │   └── chart.jpg
│   ├── 03_class_distribution_chart/
│   │   └── chart.jpg
│   └── 04_split_table/
│       └── table.jpg
├── augmentation/
│   ├── 01_original/
│   │   └── ... (all input images)
│   ├── 02_mosaic/
│   │   └── ... (mosaic composites)
│   ├── 03_mixup/
│   │   └── ... (blended pairs)
│   ├── 04_copy_paste/
│   │   └── ... (pasted composites)
│   ├── 05a_hsv_hue/
│   │   └── ... (all images, hue shifted)
│   ├── 05b_hsv_saturation/
│   │   └── ... (all images, saturation shifted)
│   ├── 05c_hsv_value/
│   │   └── ... (all images, value shifted)
│   ├── 06a_rotate/
│   │   └── ... (all images, rotated)
│   ├── 06b_scale/
│   │   └── ... (all images, scaled)
│   ├── 06c_shear/
│   │   └── ... (all images, sheared)
│   ├── 07_flip_horizontal/
│   │   └── ... (all images, flipped)
│   ├── 08_augmentation_pipeline_collage/
│   │   └── collage_per_image.jpg  (one collage per image)
│   └── 09_summary_grid_3x3/
│       └── grid.jpg
└── backbone/
    ├── 01_overview_flowchart/
    │   └── diagram.jpg
    ├── 02_csp_stage_detail/
    │   └── diagram.jpg
    ├── 03_resolution_progression/
    │   └── diagram.jpg
    ├── 04_feature_map_evolution/
    │   └── diagram.jpg
    ├── 05_csp_vs_standard/
    │   └── diagram.jpg
    ├── 06_spp_multi_scale/
    │   └── diagram.jpg
    └── 07_full_flow_annotated/
        └── diagram.jpg
```

**Key design: each step folder can contain multiple images. You pick any image (e.g., batch_1_000001.jpg) and trace it through 01 → 02 → 03 → ... → 12 to see its full pipeline history.**

Total step folders: 16 + 4 + 13 + 7 = 40 folders. Images per folder vary (4 for pseudo-mask demo, all dataset for augmentations).

## Implementation Notes

## Cross-Reference: Every Step → One Folder (Contains N Images)

### Pseudo-Mask (16 step folders -- ALL images from waste_raw_dataset)
| Step # | Folder | Contains | Images per folder |
|---|---|---|---|
| 1 (RGB) | `01_original_rgb/` | All original inputs | ALL (2,715 images) |
| 2 (Grayscale) | `02_grayscale/` | All grayscale | ALL |
| 3 (Gaussian Blur) | `03_gaussian_blur_5x5/` | All blurred | ALL |
| 4 (Otsu Threshold) | `04_otsu_threshold/` | All binary masks | ALL |
| 5 (Mean > 127?) | `05_mean_check_diagram/` | All + mean value overlay | ALL |
| 6A (Invert) | `06a_invert_mask/` | All inverted | ALL |
| 6B (Morph Close) | `06b_morphological_close/` | All closed | ALL |
| 7 (Find Contours) | `07_find_contours/` | All + contour drawn | ALL |
| 8 (Area >= 20%?) | `08_area_check_diagram/` | All + area % text | ALL |
| 9A (Edge Success) | `09a_edge_success_mask/` | Edge-detected subset | subset |
| 9B (Fallback Ellipse) | `09b_fallback_ellipse/` | Ellipse fallback subset | subset |
| 9C (Fallback Rounded Rect) | `09c_fallback_rounded_rect/` | Rounded rect subset | subset |
| 10 (Approx Polygon) | `10_approximate_polygon/` | All + polygon overlay | ALL |
| 11 (Normalize) | `11_normalized_coordinates/` | All + coords text | ALL |
| 12 (YOLO Label) | `12_yolo_seg_label/` | All + label text | ALL |
| Summary | `summary_collage_4x4/` | Step comparison chart | 1 collage |

### Stratified Split (4 step folders)
| Step # | Folder | Contains | Files |
|---|---|---|---|
| 1 (Bar Chart) | `01_split_bar_chart/` | Bar chart | chart.jpg |
| 2 (Pie Chart) | `02_split_pie_chart/` | Pie chart | chart.jpg |
| 3 (Class Distribution) | `03_class_distribution_chart/` | Grouped bar chart | chart.jpg |
| 4 (Table) | `04_split_table/` | Data table | table.jpg |

### Augmentation (13 step folders -- ALL images from backend/dataset/raw/)
| Step # | Folder | Contains | Images per folder |
|---|---|---|---|
| 1 (Original) | `01_original/` | All base images | ALL (3,973) |
| 2 (Mosaic) | `02_mosaic/` | 2x2 grid composites | ALL/4 |
| 3 (Mixup) | `03_mixup/` | Blended pairs | ALL/2 |
| 4 (Copy-Paste) | `04_copy_paste/` | Pasted composites | ALL |
| 5A (HSV Hue) | `05a_hsv_hue/` | All hue shifted | ALL |
| 5B (HSV Sat) | `05b_hsv_saturation/` | All sat shifted | ALL |
| 5C (HSV Val) | `05c_hsv_value/` | All value shifted | ALL |
| 6A (Rotate) | `06a_rotate/` | All rotated 15deg | ALL |
| 6B (Scale) | `06b_scale/` | All scaled ±50% | ALL |
| 6C (Shear) | `06c_shear/` | All sheared 5deg | ALL |
| 7 (Flip) | `07_flip_horizontal/` | All horizontally flipped | ALL |
| 8 (Pipeline Collage) | `08_augmentation_pipeline_collage/` | Per-image comparison | 1 per image |
| 9 (3x3 Grid) | `09_summary_grid_3x3/` | Summary grid | 1 grid |

### Backbone (7 step folders)
| Step # | Folder | Contains | Files |
|---|---|---|---|
| 1 (Overview) | `01_overview_flowchart/` | Architecture flowchart | diagram.jpg |
| 2 (CSP Detail) | `02_csp_stage_detail/` | CSP stage internals | diagram.jpg |
| 3 (Resolution) | `03_resolution_progression/` | Resolution comparison | diagram.jpg |
| 4 (Feature Evolution) | `04_feature_map_evolution/` | Conceptual feature maps | diagram.jpg |
| 5 (CSP vs Standard) | `05_csp_vs_standard/` | Comparison diagram | diagram.jpg |
| 6 (SPP Multi-Scale) | `06_spp_multi_scale/` | SPP layer diagram | diagram.jpg |
| 7 (Full Flow) | `07_full_flow_annotated/` | Complete annotated flow | diagram.jpg |

**Total: 40 folders. Pseudo-mask processes ALL 2,715 images. Augmentation processes ALL 3,973 images. Trace any image by filename across folders 01→02→03→...→12.**

---

### Libraries
- OpenCV (cv2): image I/O, processing, drawing
- Pillow (PIL): additional image handling, text rendering
- NumPy: array operations
- Matplotlib: charts (bar, pie) -> save as image

### Conventions
- All output JPG, quality=95
- Add step number as large text overlay (white with black shadow, top-left)
- Add brief description text (bottom)
- Consistent canvas size: 800x800 or fitting aspect ratio
- Font: DejaVuSans for text labels
- Diagrams: use OpenCV drawing functions (rectangle, circle, line, putText)
- Arrows: use cv2.arrowedLine or custom triangle markers
- Color coding: consistent palette across all images
