import math
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm
from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
JPG_QUALITY = 95
CANVAS_W, CANVAS_H = 800, 800

COLOR_BLUE = (66, 133, 244)
COLOR_GREEN = (52, 168, 83)
COLOR_ORANGE = (251, 188, 4)
COLOR_RED = (234, 67, 53)
COLOR_PURPLE = (147, 112, 219)
COLOR_TEAL = (0, 150, 136)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GRAY = (128, 128, 128)
COLOR_LGRAY = (200, 200, 200)
COLOR_DGRAY = (64, 64, 64)

SAMPLE_IMG = Path(__file__).resolve().parent.parent.parent.parent / "waste_datasource" / "inferencedata" / "batch_1_000000.jpg"
VIZ_BASE = Path(__file__).resolve().parent.parent.parent.parent / "waste_datasource" / "visualization"


def _font(size=18):
    return ImageFont.truetype(FONT_PATH, size)


def _font_bold(size=18):
    return ImageFont.truetype(FONT_BOLD_PATH, size)


def _pil_from_cv(img_bgr):
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def _cv_from_pil(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def _overlay_step(img_bgr, step_no, description, font_size=28, desc_size=18):
    pil = _pil_from_cv(img_bgr)
    draw = ImageDraw.Draw(pil)
    text = f"Step {step_no}"
    ft = _font_bold(font_size)
    bbox = draw.textbbox((0, 0), text, font=ft)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    draw.text((12, 10), text, font=ft, fill=(0, 0, 0))
    draw.text((10, 8), text, font=ft, fill=(255, 255, 255))
    if description:
        ft2 = _font(desc_size)
        db = draw.textbbox((0, 0), description, font=ft2)
        dw = db[2] - db[0]
        dh = db[3] - db[1]
        xx = (CANVAS_W - dw) // 2
        yy = CANVAS_H - dh - 15
        draw.text((xx + 1, yy + 1), description, font=ft2, fill=(0, 0, 0))
        draw.text((xx, yy), description, font=ft2, fill=(255, 255, 255))
    return _cv_from_pil(pil)


def _overlay_text(img_bgr, text, pos, font_size=20, color=(255, 255, 255), shadow=True, center_x=False):
    pil = _pil_from_cv(img_bgr)
    draw = ImageDraw.Draw(pil)
    ft = _font(font_size)
    x, y = pos
    if center_x:
        bb = draw.textbbox((0, 0), text, font=ft)
        x = (CANVAS_W - (bb[2] - bb[0])) // 2
    if shadow:
        draw.text((x + 1, y + 1), text, font=ft, fill=(0, 0, 0))
    draw.text((x, y), text, font=ft, fill=color)
    return _cv_from_pil(pil)


def _blank_canvas(bg_color=(30, 30, 30)):
    return np.full((CANVAS_H, CANVAS_W, 3), bg_color, dtype=np.uint8)


def _draw_arrow(img, x1, y1, x2, y2, color=(200, 200, 200), thickness=2):
    cv2.arrowedLine(img, (x1, y1), (x2, y2), color, thickness, tipLength=0.08)


def _draw_box(img, x1, y1, x2, y2, color, label="", fill=None, font_size=16):
    if fill:
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), fill, -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    if label:
        pil = _pil_from_cv(img)
        draw = ImageDraw.Draw(pil)
        ft = _font(font_size)
        draw.text((x1 + 8, y1 + 6), label, font=ft, fill=(255, 255, 255))
        img[:] = _cv_from_pil(pil)
    return img


def _save(img_bgr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY])



def process_pseudo_mask():
    img = cv2.imread(str(SAMPLE_IMG))
    if img is None:
        raise FileNotFoundError(f"Sample image not found: {SAMPLE_IMG}")
    img = cv2.resize(img, (640, 640))
    h, w = img.shape[:2]
    base = VIZ_BASE / "pseudo_mask"

    # Step 1: Original RGB
    d = base / "01_original_rgb"
    out = _overlay_step(img.copy(), 1, "Original RGB image loaded from disk")
    _save(out, d / "step_01_original_rgb.jpg")

    # Step 2: Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    d = base / "02_grayscale"
    out = _overlay_step(gray_bgr, 2, "RGB -> single channel luminance")
    _save(out, d / "step_02_grayscale.jpg")

    # Step 3: Gaussian Blur 5x5
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    blr_bgr = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
    d = base / "03_gaussian_blur_5x5"
    out = _overlay_step(blr_bgr, 3, "Gaussian blur 5x5 kernel")
    _save(out, d / "step_03_gaussian_blur.jpg")

    # Step 4: Otsu Threshold
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    d = base / "04_otsu_threshold"
    out = _overlay_step(thresh_bgr, 4, "Otsu auto-threshold binarization")
    _save(out, d / "step_04_otsu_threshold.jpg")

    # Step 5: Mean > 127? (diagram overlaid on image)
    mean_val = np.mean(thresh)
    canvas5 = _blank_canvas((40, 40, 40))
    _draw_box(canvas5, 150, 100, 650, 300, COLOR_BLUE, "Mean Pixel Value")
    _draw_box(canvas5, 150, 380, 350, 520, COLOR_GREEN, "Yes >127")
    _draw_box(canvas5, 450, 380, 650, 520, COLOR_ORANGE, "No <=127")
    _draw_arrow(canvas5, 400, 300, 250, 380, COLOR_WHITE)
    _draw_arrow(canvas5, 400, 300, 550, 380, COLOR_WHITE)
    canvas5 = _overlay_text(canvas5, f"Mean = {mean_val:.1f}", (300, 190), font_size=24, color=(255, 255, 255))
    d = base / "05_mean_check_diagram"
    out = _overlay_step(canvas5, 5, "Check if average pixel > 127")
    _save(out, d / "step_05_mean_check.jpg")

    # Step 6A: Invert Binary Mask
    if mean_val > 127:
        thresh = cv2.bitwise_not(thresh)
    inv_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    d = base / "06a_invert_mask"
    out = _overlay_step(inv_bgr, "6A", "Invert binary mask (black<->white)")
    _save(out, d / "step_06a_invert_mask.jpg")

    # Step 6B: Morphological Close
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    closed_bgr = cv2.cvtColor(closed, cv2.COLOR_GRAY2BGR)
    d = base / "06b_morphological_close"
    out = _overlay_step(closed_bgr, "6B", "Morphological close 5x5, 2 iterations")
    _save(out, d / "step_06b_morph_close.jpg")

    # Step 7: Find Contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img7 = img.copy()
    largest = max(contours, key=cv2.contourArea) if contours else None
    if largest is not None:
        cv2.drawContours(img7, [largest], -1, (0, 255, 0), 3)
    d = base / "07_find_contours"
    out = _overlay_step(img7, 7, "Largest contour outlined in GREEN")
    _save(out, d / "step_07_find_contours.jpg")

    # Step 8: Area >= 20%?
    canvas8 = _blank_canvas((40, 40, 40))
    if largest is not None:
        area_pct = cv2.contourArea(largest) / (h * w) * 100
    else:
        area_pct = 0
    _draw_box(canvas8, 150, 100, 650, 300, COLOR_BLUE, f"Contour area: {area_pct:.1f}% of image")
    _draw_box(canvas8, 150, 380, 350, 520, COLOR_GREEN, "Yes >=20%")
    _draw_box(canvas8, 450, 380, 650, 520, COLOR_ORANGE, "No <20%")
    _draw_arrow(canvas8, 400, 300, 250, 380, COLOR_WHITE)
    _draw_arrow(canvas8, 400, 300, 550, 380, COLOR_WHITE)
    d = base / "08_area_check_diagram"
    out = _overlay_step(canvas8, 8, "Check if contour area >= 20% of image")
    _save(out, d / "step_08_area_check.jpg")

    # Step 9A: Edge Success (if area >= 20%)
    img9a = img.copy()
    if largest is not None and area_pct >= 20:
        overlay = img9a.copy()
        cv2.drawContours(overlay, [largest], -1, (0, 255, 0), -1)
        img9a = cv2.addWeighted(overlay, 0.4, img9a, 0.6, 0)
        cv2.drawContours(img9a, [largest], -1, (0, 255, 0), 2)
    d = base / "09a_edge_success_mask"
    out = _overlay_step(img9a, "9A", "Edge detection result (semi-transparent green fill)")
    _save(out, d / "step_09a_edge_success.jpg")

    # Step 9B: Fallback Ellipse
    img9b = img.copy()
    cx, cy = w // 2, h // 2
    axes = (w // 3, h // 3)
    cv2.ellipse(img9b, (cx, cy), axes, 0, 0, 360, COLOR_BLUE, 3)
    d = base / "09b_fallback_ellipse"
    out = _overlay_step(img9b, "9B", "Fallback: elliptical polygon (blue dashed)")
    _save(out, d / "step_09b_fallback_ellipse.jpg")

    # Step 9C: Fallback Rounded Rect
    img9c = img.copy()
    p1, p2 = (w // 5, h // 5), (4 * w // 5, 4 * h // 5)
    cv2.rectangle(img9c, p1, p2, COLOR_ORANGE, 3)
    cv2.circle(img9c, (p1[0] + 30, p1[1] + 30), 30, COLOR_ORANGE, 3)
    cv2.circle(img9c, (p2[0] - 30, p1[1] + 30), 30, COLOR_ORANGE, 3)
    cv2.circle(img9c, (p1[0] + 30, p2[1] - 30), 30, COLOR_ORANGE, 3)
    cv2.circle(img9c, (p2[0] - 30, p2[1] - 30), 30, COLOR_ORANGE, 3)
    d = base / "09c_fallback_rounded_rect"
    out = _overlay_step(img9c, "9C", "Fallback: rounded rectangle (orange dashed)")
    _save(out, d / "step_09c_fallback_rect.jpg")

    # Step 10: Approx Polygon
    img10 = img.copy()
    if largest is not None:
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        cv2.drawContours(img10, [approx], -1, COLOR_RED, 3)
        for pt in approx:
            px, py = pt[0]
            cv2.circle(img10, (px, py), 5, COLOR_RED, -1)
        img10 = _overlay_text(img10, f"Vertices: {len(approx)}", (10, h - 60), font_size=22, color=COLOR_RED)
    d = base / "10_approximate_polygon"
    out = _overlay_step(img10, 10, "Approx polygon (RED) with vertices")
    _save(out, d / "step_10_approx_polygon.jpg")

    # Step 11: Normalize to [0,1]
    img11 = img.copy()
    if largest is not None:
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        cv2.drawContours(img11, [approx], -1, COLOR_PURPLE, 2)
        norm_coords = []
        for pt in approx:
            px, py = pt[0]
            nx, ny = px / w, py / h
            norm_coords.append((nx, ny))
            cv2.circle(img11, (px, py), 4, COLOR_PURPLE, -1)
        label = f"[{norm_coords[0][0]:.3f},{norm_coords[0][1]:.3f}] ..."
        img11 = _overlay_text(img11, label, (10, h - 60), font_size=16, color=COLOR_PURPLE)
    d = base / "11_normalized_coordinates"
    out = _overlay_step(img11, 11, "Normalized coordinates [x/w, y/h]")
    _save(out, d / "step_11_normalized.jpg")

    # Step 12: YOLO-seg Label
    img12 = img.copy()
    if largest is not None:
        epsilon = 0.01 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)
        cv2.drawContours(img12, [approx], -1, COLOR_TEAL, 2)
        parts = []
        for pt in approx:
            parts.append(f"{pt[0][0] / w:.4f}")
            parts.append(f"{pt[0][1] / h:.4f}")
        label_str = "0 " + " ".join(parts[:12]) + " ..."
        img12 = _overlay_text(img12, label_str, (10, 50), font_size=14, color=COLOR_TEAL)
    d = base / "12_yolo_seg_label"
    out = _overlay_step(img12, 12, "Final YOLO-seg label format")
    _save(out, d / "step_12_yolo_label.jpg")

    # Summary Collage 4x4
    step_imgs = []
    for step in range(1, 13):
        if step in (6, 9):
            for sub in ("a", "b", "c"):
                spath = base / f"0{step}{sub}_*.jpg"
        paths = sorted((base / f"0{step}_*".replace("*", "")).parent.glob(f"*step_*"))
    d = base / "summary_collage_4x4"
    img_list = sorted(base.rglob("step_*.jpg"))
    if img_list:
        sel = img_list[:16]
        rows, cols = 4, 4
        cell_w, cell_h = CANVAS_W // cols, CANVAS_H // rows
        collage = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        for i, p in enumerate(sel):
            r, c = i // cols, i % cols
            tile = cv2.imread(str(p))
            if tile is not None:
                tile = cv2.resize(tile, (cell_w, cell_h))
                collage[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = tile
        collage = _overlay_text(collage, "Pseudo-Mask Pipeline: 16 Steps", (0, 5), font_size=20, color=COLOR_WHITE, center_x=True)
        _save(collage, d / "summary_collage_4x4.jpg")

    return {"folders": ["pseudo_mask/01_original_rgb", "pseudo_mask/02_grayscale", "pseudo_mask/03_gaussian_blur_5x5", "pseudo_mask/04_otsu_threshold", "pseudo_mask/05_mean_check_diagram", "pseudo_mask/06a_invert_mask", "pseudo_mask/06b_morphological_close", "pseudo_mask/07_find_contours", "pseudo_mask/08_area_check_diagram", "pseudo_mask/09a_edge_success_mask", "pseudo_mask/09b_fallback_ellipse", "pseudo_mask/09c_fallback_rounded_rect", "pseudo_mask/10_approximate_polygon", "pseudo_mask/11_normalized_coordinates", "pseudo_mask/12_yolo_seg_label", "pseudo_mask/summary_collage_4x4"]}



def process_stratified_split():
    base = VIZ_BASE / "stratified_split"
    train_c, val_c, test_c = 2765, 593, 593
    train_o, val_o, test_o = 476, 102, 102
    train_n, val_n, test_n = 2289, 491, 491
    labels = ["Train", "Val", "Test"]
    totals = [train_c, val_c, test_c]
    orgs = [train_o, val_o, test_o]
    norgs = [train_n, val_n, test_n]
    pcts = [70, 15, 15]
    colors_hex = ["#4285F4", "#34A853", "#FBBC04"]
    fm.fontManager.addfont(FONT_PATH)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FONT_PATH).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    # Step 1: Horizontal bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels, totals, color=colors_hex)
    for bar, tot, pct in zip(bars, totals, pcts):
        ax.text(bar.get_width() + 30, bar.get_y() + bar.get_height() / 2,
                f"{tot:,} ({pct}%)", va="center", fontsize=12)
    ax.set_xlabel("Image Count")
    ax.set_title("Stratified Split 70/15/15")
    ax.margins(x=0.2)
    fig.tight_layout()
    d = base / "01_split_bar_chart"
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(d / "chart.jpg"), dpi=120)
    plt.close(fig)

    # Step 2: Pie chart
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    wedges, texts, autotexts = ax2.pie(
        totals, labels=labels, autopct="%1.1f%%", startangle=90,
        colors=colors_hex, textprops={"fontsize": 12}
    )
    ax2.set_title("Split Distribution (70/15/15)", fontsize=14)
    d = base / "02_split_pie_chart"
    d.mkdir(parents=True, exist_ok=True)
    fig2.savefig(str(d / "chart.jpg"), dpi=120)
    plt.close(fig2)

    # Step 3: Class distribution (grouped bar)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    bw = 0.35
    ax3.bar(x - bw / 2, orgs, bw, label="Organik", color="#2E7D32")
    ax3.bar(x + bw / 2, norgs, bw, label="Non-Organik", color="#C62828")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylabel("Image Count")
    ax3.set_title("Per-Split Class Distribution")
    ax3.legend()
    fig3.tight_layout()
    d = base / "03_class_distribution_chart"
    d.mkdir(parents=True, exist_ok=True)
    fig3.savefig(str(d / "chart.jpg"), dpi=120)
    plt.close(fig3)

    # Step 4: Table
    fig4, ax4 = plt.subplots(figsize=(8, 3))
    ax4.axis("off")
    col_labels = ["Split", "Total", "Organik", "Non-Organik", "% of Total"]
    rows_data = [
        ["Train", f"{train_c:,}", f"{train_o:,}", f"{train_n:,}", "70%"],
        ["Val", f"{val_c:,}", f"{val_o:,}", f"{val_n:,}", "15%"],
        ["Test", f"{test_c:,}", f"{test_o:,}", f"{test_n:,}", "15%"],
    ]
    table = ax4.table(cellText=rows_data, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.8)
    for (i, j), cell in table.get_celld().items():
        if i == 0:
            cell.set_facecolor("#4285F4")
            cell.set_text_props(color="white", fontweight="bold")
        elif i % 2 == 0:
            cell.set_facecolor("#f0f0f0")
    d = base / "04_split_table"
    d.mkdir(parents=True, exist_ok=True)
    fig4.savefig(str(d / "table.jpg"), dpi=120, bbox_inches="tight")
    plt.close(fig4)

    return {"folders": ["stratified_split/01_split_bar_chart", "stratified_split/02_split_pie_chart", "stratified_split/03_class_distribution_chart", "stratified_split/04_split_table"]}



def _load_sample_images():
    infer_dir = SAMPLE_IMG.parent
    paths = sorted(infer_dir.glob("*.jpg"))[:4]
    imgs = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            imgs.append(cv2.resize(img, (320, 320)))
    while len(imgs) < 4:
        imgs.append(np.full((320, 320, 3), 128, dtype=np.uint8))
    return imgs


def process_augmentation():
    imgs = _load_sample_images()
    main_img = cv2.imread(str(SAMPLE_IMG))
    if main_img is None:
        main_img = np.full((640, 640, 3), 128, dtype=np.uint8)
    main_img = cv2.resize(main_img, (640, 640))
    base = VIZ_BASE / "augmentation"

    # Step 1: Original
    d = base / "01_original"
    out = _overlay_step(main_img.copy(), 1, "Original image before augmentation")
    _save(out, d / "step_01_original.jpg")

    # Step 2: Mosaic 1.0 (2x2 grid)
    mosaic = np.zeros((640, 640, 3), dtype=np.uint8)
    for i, sm in enumerate(imgs):
        r, c = i // 2, i % 2
        mosaic[r * 320:(r + 1) * 320, c * 320:(c + 1) * 320] = sm
    cv2.rectangle(mosaic, (320, 0), (320, 640), (255, 255, 255), 3)
    cv2.rectangle(mosaic, (0, 320), (640, 320), (255, 255, 255), 3)
    d = base / "02_mosaic"
    out = _overlay_step(mosaic, 2, "Mosaic 1.0 — 4 images in 2x2 grid")
    _save(out, d / "step_02_mosaic.jpg")

    # Step 3: Mixup 0.2
    mixup = cv2.addWeighted(imgs[0], 0.8, cv2.resize(imgs[1], (320, 320)), 0.2, 0)
    mixup_big = cv2.resize(mixup, (640, 640))
    d = base / "03_mixup"
    out = _overlay_step(mixup_big, 3, "Mixup alpha=0.2 — two images blended")
    _save(out, d / "step_03_mixup.jpg")

    # Step 4: Copy-Paste
    copy_paste = main_img.copy()
    obj = cv2.resize(imgs[2], (120, 120))
    x_off, y_off = 400, 450
    copy_paste[y_off:y_off + 120, x_off:x_off + 120] = obj
    cv2.rectangle(copy_paste, (x_off, y_off), (x_off + 120, y_off + 120), COLOR_ORANGE, 3)
    d = base / "04_copy_paste"
    out = _overlay_step(copy_paste, 4, "Copy-Paste 0.15 — object pasted onto another image")
    _save(out, d / "step_04_copy_paste.jpg")

    # Step 5A: HSV Hue shift
    hsv = cv2.cvtColor(main_img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + 18) % 180
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv_hue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    d = base / "05a_hsv_hue"
    out = _overlay_step(hsv_hue, "5A", "HSV Hue shift (H=0.05)")
    _save(out, d / "step_05a_hsv_hue.jpg")

    # Step 5B: HSV Saturation boost
    hsv = cv2.cvtColor(main_img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * 1.8
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    d = base / "05b_hsv_saturation"
    out = _overlay_step(hsv_sat, "5B", "HSV Saturation boost (S=0.8)")
    _save(out, d / "step_05b_hsv_saturation.jpg")

    # Step 5C: HSV Value shift
    hsv = cv2.cvtColor(main_img.copy(), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = hsv[:, :, 2] * 0.5
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    hsv_val = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    d = base / "05c_hsv_value"
    out = _overlay_step(hsv_val, "5C", "HSV Value/brightness shift (V=0.5)")
    _save(out, d / "step_05c_hsv_value.jpg")

    # Step 6A: Rotate 15 deg
    h6, w6 = main_img.shape[:2]
    M_rot = cv2.getRotationMatrix2D((w6 // 2, h6 // 2), 15, 1.0)
    rot_img = cv2.warpAffine(main_img, M_rot, (w6, h6), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    d = base / "06a_rotate"
    out = _overlay_step(rot_img, "6A", "Rotate 15 degrees with black corners")
    _save(out, d / "step_06a_rotate.jpg")

    # Step 6B: Scale 50%
    M_scale = cv2.getRotationMatrix2D((w6 // 2, h6 // 2), 0, 0.5)
    scaled = cv2.warpAffine(main_img, M_scale, (w6, h6), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    d = base / "06b_scale"
    out = _overlay_step(scaled, "6B", "Scale 50% zoom out")
    _save(out, d / "step_06b_scale.jpg")

    # Step 6C: Shear 5 deg
    shear_m = np.float32([[1, math.tan(math.radians(5)), 0],
                          [0, 1, 0]])
    sheared = cv2.warpAffine(main_img, shear_m, (w6, h6), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    d = base / "06c_shear"
    out = _overlay_step(sheared, "6C", "Shear distortion 5 degrees")
    _save(out, d / "step_06c_shear.jpg")

    # Step 7: Flip Horizontal
    flipped = cv2.flip(main_img, 1)
    d = base / "07_flip_horizontal"
    out = _overlay_step(flipped, 7, "Horizontal flip 50%")
    _save(out, d / "step_07_flip.jpg")

    # Step 8: Pipeline Collage (split panel)
    augs = [
        ("Original", main_img), ("Mosaic", mosaic),
        ("Mixup", mixup_big), ("Copy-Paste", copy_paste),
        ("HSV Hue", hsv_hue), ("HSV Sat", hsv_sat),
        ("HSV Val", hsv_val), ("Rotate", rot_img),
        ("Scale", scaled), ("Shear", sheared),
        ("Flip", flipped),
    ]
    n_augs = len(augs)
    cols, rows = 4, 3
    cell_w, cell_h = CANVAS_W // cols, CANVAS_H // rows
    collage8 = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    for i, (name, aimg) in enumerate(augs):
        r, c = i // cols, i % cols
        if r >= rows:
            break
        tile = cv2.resize(aimg, (cell_w, cell_h))
        tile = _overlay_text(tile, name, (4, 4), font_size=12, color=(255, 255, 255))
        collage8[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w] = tile
    d = base / "08_augmentation_pipeline_collage"
    _save(collage8, d / "collage.jpg")

    # Step 9: 3x3 Summary Grid
    cols9, rows9 = 3, 3
    cw9, ch9 = CANVAS_W // cols9, CANVAS_H // rows9
    grid9 = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    grid_augs = augs[:9]
    for i, (name, aimg) in enumerate(grid_augs):
        r, c = i // cols9, i % cols9
        tile = cv2.resize(aimg, (cw9, ch9))
        tile = _overlay_text(tile, name, (4, 4), font_size=14, color=(255, 255, 255))
        grid9[r * ch9:(r + 1) * ch9, c * cw9:(c + 1) * cw9] = tile
    d = base / "09_summary_grid_3x3"
    _save(grid9, d / "summary_grid_3x3.jpg")

    return {"folders": ["augmentation/01_original", "augmentation/02_mosaic", "augmentation/03_mixup", "augmentation/04_copy_paste", "augmentation/05a_hsv_hue", "augmentation/05b_hsv_saturation", "augmentation/05c_hsv_value", "augmentation/06a_rotate", "augmentation/06b_scale", "augmentation/06c_shear", "augmentation/07_flip_horizontal", "augmentation/08_augmentation_pipeline_collage", "augmentation/09_summary_grid_3x3"]}



def process_backbone():
    base = VIZ_BASE / "backbone"

    # Step 1: backbone_overview (full architecture flowchart)
    canvas1 = _blank_canvas((35, 35, 35))
    stages = [
        ("Input\n640x640x3", 60, 30, 180, 90),
        ("Stem\n320x320x64", 60, 140, 180, 210),
        ("Stage1\n160x160x128", 60, 250, 180, 320),
        ("Stage2\n80x80x256", 60, 360, 180, 430),
        ("Stage3\n40x40x512", 60, 470, 180, 540),
        ("Stage4\n20x20x512", 60, 580, 180, 650),
        ("SPP\n20x20x512", 60, 690, 180, 760),
    ]
    for label, x1, y1, x2, y2 in stages:
        _draw_box(canvas1, x1, y1, x2, y2, COLOR_BLUE, label, fill=(50, 50, 50))
    for i in range(len(stages) - 1):
        y1b = stages[i][4]
        y2t = stages[i + 1][2]
        _draw_arrow(canvas1, 120, y1b, 120, y2t, COLOR_WHITE)
    # Right side resolution labels
    res_labels = ["1x", "2x", "4x", "8x", "16x", "32x", "32x"]
    for i, rl in enumerate(res_labels):
        canvas1 = _overlay_text(canvas1, f"Stride: {rl}", (240, stages[i][2] + 20), font_size=14, color=COLOR_ORANGE)
    d = base / "01_overview_flowchart"
    out = _overlay_step(canvas1, 1, "CSPDarknet backbone: full architecture flow")
    _save(out, d / "diagram.jpg")

    # Step 2: CSP Stage Detail
    canvas2 = _blank_canvas((35, 35, 35))
    _draw_box(canvas2, 50, 50, 200, 130, COLOR_BLUE, "Input\nStage", fill=(50, 50, 50))
    _draw_box(canvas2, 50, 200, 200, 330, COLOR_BLUE, "Branch Path\n2x Conv", fill=(50, 50, 50))
    _draw_box(canvas2, 280, 200, 430, 330, COLOR_GREEN, "Main Path\n2x Conv", fill=(50, 50, 50))
    _draw_box(canvas2, 50, 400, 200, 500, COLOR_ORANGE, "Split", fill=(50, 50, 50))
    _draw_box(canvas2, 280, 400, 430, 530, COLOR_PURPLE, "Concat + Conv", fill=(50, 50, 50))
    _draw_box(canvas2, 280, 600, 430, 700, COLOR_TEAL, "Output\nStage", fill=(50, 50, 50))
    _draw_arrow(canvas2, 125, 130, 125, 200, COLOR_WHITE)
    _draw_arrow(canvas2, 125, 330, 125, 400, COLOR_WHITE)
    _draw_arrow(canvas2, 200, 265, 280, 265, COLOR_WHITE)
    _draw_arrow(canvas2, 355, 330, 355, 400, COLOR_WHITE)
    _draw_arrow(canvas2, 355, 530, 355, 600, COLOR_WHITE)
    d = base / "02_csp_stage_detail"
    out = _overlay_step(canvas2, 2, "CSP stage internals: split → 2 paths → concat")
    _save(out, d / "diagram.jpg")

    # Step 3: Resolution Progression
    canvas3 = _blank_canvas((35, 35, 35))
    sizes = [(640, 0), (320, 1), (160, 2), (80, 3), (40, 4), (20, 5)]
    for i in range(len(sizes)):
        sz, idx = sizes[i]
        factor = 64 // (2 ** i) if i < 5 else 2
        w_b = 40 + i * 120
        h_b = 250
        box_sz = max(10, sz // 10)
        color = [(66, 133, 244), (52, 168, 83), (251, 188, 4), (234, 67, 53), (147, 112, 219), (0, 150, 136)][i]
        cv2.rectangle(canvas3, (w_b, h_b), (w_b + box_sz, h_b + box_sz), color, 2)
        canvas3 = _overlay_text(canvas3, f"{sz}x{sz}", (w_b - 10, h_b + box_sz + 10), font_size=11, color=color)
        if i > 0:
            p_w = 40 + (i - 1) * 120
            _draw_arrow(canvas3, p_w + max(10, sizes[i - 1][0] // 10) + 5, h_b + max(10, sizes[i - 1][0] // 10) // 2,
                       w_b, h_b + box_sz // 2, COLOR_WHITE)
    d = base / "03_resolution_progression"
    out = _overlay_step(canvas3, 3, "Resolution progression: 640 → 320 → 160 → 80 → 40 → 20")
    _save(out, d / "diagram.jpg")

    # Step 4: Feature Map Evolution
    canvas4 = _blank_canvas((35, 35, 35))
    feat_labels = [
        ("Stage 1\nEdges", COLOR_BLUE),
        ("Stage 2\nPatterns", COLOR_GREEN),
        ("Stage 3\nTextures", COLOR_ORANGE),
        ("Stage 4\nSemantics", COLOR_RED),
    ]
    for i, (label, color) in enumerate(feat_labels):
        x, y = 60 + i * 180, 200
        _draw_box(canvas4, x, y, x + 130, y + 130, color, label, fill=(50, 50, 50))
        heat = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        canvas4[y + 10:y + 110, x + 15:x + 115] = cv2.resize(heat_color, (100, 100))
        if i > 0:
            _draw_arrow(canvas4, x - 30, 265, x, 265, COLOR_WHITE)
    d = base / "04_feature_map_evolution"
    out = _overlay_step(canvas4, 4, "Feature map evolution: edges → patterns → textures → objects")
    _save(out, d / "diagram.jpg")

    # Step 5: CSP vs Standard
    canvas5 = _blank_canvas((35, 35, 35))
    # Standard (left)
    _draw_box(canvas5, 30, 100, 170, 200, COLOR_GRAY, "Input", fill=(50, 50, 50))
    _draw_box(canvas5, 30, 250, 170, 350, COLOR_GRAY, "Conv x N", fill=(50, 50, 50))
    _draw_box(canvas5, 30, 400, 170, 500, COLOR_GRAY, "Output", fill=(50, 50, 50))
    _draw_arrow(canvas5, 100, 200, 100, 250, COLOR_WHITE)
    _draw_arrow(canvas5, 100, 350, 100, 400, COLOR_WHITE)
    canvas5 = _overlay_text(canvas5, "Standard", (50, 50), font_size=20, color=COLOR_RED)
    canvas5 = _overlay_text(canvas5, "~100% FLOPs", (40, 530), font_size=14, color=COLOR_RED)

    # CSP (right)
    _draw_box(canvas5, 330, 100, 470, 200, COLOR_BLUE, "Input", fill=(50, 50, 50))
    _draw_box(canvas5, 330, 250, 470, 320, COLOR_GREEN, "Branch", fill=(50, 50, 50))
    _draw_box(canvas5, 330, 360, 470, 430, COLOR_ORANGE, "Main", fill=(50, 50, 50))
    _draw_box(canvas5, 330, 480, 470, 580, COLOR_PURPLE, "Concat", fill=(50, 50, 50))
    _draw_box(canvas5, 330, 630, 470, 730, COLOR_TEAL, "Output", fill=(50, 50, 50))
    _draw_arrow(canvas5, 400, 200, 400, 250, COLOR_WHITE)
    _draw_arrow(canvas5, 400, 320, 400, 360, COLOR_WHITE)
    _draw_arrow(canvas5, 400, 430, 400, 480, COLOR_WHITE)
    _draw_arrow(canvas5, 400, 580, 400, 630, COLOR_WHITE)
    canvas5 = _overlay_text(canvas5, "CSP", (370, 50), font_size=20, color=COLOR_GREEN)
    canvas5 = _overlay_text(canvas5, "~80% FLOPs", (330, 760), font_size=14, color=COLOR_GREEN)

    d = base / "05_csp_vs_standard"
    out = _overlay_step(canvas5, 5, "CSP vs Standard backbone: ~20% FLOPs savings")
    _save(out, d / "diagram.jpg")

    # Step 6: SPP Multi-Scale
    canvas6 = _blank_canvas((35, 35, 35))
    _draw_box(canvas6, 300, 50, 500, 150, COLOR_BLUE, "Input\n20x20x512", fill=(50, 50, 50))
    pool_sizes = [5, 9, 13]
    for i, k in enumerate(pool_sizes):
        y = 220 + i * 140
        _draw_box(canvas6, 200, y, 600, y + 90, [(66, 133, 244), (52, 168, 83), (251, 188, 4)][i],
                  f"MaxPool k={k}x{k}", fill=(50, 50, 50))
        if i == 0:
            _draw_arrow(canvas6, 400, 150, 400, 220, COLOR_WHITE)
        else:
            _draw_arrow(canvas6, 300, y - 50, 300, y, COLOR_WHITE)
    _draw_box(canvas6, 200, 620, 600, 720, COLOR_PURPLE, "Concat + Conv", fill=(50, 50, 50))
    _draw_arrow(canvas6, 400, 590, 400, 620, COLOR_WHITE)
    d = base / "06_spp_multi_scale"
    out = _overlay_step(canvas6, 6, "SPP layer: multi-scale pooling (k=5,9,13)")
    _save(out, d / "diagram.jpg")

    # Step 7: Full Flow Annotated
    canvas7 = _blank_canvas((35, 35, 35))
    stages7 = [
        ("Input", "640x640x3", "1x", 30, 20),
        ("Stem", "320x320x64", "2x", 30, 120),
        ("S1", "160x160x128", "4x", 30, 220),
        ("S2", "80x80x256", "8x", 30, 320),
        ("S3", "40x40x512", "16x", 30, 420),
        ("S4", "20x20x512", "32x", 30, 520),
        ("SPP", "20x20x512", "32x", 30, 620),
    ]
    for name, res, stride, sx, sy in stages7:
        _draw_box(canvas7, sx, sy, sx + 100, sy + 70, COLOR_BLUE, name, fill=(50, 50, 50))
        canvas7 = _overlay_text(canvas7, res, (sx + 120, sy + 10), font_size=12, color=COLOR_ORANGE)
        canvas7 = _overlay_text(canvas7, f"stride {stride}", (sx + 120, sy + 38), font_size=12, color=COLOR_GREEN)
        if sy > 20:
            _draw_arrow(canvas7, sx + 50, sy - 5, sx + 50, sy, COLOR_WHITE)
    d = base / "07_full_flow_annotated"
    out = _overlay_step(canvas7, 7, "Complete backbone flow with resolution, channels, stride")
    _save(out, d / "diagram.jpg")

    return {"folders": ["backbone/01_overview_flowchart", "backbone/02_csp_stage_detail", "backbone/03_resolution_progression", "backbone/04_feature_map_evolution", "backbone/05_csp_vs_standard", "backbone/06_spp_multi_scale", "backbone/07_full_flow_annotated"]}



def process_backend():
    base = VIZ_BASE / "backend"

    # Step 1: API Detection Flow
    canvas1 = _blank_canvas((35, 35, 35))
    steps1 = [
        "User Upload", "FastAPI\nReceive", "Validate\nFile",
        "Save\nTemp", "YOLO\nInference", "Annotate\nResult",
        "Save\nOutput", "Return\nJSON"
    ]
    for i, label in enumerate(steps1):
        x = 60 + i * 90
        _draw_box(canvas1, x, 300, x + 70, 400, [(66, 133, 244), (52, 168, 83), (251, 188, 4), (234, 67, 53), (147, 112, 219), (0, 150, 136), (66, 133, 244), (52, 168, 83)][i], label, fill=(50, 50, 50), font_size=11)
        if i > 0:
            _draw_arrow(canvas1, x - 10, 350, x, 350, COLOR_WHITE, thickness=1)
    canvas1 = _overlay_text(canvas1, "routes/detect.py → services/detector.py", (150, 460), font_size=14, color=COLOR_ORANGE)
    d = base / "01_api_detection_flow"
    out = _overlay_step(canvas1, 1, "Full request-response flow: upload → inference → JSON")
    _save(out, d / "diagram.jpg")

    # Step 2: Inference Pipeline
    canvas2 = _blank_canvas((35, 35, 35))
    steps2 = [
        ("load_model()", "lazy singleton", 100, 100, COLOR_BLUE),
        ("_run_inference()", "YOLO predict", 100, 230, COLOR_GREEN),
        ("_draw_detections()", "plot masks+boxes", 100, 360, COLOR_ORANGE),
        ("_get_recommendation()", "category advice", 100, 490, COLOR_RED),
    ]
    for label, desc, x, y, color in steps2:
        _draw_box(canvas2, x, y, x + 200, y + 80, color, label, fill=(50, 50, 50))
        canvas2 = _overlay_text(canvas2, desc, (x + 220, y + 25), font_size=12, color=COLOR_LGRAY)
        if y > 100:
            _draw_arrow(canvas2, x + 100, y - 5, x + 100, y, COLOR_WHITE)
    canvas2 = _overlay_text(canvas2, "services/detector.py", (100, 600), font_size=16, color=COLOR_ORANGE)
    d = base / "02_inference_pipeline"
    out = _overlay_step(canvas2, 2, "YOLO inference internals: function call chain")
    _save(out, d / "diagram.jpg")

    # Step 3: Training Pipeline API
    canvas3 = _blank_canvas((35, 35, 35))
    steps3 = [
        ("POST /pipeline/\nrun-full", 200, 50, COLOR_BLUE),
        ("kaggle_cms.py", 200, 160, COLOR_GREEN),
        ("prepare_from_local()\nkaggle_service.py", 200, 270, COLOR_ORANGE),
        ("train_one()\ncli/train.py", 200, 380, COLOR_RED),
        ("validate()", 200, 490, COLOR_PURPLE),
        ("copy best.pt\nto models/", 200, 600, COLOR_TEAL),
    ]
    for label, x, y, color in steps3:
        _draw_box(canvas3, x, y, x + 200, y + 80, color, label, fill=(50, 50, 50), font_size=13)
        if y > 50:
            _draw_arrow(canvas3, x + 100, y - 5, x + 100, y, COLOR_WHITE)
    d = base / "03_training_pipeline_api"
    out = _overlay_step(canvas3, 3, "Full training pipeline: prepare → train → validate → copy")
    _save(out, d / "diagram.jpg")

    # Step 4: Backend Routes Overview
    canvas4 = _blank_canvas((35, 35, 35))
    routes = [
        ("/api/detect/*", "detect.py", 60, 80, COLOR_BLUE),
        ("/api/dataset/*", "dataset.py", 60, 200, COLOR_GREEN),
        ("/api/kaggle/*", "kaggle_cms.py", 60, 320, COLOR_ORANGE),
        ("/api/annotation/*", "annotation.py", 60, 440, COLOR_RED),
        ("/health", "main.py", 60, 560, COLOR_PURPLE),
    ]
    for route, fname, x, y, color in routes:
        _draw_box(canvas4, x, y, x + 180, y + 80, color, route, fill=(50, 50, 50))
        canvas4 = _overlay_text(canvas4, fname, (x + 200, y + 28), font_size=14, color=COLOR_LGRAY)
    d = base / "04_backend_routes_overview"
    out = _overlay_step(canvas4, 4, "All backend route groups with router file names")
    _save(out, d / "diagram.jpg")

    # Step 5: Detect Request-Response Schema
    canvas5 = _blank_canvas((35, 35, 35))
    _draw_box(canvas5, 50, 80, 350, 250, COLOR_BLUE, "HTTP Request\nPOST /api/detect/\nContent-Type: multipart\nfile: UploadFile", fill=(50, 50, 50))
    _draw_arrow(canvas5, 350, 165, 400, 165, COLOR_WHITE)
    _draw_box(canvas5, 400, 80, 750, 400, COLOR_GREEN, "Response JSON\nDetectResponse\n{\n  objects: [...],\n  fileType: str,\n  framesProcessed: int,\n  resultUrl: str,\n  recommendation: str\n}", fill=(50, 50, 50), font_size=11)
    canvas5 = _overlay_text(canvas5, "schemas/detection.py", (420, 430), font_size=14, color=COLOR_ORANGE)
    d = base / "05_detect_request_response"
    out = _overlay_step(canvas5, 5, "Request (multipart) → Response (DetectResponse schema)")
    _save(out, d / "diagram.jpg")

    # Step 6: Model Lifecycle
    canvas6 = _blank_canvas((35, 35, 35))
    ms = [
        ("Startup\n(lazy)", 100, 80, COLOR_GRAY),
        ("First Request\nload_model()", 100, 200, COLOR_BLUE),
        ("Inference\nYOLO predict", 100, 320, COLOR_GREEN),
        ("Reload\nPOST /reload", 100, 440, COLOR_ORANGE),
        ("Unload\n(cleanup)", 100, 560, COLOR_RED),
    ]
    for label, x, y, color in ms:
        _draw_box(canvas6, x, y, x + 200, y + 80, color, label, fill=(50, 50, 50))
        if y > 80:
            _draw_arrow(canvas6, x + 100, y - 5, x + 100, y, COLOR_WHITE)
    canvas6 = _overlay_text(canvas6, "singleton _model global", (320, 300), font_size=14, color=COLOR_ORANGE)
    d = base / "06_model_lifecycle"
    out = _overlay_step(canvas6, 6, "Model singleton lifecycle: lazy load → inference → reload")
    _save(out, d / "diagram.jpg")

    # Step 7: Error Handling Flow
    canvas7 = _blank_canvas((35, 35, 35))
    _draw_box(canvas7, 80, 100, 280, 200, COLOR_BLUE, "Request\nIncoming", fill=(50, 50, 50))
    _draw_box(canvas7, 80, 280, 280, 380, COLOR_RED, "Invalid file type?\n→ 400 Bad Request", fill=(50, 50, 50))
    _draw_box(canvas7, 330, 280, 530, 380, COLOR_RED, "Detection fail?\n→ 500 Server Error", fill=(50, 50, 50))
    _draw_box(canvas7, 330, 440, 530, 540, COLOR_RED, "Model not found?\n→ 500 Server Error", fill=(50, 50, 50))
    _draw_box(canvas7, 80, 440, 280, 540, COLOR_GREEN, "Success?\n→ 200 OK", fill=(50, 50, 50))
    _draw_arrow(canvas7, 180, 200, 180, 280, COLOR_WHITE)
    _draw_arrow(canvas7, 280, 340, 330, 340, COLOR_WHITE)
    canvas7 = _overlay_text(canvas7, "try/except chain", (180, 600), font_size=16, color=COLOR_ORANGE)
    d = base / "07_error_handling_flow"
    out = _overlay_step(canvas7, 7, "Error paths: file type → 400, detection → 500, model → 500")
    _save(out, d / "diagram.jpg")

    return {"folders": ["backend/01_api_detection_flow", "backend/02_inference_pipeline", "backend/03_training_pipeline_api", "backend/04_backend_routes_overview", "backend/05_detect_request_response", "backend/06_model_lifecycle", "backend/07_error_handling_flow"]}



def process_frontend():
    base = VIZ_BASE / "frontend"

    # Step 1: Sidebar Navigation
    canvas1 = _blank_canvas((240, 240, 240))
    cv2.rectangle(canvas1, (0, 0), (200, 800), (50, 50, 50), -1)
    items = ["Dashboard", "Training Eval", "Validation", "Test Results", "Inference"]
    active = 0
    for i, item in enumerate(items):
        y = 60 + i * 80
        bg = (66, 133, 244) if i == active else (60, 60, 60)
        cv2.rectangle(canvas1, (10, y), (190, y + 55), bg, -1)
        canvas1 = _overlay_text(canvas1, item, (20, y + 14), font_size=16, color=(255, 255, 255))
    canvas1 = _overlay_text(canvas1, "SmartBin", (40, 10), font_size=22, color=COLOR_ORANGE)
    canvas1 = _overlay_text(canvas1, "Active: Dashboard", (250, 30), font_size=18, color=COLOR_BLACK)
    d = base / "01_sidebar_navigation"
    out = _overlay_step(canvas1, 1, "Sidebar: Navigation with active state")
    _save(out, d / "wireframe.jpg")

    # Step 2: Detection UX Flow
    canvas2 = _blank_canvas((240, 240, 240))
    ux_steps = [
        ("Open /dashboard", 40, 60, COLOR_BLUE),
        ("Drag-drop file", 40, 160, COLOR_GREEN),
        ("Preview image", 40, 260, COLOR_ORANGE),
        ("Click 'Deteksi'", 40, 360, COLOR_RED),
        ("Loading state", 40, 460, COLOR_PURPLE),
        ("See result +\nrecommendation", 40, 560, COLOR_TEAL),
    ]
    for label, x, y, color in ux_steps:
        _draw_box(canvas2, x, y, x + 180, y + 70, color, label, fill=(220, 220, 220), font_size=13)
        if y > 60:
            _draw_arrow(canvas2, x + 90, y - 5, x + 90, y, (100, 100, 100), thickness=1)
    d = base / "02_detection_ux_flow"
    out = _overlay_step(canvas2, 2, "User journey: upload → detect → see recommendation")
    _save(out, d / "wireframe.jpg")

    # Step 3: Component Tree
    canvas3 = _blank_canvas((240, 240, 240))
    tree_nodes = [
        ("layouts/default.vue", 200, 30, COLOR_BLUE),
        ("pages/dashboard.vue", 60, 170, COLOR_GREEN),
        ("pages/test.vue", 340, 170, COLOR_GREEN),
        ("FileUpload.vue", 30, 310, COLOR_ORANGE),
        ("DetectionResult.vue", 180, 310, COLOR_ORANGE),
        ("BoundingBoxReport.vue", 330, 310, COLOR_ORANGE),
        ("ZoomModal.vue", 480, 310, COLOR_ORANGE),
    ]
    for label, x, y, color in tree_nodes:
        _draw_box(canvas3, x, y, x + 170, y + 60, color, label, fill=(220, 220, 220), font_size=12)
    _draw_arrow(canvas3, 285, 90, 145, 170, (100, 100, 100), thickness=1)
    _draw_arrow(canvas3, 285, 90, 425, 170, (100, 100, 100), thickness=1)
    _draw_arrow(canvas3, 145, 230, 115, 310, (100, 100, 100), thickness=1)
    _draw_arrow(canvas3, 145, 230, 265, 310, (100, 100, 100), thickness=1)
    _draw_arrow(canvas3, 425, 230, 415, 310, (100, 100, 100), thickness=1)
    _draw_arrow(canvas3, 425, 230, 565, 310, (100, 100, 100), thickness=1)
    d = base / "03_component_tree"
    out = _overlay_step(canvas3, 3, "Vue component hierarchy: layout → pages → components")
    _save(out, d / "tree.jpg")

    # Step 4: API Integration Flow
    canvas4 = _blank_canvas((240, 240, 240))
    _draw_box(canvas4, 50, 80, 220, 180, COLOR_BLUE, "Vue Component\n(UI)", fill=(200, 200, 200))
    _draw_box(canvas4, 290, 80, 460, 180, COLOR_GREEN, "$fetch / useFetch\n(API call)", fill=(200, 200, 200))
    _draw_box(canvas4, 530, 80, 700, 180, COLOR_ORANGE, "FastAPI\n(Backend)", fill=(200, 200, 200))
    _draw_box(canvas4, 290, 260, 460, 360, COLOR_RED, "Reactive State\n(update)", fill=(200, 200, 200))
    _draw_arrow(canvas4, 220, 130, 290, 130, (100, 100, 100))
    _draw_arrow(canvas4, 460, 130, 530, 130, (100, 100, 100))
    _draw_arrow(canvas4, 530, 160, 530, 300, (100, 100, 100))
    _draw_arrow(canvas4, 375, 260, 375, 300, (100, 100, 100))
    d = base / "04_api_integration_flow"
    out = _overlay_step(canvas4, 4, "Frontend-Backend: Component → $fetch → FastAPI → State")
    _save(out, d / "sequence.jpg")

    # Step 5: Batch Detect Flow
    canvas5 = _blank_canvas((240, 240, 240))
    batch_steps = [
        ("Select Multiple\nFiles", 40, 60, COLOR_BLUE),
        ("Preview Each\nImage", 40, 180, COLOR_GREEN),
        ("Click 'Deteksi\nSekarang'", 40, 300, COLOR_ORANGE),
        ("Sequential POST\n/api/detect/bulk", 40, 420, COLOR_RED),
        ("Results Grid\nDisplay", 40, 540, COLOR_PURPLE),
        ("Save to\nlocalStorage", 40, 660, COLOR_TEAL),
    ]
    for label, x, y, color in batch_steps:
        _draw_box(canvas5, x, y, x + 200, y + 80, color, label, fill=(220, 220, 220), font_size=12)
        if y > 60:
            _draw_arrow(canvas5, x + 100, y - 5, x + 100, y, (100, 100, 100))
    d = base / "05_batch_detect_flow"
    out = _overlay_step(canvas5, 5, "Multi-file upload: select → preview → sequential detect → grid")
    _save(out, d / "flowchart.jpg")

    # Step 6: Responsive Layout
    canvas6 = _blank_canvas((240, 240, 240))
    cv2.rectangle(canvas6, (0, 0), (64, 800), (50, 50, 50), -1)
    canvas6 = _overlay_text(canvas6, "Sidebar", (12, 400), font_size=13, color=(200, 200, 200))
    cv2.rectangle(canvas6, (64, 0), (800, 80), (60, 60, 60), -1)
    canvas6 = _overlay_text(canvas6, "Header (sticky, h-20)", (300, 25), font_size=16, color=(200, 200, 200))
    cv2.rectangle(canvas6, (64, 80), (800, 800), (80, 80, 80), -1)
    canvas6 = _overlay_text(canvas6, "Main Content (flex-1, p-6)", (300, 400), font_size=20, color=(200, 200, 200))
    d = base / "06_responsive_layout"
    out = _overlay_step(canvas6, 6, "Page layout: Sidebar (64px) + Header (h-20) + Main (flex-1)")
    _save(out, d / "wireframe.jpg")

    # Step 7: State Management
    canvas7 = _blank_canvas((240, 240, 240))
    _draw_box(canvas7, 40, 60, 360, 160, COLOR_BLUE, "dashboard.vue\nloading, error,\nresult, fileItem", fill=(220, 220, 220))
    _draw_box(canvas7, 420, 60, 740, 160, COLOR_GREEN, "test.vue\nfiles[], results[],\nerrors[], loading", fill=(220, 220, 220))
    _draw_box(canvas7, 40, 260, 360, 360, COLOR_ORANGE, "result.vue\nroute.query.data", fill=(220, 220, 220))
    _draw_box(canvas7, 420, 260, 740, 360, COLOR_PURPLE, "train-eval.vue\n(metrics data)", fill=(220, 220, 220))
    _draw_box(canvas7, 40, 460, 740, 560, COLOR_TEAL, "Template Rendering: v-if loading → spinner, v-else → results", fill=(220, 220, 220))
    _draw_arrow(canvas7, 200, 160, 200, 260, (100, 100, 100))
    _draw_arrow(canvas7, 580, 160, 580, 260, (100, 100, 100))
    _draw_arrow(canvas7, 390, 360, 390, 460, (100, 100, 100))
    d = base / "07_state_management"
    out = _overlay_step(canvas7, 7, "Reactive states per page: loading → results → template")
    _save(out, d / "diagram.jpg")

    return {"folders": ["frontend/01_sidebar_navigation", "frontend/02_detection_ux_flow", "frontend/03_component_tree", "frontend/04_api_integration_flow", "frontend/05_batch_detect_flow", "frontend/06_responsive_layout", "frontend/07_state_management"]}



def run_single(process_name, progress_callback=None):
    func = {
        "pseudo_mask": process_pseudo_mask,
        "stratified_split": process_stratified_split,
        "augmentation": process_augmentation,
        "backbone": process_backbone,
        "backend": process_backend,
        "frontend": process_frontend,
    }.get(process_name)
    if func is None:
        raise ValueError(f"Unknown process: {process_name}")
    if progress_callback:
        progress_callback({"step": process_name, "status": "start", "message": f"Generating {process_name} visualizations..."})
    try:
        r = func()
        if progress_callback:
            progress_callback({"step": process_name, "status": "done", "message": f"{process_name} visualizations generated"})
        return (process_name, {"success": True, "folders": r["folders"]})
    except Exception as e:
        if progress_callback:
            progress_callback({"step": process_name, "status": "error", "message": str(e)})
        return (process_name, {"success": False, "error": str(e)})


def run_all(progress_callback=None):
    results = {}
    processes = ["pseudo_mask", "stratified_split", "augmentation", "backbone", "backend", "frontend"]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(run_single, name, progress_callback): name for name in processes}
        for future in as_completed(futures):
            name, result = future.result()
            results[name] = result
    return results


if __name__ == "__main__":
    print("Generating pipeline visualizations...")
    r = run_all()
    for k, v in r.items():
        status = "OK" if v["success"] else "FAIL"
        print(f"  [{status}] {k}")
    print(f"Output: {VIZ_BASE}")
