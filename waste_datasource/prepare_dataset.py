import json, os, sys, time, shutil
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

# --- Config ---
ANNOTATIONS_FILE = "annotations_unofficial.json"
RAW_DIR = Path("waste_raw_dataset")
OUT_ORG = Path("../backend/dataset/raw/Organik")
OUT_NON = Path("../backend/dataset/raw/Non-Organik")

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_ORG.mkdir(parents=True, exist_ok=True)
OUT_NON.mkdir(parents=True, exist_ok=True)

# Organic TACO categories (from backend config)
ORGANIC_CATEGORIES = {
    13, 14, 15, 16, 17, 18, 19, 20,  # cartons, pizza box, paper cup
    25,  # Food waste
    30, 31, 32, 33, 34,  # magazine paper, tissues, wrapping paper, normal paper, paper bag
    56,  # Paper straw
}

data = json.load(open(ANNOTATIONS_FILE))

# Map category_id -> class (0=Organik, 1=Non-Organik)
cat_map = {}
for cat in data["categories"]:
    cat_map[cat["id"]] = 0 if cat["id"] in ORGANIC_CATEGORIES else 1

# Per-image annotation summary
img_anns = defaultdict(list)
for ann in data["annotations"]:
    img_anns[ann["image_id"]].append(ann)

# Classify each image
img_info = {img["id"]: img for img in data["images"]}

org_count = 0
non_count = 0
mixed_count = 0
skip_count = 0

def classify_and_download(img_id):
    global org_count, non_count, mixed_count, skip_count
    img = img_info[img_id]
    anns = img_anns.get(img_id, [])
    if not anns:
        skip_count += 1
        return None

    classes = set(cat_map[a["category_id"]] for a in anns)
    is_org = 0 in classes
    is_non = 1 in classes

    if is_org and is_non:
        target_dir = OUT_ORG  # put in organik if mixed (majority rule not needed)
        mixed_count += 1
    elif is_org:
        target_dir = OUT_ORG
        org_count += 1
    else:
        target_dir = OUT_NON
        non_count += 1

    # Determine download URL (prefer flickr_640_url, fall back to flickr_url)
    url = img.get("flickr_640_url") or img.get("flickr_url")
    if not url:
        skip_count += 1
        return None

    fname = url.split("/")[-1]
    raw_path = RAW_DIR / fname
    out_path = target_dir / fname

    if out_path.exists():
        return fname

    # Download if not in raw cache
    if not raw_path.exists() or raw_path.stat().st_size < 1000:
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urlopen(req, timeout=30)
                with open(raw_path, "wb") as f:
                    f.write(resp.read())
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    print(f"  FAIL {fname}: {e}")
                    return None

    # Copy to target
    shutil.copy2(str(raw_path), str(out_path))
    return fname

# Process all annotated images
annotated_ids = list(img_anns.keys())
print(f"Annotated images: {len(annotated_ids)}")

success = 0
failed = 0
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = {pool.submit(classify_and_download, iid): iid for iid in annotated_ids}
    for i, f in enumerate(as_completed(futures)):
        result = f.result()
        if result:
            success += 1
        else:
            failed += 1
        if (i + 1) % 200 == 0 or i == len(annotated_ids) - 1:
            print(f"Progress: {i+1}/{len(annotated_ids)} (ok={success}, fail={failed})")

print(f"\nDone!")
print(f"  Organik images: {org_count + mixed_count} ({org_count} pure + {mixed_count} mixed)")
print(f"  Non-Organik images: {non_count}")
print(f"  Skipped (no annotations): {skip_count}")
print(f"  Download failed: {failed}")
print(f"\nOutput:")
print(f"  Organik -> {OUT_ORG}/ ({len(list(OUT_ORG.iterdir()))} files)")
print(f"  Non-Organik -> {OUT_NON}/ ({len(list(OUT_NON.iterdir()))} files)")
