import json, os, time, shutil, sys
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
ANNOTATIONS = BASE / "annotations.json"
RAW_DIR = BASE / "waste_raw_dataset"
OUT_ORG = BASE.parent / "backend" / "dataset" / "raw" / "Organik"
OUT_NON = BASE.parent / "backend" / "dataset" / "raw" / "Non-Organik"

for d in [RAW_DIR, OUT_ORG, OUT_NON]:
    d.mkdir(parents=True, exist_ok=True)

ORGANIC_CATEGORIES = {13, 14, 15, 16, 17, 18, 19, 20, 25, 30, 31, 32, 33, 34, 56}

data = json.load(open(ANNOTATIONS))
cat_map = {c["id"]: 0 if c["id"] in ORGANIC_CATEGORIES else 1 for c in data["categories"]}
img_info = {img["id"]: img for img in data["images"]}

img_anns = defaultdict(list)
for ann in data["annotations"]:
    img_anns[ann["image_id"]].append(ann)

def process(img_id):
    img = img_info[img_id]
    anns = img_anns.get(img_id, [])
    if not anns:
        return None, "no_anns"

    classes = {cat_map[a["category_id"]] for a in anns}
    target = OUT_ORG if 0 in classes else OUT_NON

    url = img.get("flickr_640_url")
    if not url:
        url = img.get("flickr_url", "").replace("_o.png", "_z.jpg")
    if not url:
        return None, "no_url"

    fname = url.split("/")[-1]
    raw_path = RAW_DIR / fname
    out_path = target / fname

    if out_path.exists():
        return fname, "exists"

    if not raw_path.exists() or raw_path.stat().st_size < 1000:
        ok = False
        for attempt in range(3):
            try:
                req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
                resp = urlopen(req, timeout=30)
                with open(raw_path, "wb") as f:
                    f.write(resp.read())
                ok = True
                break
            except HTTPError as e:
                if e.code == 404 and "_z.jpg" in url:
                    fallback = img.get("flickr_url", "")
                    if fallback and fallback != url:
                        url = fallback
                        continue
                return None, f"http_{e.code}"
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                else:
                    return None, str(e)[:60]
        if not ok:
            return None, "download_fail"

    shutil.copy2(str(raw_path), str(out_path))
    return fname, "ok"

annotated_ids = list(img_anns.keys())
print(f"Total annotated images: {len(annotated_ids)}")

ok = exists = fail = 0
with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(process, iid): iid for iid in annotated_ids}
    for i, f in enumerate(as_completed(futures)):
        fname, status = f.result()
        if status == "ok":
            ok += 1
        elif status == "exists":
            exists += 1
        else:
            fail += 1
        if (i + 1) % 200 == 0 or i == len(annotated_ids) - 1:
            print(f"Progress: {i+1}/{len(annotated_ids)} (ok={ok}, exists={exists}, fail={fail})")

org_count = len(list(OUT_ORG.iterdir()))
non_count = len(list(OUT_NON.iterdir()))
print(f"\nDone! Organik={org_count}, Non-Organik={non_count}, Fail={fail}")
