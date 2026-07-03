import json, os, sys, time
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ANNOTATIONS = "/home/arya/AI_Realm/SmartBin/datasource/annotations.json"
OUTDIR = "/home/arya/AI_Realm/SmartBin/backend/dataset/raw"

os.makedirs(OUTDIR, exist_ok=True)
data = json.load(open(ANNOTATIONS))

def download(img):
    iid = img["id"]
    fname = img["file_name"]
    flickr_url = img.get("flickr_url", "")

    if not flickr_url:
        return None, "no flickr_url"

    # Transform _o.png -> _z.jpg for smaller 640px version
    z_url = flickr_url.replace("_o.png", "_z.jpg")

    out_path = os.path.join(OUTDIR, fname.replace("/", "_"))
    if os.path.exists(out_path):
        return iid, "exists"

    for attempt in range(3):
        try:
            req = Request(z_url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=30)
            with open(out_path, "wb") as f:
                f.write(resp.read())
            return iid, "ok"
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return iid, f"fail: {e}"

images = data["images"]
annotated_ids = set(a["image_id"] for a in data["annotations"])
to_download = [i for i in images if i["id"] in annotated_ids]

print(f"Total annotated images: {len(to_download)}")
success = 0
failed = 0

with ThreadPoolExecutor(max_workers=12) as pool:
    futures = {pool.submit(download, img): img["id"] for img in to_download}
    for i, f in enumerate(as_completed(futures)):
        iid, status = f.result()
        if status == "ok":
            success += 1
        elif status == "exists":
            success += 1
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"Progress: {i+1}/{len(to_download)} (ok={success}, fail={failed})")

print(f"\nDone: {success} ok, {failed} failed")
