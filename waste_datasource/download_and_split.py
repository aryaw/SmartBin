import csv, os, sys, time, json, shutil, random
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

random.seed(42)

CSV_FILE = "all_image_urls.csv"
RAW_DIR = Path("waste_raw_dataset")
SPLIT_DIR = Path("split")
SPLITS = {"train": 0.70, "test": 0.15, "inference": 0.15}

RAW_DIR.mkdir(parents=True, exist_ok=True)

# Read unique z.jpg URLs from CSV
urls = []
with open(CSV_FILE) as f:
    reader = csv.reader(f)
    seen = set()
    for row in reader:
        z_url = row[0].strip()
        if z_url and z_url not in seen:
            seen.add(z_url)
            urls.append(z_url)

print(f"Unique images to download: {len(urls)}")

def download(url):
    fname = url.split("/")[-1]
    out_path = RAW_DIR / fname
    if out_path.exists() and out_path.stat().st_size > 1000:
        return fname, "exists"
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=30)
            with open(out_path, "wb") as f:
                f.write(resp.read())
            return fname, "ok"
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                return fname, f"fail: {e}"

success = 0
failed = 0
with ThreadPoolExecutor(max_workers=12) as pool:
    futures = {pool.submit(download, url): url for url in urls}
    for i, f in enumerate(as_completed(futures)):
        fname, status = f.result()
        if status == "ok":
            success += 1
        elif status == "exists":
            success += 1
        else:
            failed += 1
        if (i + 1) % 200 == 0 or i == len(urls) - 1:
            print(f"Progress: {i+1}/{len(urls)} (ok={success}, fail={failed})")

print(f"\nDownload done: {success} ok, {failed} failed")

# Split into train/test/inference
valid_files = [f.name for f in sorted(RAW_DIR.iterdir()) if f.suffix.lower() in {".jpg", ".jpeg", ".png"} and f.stat().st_size > 1000]
print(f"\nValid images in RAW_DIR: {len(valid_files)}")
random.shuffle(valid_files)

n = len(valid_files)
train_end = int(n * SPLITS["train"])
test_end = train_end + int(n * SPLITS["test"])

for split_name, file_list in [
    ("train", valid_files[:train_end]),
    ("test", valid_files[train_end:test_end]),
    ("inference", valid_files[test_end:]),
]:
    split_dir = SPLIT_DIR / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    for fname in file_list:
        src = RAW_DIR / fname
        dst = split_dir / fname
        if not dst.exists():
            shutil.copy2(str(src), str(dst))
    print(f"  {split_name}: {len(file_list)} images -> {split_dir}/")
