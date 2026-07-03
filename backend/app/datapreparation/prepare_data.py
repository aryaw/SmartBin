import json, os, shutil, random
from pathlib import Path
from collections import defaultdict

random.seed(42)

RAW_IMAGES = "/home/arya/AI_Realm/SmartBin/backend/dataset/raw"
ANNOTATIONS = "/home/arya/AI_Realm/SmartBin/datasource/annotations.json"
DATASET_DIR = "/home/arya/AI_Realm/SmartBin/backend/dataset"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# TACO category → SmartBin class mapping
# Organik (0): only Food waste
# Non-Organik (1): everything else
ORGANIK_IDS = {25}  # Food waste

data = json.load(open(ANNOTATIONS))

# Map category_id → class_id (0=Organik, 1=Non-Organik)
cat_map = {}
for cat in data["categories"]:
    if cat["id"] in ORGANIK_IDS:
        cat_map[cat["id"]] = 0
    else:
        cat_map[cat["id"]] = 1

# Build image info
images = {img["id"]: img for img in data["images"]}

# Group annotations by image_id
img_anns = defaultdict(list)
for ann in data["annotations"]:
    img_id = ann["image_id"]
    img_anns[img_id].append(ann)

# Collect all annotated image IDs
annotated_ids = list(img_anns.keys())
random.shuffle(annotated_ids)

# Split
n = len(annotated_ids)
train_end = int(n * SPLITS["train"])
val_end = train_end + int(n * SPLITS["val"])

train_ids = annotated_ids[:train_end]
val_ids = annotated_ids[train_end:val_end]
test_ids = annotated_ids[val_end:]

print(f"Total annotated images: {n}")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")

def process_split(img_ids, split_name):
    img_dir = Path(DATASET_DIR) / split_name / "images"
    label_dir = Path(DATASET_DIR) / split_name / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    class_counts = defaultdict(int)
    for img_id in img_ids:
        img = images[img_id]
        src_name = img["file_name"].replace("/", "_")
        src_path = Path(RAW_IMAGES) / src_name
        if not src_path.exists():
            print(f"  WARN: {src_path} not found")
            continue

        # Copy image
        dst_img = img_dir / src_name
        shutil.copy2(str(src_path), str(dst_img))

        # Write YOLO label file (handle .JPG, .jpg, .png, .PNG)
        label_name = Path(src_name).stem + ".txt"
        label_path = label_dir / label_name
        h, w = img["height"], img["width"]
        with open(label_path, "w") as f:
            for ann in img_anns[img_id]:
                cls_id = cat_map[ann["category_id"]]
                class_counts[cls_id] += 1
                bbox = ann["bbox"]  # [x, y, width, height] COCO format
                x, y, bw, bh = bbox
                # Normalize
                x_center = (x + bw / 2) / w
                y_center = (y + bh / 2) / h
                bw_norm = bw / w
                bh_norm = bh / h
                f.write(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

    return class_counts

print("\nProcessing splits...")
for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
    counts = process_split(ids, split_name)
    print(f"  {split_name}: {len(ids)} images, Organik={counts.get(0,0)}, Non-Organik={counts.get(1,0)}")

# Count total images per class across all splits
total_org = sum(1 for ann in data["annotations"] if cat_map[ann["category_id"]] == 0)
total_non = sum(1 for ann in data["annotations"] if cat_map[ann["category_id"]] == 1)
print(f"\nTotal annotations: Organik={total_org}, Non-Organik={total_non}")

print("\nDone!")
