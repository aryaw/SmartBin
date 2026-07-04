import json
import shutil
from pathlib import Path

import httpx
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, Response

from app.config import BASE_DIR, ORGANIC_CATEGORIES

router = APIRouter(prefix="/api/datasource", tags=["Datasource"])

DATASOURCE_DIR = BASE_DIR.parent / "datasource"
ANNOTATIONS_FILE = DATASOURCE_DIR / "annotations.json"
ANNOTATIONS_UNOFFICIAL_FILE = DATASOURCE_DIR / "annotations_unofficial.json"
CSV_FILE = DATASOURCE_DIR / "all_image_urls.csv"
CACHE_DIR = BASE_DIR / "static" / "datasource_cache"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _load_annotations(file_path: Path) -> dict:
    if not file_path.exists():
        return {"images": [], "annotations": [], "categories": []}
    return json.loads(file_path.read_text())


@router.get("/grid")
async def datasource_grid():
    data = _load_annotations(ANNOTATIONS_FILE)
    images = []
    for img in data.get("images", []):
        images.append({
            "id": img["id"],
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "url": img.get("flickr_url", img.get("coco_url", "")),
            "thumb_url": img.get("flickr_640_url", img.get("flickr_url", "")),
        })

    cat_map = {}
    for cat in data.get("categories", []):
        cat_map[cat["id"]] = cat["name"]

    ann_by_image = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        if img_id not in ann_by_image:
            ann_by_image[img_id] = []
        x, y, w, h = ann["bbox"]
        cat_name = cat_map.get(ann["category_id"], f"class_{ann['category_id']}")
        is_organik = ann["category_id"] in ORGANIC_CATEGORIES
        ann_by_image[img_id].append({
            "id": ann["id"],
            "category_id": ann["category_id"],
            "category": "Organik" if is_organik else "Non-Organik",
            "bbox": [round(v, 2) for v in [x, y, w, h]],
            "area": round(ann.get("area", w * h), 2),
        })

    return {
        "images": images,
        "annotations": ann_by_image,
        "categories": [{"id": k, "name": v} for k, v in cat_map.items()],
        "stats": {
            "total_images": len(images),
            "total_annotations": len(data.get("annotations", [])),
        },
    }


@router.get("/image")
async def datasource_image(url: str = Query(...)):
    stem = str(hash(url))
    cached = CACHE_DIR / f"{stem}.jpg"
    if cached.exists():
        return FileResponse(str(cached), media_type="image/jpeg")

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            content = resp.read()
            cached.write_bytes(content)
            return Response(content=content, media_type=resp.headers.get("content-type", "image/jpeg"))
    except Exception as e:
        raise HTTPException(502, f"Failed to fetch image: {str(e)}")


@router.get("/file/{filename}")
async def datasource_file(filename: str):
    file_path = DATASOURCE_DIR / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(str(file_path))


@router.get("/annotations")
async def get_annotations():
    data = _load_annotations(ANNOTATIONS_FILE)
    return data


@router.get("/annotations/unofficial")
async def get_annotations_unofficial():
    data = _load_annotations(ANNOTATIONS_UNOFFICIAL_FILE)
    return data
