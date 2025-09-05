import os, json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from PIL import Image

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

@dataclass
class Item:
    id: int
    path: str          # absolute path
    rel_path: str      # path relative to images_root
    width: int
    height: int
    thumb_path: str    # absolute path to thumbnail
    tags: list

def _is_image(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_EXTS

def scan_images(images_root: str) -> List[str]:
    images = []
    for root, _, files in os.walk(images_root):
        for f in files:
            p = os.path.join(root, f)
            if _is_image(p):
                images.append(os.path.abspath(p))
    images.sort()
    return images

def _thumb_path_for(images_root: str, thumbs_root: str, abs_path: str) -> str:
    rel = os.path.relpath(abs_path, images_root)
    rel_no_ext = os.path.splitext(rel)[0] + ".jpg"
    out = os.path.join(thumbs_root, rel_no_ext)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    return out

def make_thumbnail(src_path: str, dst_path: str, max_size=(256, 256)) -> Tuple[int, int]:
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        w, h = im.size
        im.thumbnail(max_size)  # in-place, preserves aspect
        im.save(dst_path, format="JPEG", quality=85)
        return w, h

def ingest(images_root: str, thumbs_root: str, index_dir: str) -> List[Item]:
    os.makedirs(thumbs_root, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    abs_images = scan_images(images_root)
    items: List[Item] = []
    skipped = 0

    for i, p in enumerate(abs_images):
        try:
            tpath = _thumb_path_for(images_root, thumbs_root, p)
            w, h = make_thumbnail(p, tpath)
            items.append(Item(
                id=i,
                path=p,
                rel_path=os.path.relpath(p, images_root),
                width=w,
                height=h,
                thumb_path=tpath,
                tags=[]
            ))
        except Exception as e:
            skipped += 1
            print(f"[ingest] Skipping {p}: {e}")

    # save metadata
    items_json = [asdict(x) for x in items]
    with open(os.path.join(index_dir, "items.json"), "w", encoding="utf-8") as f:
        json.dump(items_json, f, ensure_ascii=False, indent=2)

    print(f"[ingest] Collected {len(items)} images ({skipped} skipped).")
    return items