import os, json
import numpy as np
from typing import List, Dict
from PIL import Image
from app.embed import Embedder

def load_items(index_dir: str) -> List[Dict]:
    with open(os.path.join(index_dir, "items.json"), "r", encoding="utf-8") as f:
        return json.load(f)

def _open_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    return im.convert("RGB")

def embed_all(items: List[Dict], batch_size: int = 64) -> np.ndarray:
    emb = Embedder()
    D = 2048  # ResNet-50 penultimate layer
    N = len(items)
    vectors = np.zeros((N, D), dtype=np.float32)

    # simple batched loop (no dataloader to keep deps minimal)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_imgs = []
        paths = [items[i]["path"] for i in range(start, end)]
        for p in paths:
            try:
                batch_imgs.append(_open_rgb(p))
            except Exception as e:
                # On failure, insert a zero vector; you can also drop the item if preferred
                print(f"[embed] Failed to open {p}: {e}")
                batch_imgs.append(None)

        # embed each (kept simple; you can vectorize further later)
        for j, img in enumerate(batch_imgs):
            idx = start + j
            if img is None:
                continue
            v = emb.embed_image(img)  # already L2-normalized
            vectors[idx, :] = v.astype(np.float32)

        print(f"[embed] {end}/{N} embedded")

    # Optional: normalize again to be extra safe
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    vectors = vectors / norms
    return vectors

def save_vectors(index_dir: str, vectors: np.ndarray):
    os.makedirs(index_dir, exist_ok=True)
    out_path = os.path.join(index_dir, "vectors.npy")
    np.save(out_path, vectors)
    print(f"[embed] Saved vectors to {out_path} ({vectors.shape[0]} x {vectors.shape[1]})")