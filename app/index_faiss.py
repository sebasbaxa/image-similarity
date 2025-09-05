import os, json
import numpy as np
import faiss

class FaissIndexer:
    def __init__(self, dim: int):
        # Inner product on L2-normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.items = []

    def build(self, vectors: np.ndarray, items: list, renorm: bool = True):
        assert vectors.ndim == 2, f"vectors must be 2D, got {vectors.shape}"
        assert len(items) == vectors.shape[0], "items and vectors length mismatch"
        vecs = vectors.astype("float32")

        # Be defensive: (re)normalize rows so IP == cosine
        if renorm:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms

        self.index.add(vecs)
        self.items = items

    def save(self, index_dir: str):
        os.makedirs(index_dir, exist_ok=True)
        faiss.write_index(self.index, os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "items.json"), "w", encoding="utf-8") as f:
            json.dump(self.items, f, ensure_ascii=False, indent=2)

    def load(self, index_dir: str):
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        with open(os.path.join(index_dir, "items.json"), "r", encoding="utf-8") as f:
            self.items = json.load(f)

    def query(self, vec: np.ndarray, k: int = 10):
        q = vec.astype("float32")
        q /= (np.linalg.norm(q) + 1e-12)
        D, I = self.index.search(q.reshape(1, -1), k)
        # returns list of (item_dict, score) with score=cosine similarity in [-1,1]
        return [(self.items[i], float(s)) for i, s in zip(I[0], D[0])]