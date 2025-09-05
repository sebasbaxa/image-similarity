from __future__ import annotations
import os, json
from typing import List, Dict, Optional, Tuple, Callable

import numpy as np
from PIL import Image

from app.embed import Embedder
from app.index_faiss import FaissIndexer

Result = Dict[str, object]  # {rank, score, path, rel_path, thumb_path, width, height, tags}

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

class SearchEngine:
    """
    High-level search API:
      - loads index.faiss + items.json (+ vectors.npy optional)
      - embeds a query image
      - queries FAISS
      - optional exact re-rank on the shortlist using vectors.npy
    """
    def __init__(self, index_dir: str, emb_dim: int = 2048, load_vectors: bool = True):
        self.index_dir = index_dir
        self.embedder = Embedder()
        self.indexer = FaissIndexer(emb_dim)
        self.indexer.load(index_dir)

        self.items: List[Dict] = self.indexer.items
        self.vectors: Optional[np.ndarray] = None

        if load_vectors:
            vecs_path = os.path.join(index_dir, "vectors.npy")
            if os.path.exists(vecs_path):
                self.vectors = np.load(vecs_path).astype("float32")
            else:
                print(f"[search] Warning: {vecs_path} not found; re-rank disabled.")

    def _format_results(self, ids_scores: List[Tuple[Dict, float]], k: int) -> List[Result]:
        out: List[Result] = []
        for r, (item, score) in enumerate(ids_scores[:k], start=1):
            out.append({
                "rank": r,
                "score": float(score),  # cosine similarity in [-1,1]
                "path": item["path"],
                "rel_path": item.get("rel_path", ""),
                "thumb_path": item.get("thumb_path", ""),
                "width": item.get("width", None),
                "height": item.get("height", None),
                "tags": item.get("tags", []),
            })
        return out

    def _apply_filter(self, candidates: List[Tuple[Dict, float]],
                      predicate: Optional[Callable[[Dict], bool]]) -> List[Tuple[Dict, float]]:
        if predicate is None:
            return candidates
        return [(it, sc) for (it, sc) in candidates if predicate(it)]

    def _rerank_exact(self,
                      qvec: np.ndarray,
                      shortlist: List[Tuple[Dict, float]],
                      topk: int) -> List[Tuple[Dict, float]]:
        """Recompute exact cosine against stored vectors for the shortlist."""
        if self.vectors is None:
            return shortlist

        # map items -> their vector rows by id (we saved ids sequentially during ingest)
        idxs = []
        for item, _ in shortlist:
            # try id field first; else derive from position in items list
            if "id" in item:
                idxs.append(int(item["id"]))
            else:
                # fallback: use index lookup by path (O(n); acceptable for small shortlist)
                idxs.append(next(i for i, it in enumerate(self.items) if it["path"] == item["path"]))

        cand_mat = self.vectors[idxs]  # (M, D)
        # ensure normalized
        cand_mat = cand_mat / (np.linalg.norm(cand_mat, axis=1, keepdims=True) + 1e-12)
        q = qvec / (np.linalg.norm(qvec) + 1e-12)
        scores = cand_mat @ q  # cosine similarities

        rescored = [(shortlist[i][0], float(scores[i])) for i in range(len(shortlist))]
        rescored.sort(key=lambda t: t[1], reverse=True)
        return rescored[:topk]

    # ----- Public API -----

    def query_by_pil(self,
                     pil_img: Image.Image,
                     k: int = 10,
                     filter_pred: Optional[Callable[[Dict], bool]] = None,
                     rerank_exact: bool = True,
                     oversample: int = 2) -> List[Result]:
        """
        Query with a PIL image. Returns top-k formatted results.
        - oversample: how many to ask FAISS for before re-ranking (k * oversample)
        """
        qvec = self.embedder.embed_image(pil_img.convert("RGB"))
        return self._query_with_vector(qvec, k, filter_pred, rerank_exact, oversample)

    def query_by_path(self,
                      image_path: str,
                      k: int = 10,
                      filter_pred: Optional[Callable[[Dict], bool]] = None,
                      rerank_exact: bool = True,
                      oversample: int = 2) -> List[Result]:
        with Image.open(image_path) as im:
            im = im.convert("RGB")
            return self.query_by_pil(im, k=k, filter_pred=filter_pred,
                                     rerank_exact=rerank_exact, oversample=oversample)

    def _query_with_vector(self,
                           qvec: np.ndarray,
                           k: int,
                           filter_pred: Optional[Callable[[Dict], bool]],
                           rerank_exact: bool,
                           oversample: int) -> List[Result]:
        # ask FAISS for a bigger list, then filter & rerank
        shortlist = self.indexer.query(qvec, k=max(k * max(1, oversample), k))

        # optional metadata filtering (e.g., by tag)
        shortlist = self._apply_filter(shortlist, filter_pred)

        # optional exact re-ranking
        if rerank_exact and len(shortlist) > 0:
            shortlist = self._rerank_exact(qvec, shortlist, topk=k)

        # format
        return self._format_results(shortlist, k=k)

    # convenience filter helpers
    @staticmethod
    def tag_filter(required_tag: str) -> Callable[[Dict], bool]:
        rt = required_tag.lower()
        return lambda item: any(str(t).lower() == rt for t in item.get("tags", []))