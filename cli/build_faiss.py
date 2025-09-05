import os, json, argparse, numpy as np
from app.index_faiss import FaissIndexer

def main():
    ap = argparse.ArgumentParser(description="Build and save FAISS index from vectors.npy + items.json")
    ap.add_argument("--index", default="data/index", help="Directory with items.json & vectors.npy; output index.faiss saved here")
    ap.add_argument("--renorm", action="store_true", help="Re-normalize vectors before adding (recommended)")
    args = ap.parse_args()

    items_path = os.path.join(args.index, "items.json")
    vecs_path  = os.path.join(args.index, "vectors.npy")

    if not (os.path.exists(items_path) and os.path.exists(vecs_path)):
        raise SystemExit(f"Missing inputs. Expected {items_path} and {vecs_path}")

    items = json.load(open(items_path, "r", encoding="utf-8"))
    vecs  = np.load(vecs_path)

    dim = vecs.shape[1]
    ix = FaissIndexer(dim)
    ix.build(vecs, items, renorm=args.renorm)
    ix.save(args.index)

    print(f"[faiss] Built index for {vecs.shape[0]} vectors (dim={dim})")
    print(f"[faiss] Saved -> {os.path.join(args.index, 'index.faiss')}")

if __name__ == "__main__":
    main()