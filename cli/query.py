import argparse, os
from app.search import SearchEngine

def main():
    ap = argparse.ArgumentParser(description="Query FAISS index with an image path")
    ap.add_argument("--index", default="data/index", help="Directory with index.faiss, items.json, vectors.npy")
    ap.add_argument("--img", required=True, help="Path to query image")
    ap.add_argument("--topk", type=int, default=5, help="How many results to return")
    ap.add_argument("--tag", default=None, help="Filter results to items having this exact tag")
    ap.add_argument("--no-rerank", action="store_true", help="Disable exact re-ranking on shortlist")
    ap.add_argument("--no-vectors", action="store_true", help="Do not load vectors.npy (forces no re-rank)")
    args = ap.parse_args()

    if not os.path.exists(args.img):
        raise SystemExit(f"[query] Image not found: {args.img}")

    se = SearchEngine(index_dir=args.index, load_vectors=not args.no_vectors)

    pred = None
    if args.tag:
        pred = se.tag_filter(args.tag)

    results = se.query_by_path(
        image_path=args.img,
        k=args.topk,
        filter_pred=pred,
        rerank_exact=(not args.no_rerank),
        oversample=2,
    )

    print(f"\nTop-{args.topk} for {args.img} (index: {args.index}):\n")
    if not results:
        print("No results.")
        return

    for r in results:
        print(f"{r['rank']:>2}. score={r['score']:.4f}  "
              f"{r['rel_path'] or r['path']}")
        if r.get("thumb_path"):
            print(f"    thumb: {r['thumb_path']}")
        if r.get("tags"):
            print(f"    tags: {', '.join(map(str, r['tags']))}")
    print("")

if __name__ == "__main__":
    main()