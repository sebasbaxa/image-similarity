import argparse, os, json
from app.ingest import ingest
from app.embed_batch import embed_all, save_vectors

def main():
    ap = argparse.ArgumentParser(description="Ingest images and batch-embed to vectors.npy")
    ap.add_argument("--images", default="data/images", help="Folder with images to index")
    ap.add_argument("--thumbs", default="data/thumbs", help="Where to store thumbnails")
    ap.add_argument("--index",  default="data/index",  help="Where to store items.json & vectors.npy")
    ap.add_argument("--batch",  type=int, default=64, help="Batch size for embedding")
    args = ap.parse_args()

    print(f"[run] images={args.images}  thumbs={args.thumbs}  index={args.index}")
    items = ingest(args.images, args.thumbs, args.index)
    vectors = embed_all([i.__dict__ if hasattr(i, "__dict__") else i for i in items], batch_size=args.batch)
    save_vectors(args.index, vectors)

if __name__ == "__main__":
    main()