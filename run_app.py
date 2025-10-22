import argparse
import os
import subprocess
import sys
from app.ingest import ingest
from app.embed_batch import embed_all, save_vectors
from app.index_faiss import FaissIndexer
import streamlit.web.bootstrap

def build_embeddings(images_dir, thumbs_dir, index_dir, batch_size=64):
    print(f"[run] images={images_dir}  thumbs={thumbs_dir}  index={index_dir}")
    items = ingest(images_dir, thumbs_dir, index_dir)
    vectors = embed_all([i.__dict__ if hasattr(i, "__dict__") else i for i in items], batch_size=batch_size)
    save_vectors(index_dir, vectors)
    return vectors

def build_faiss(index_dir, vectors, renorm=True):
    items_path = os.path.join(index_dir, "items.json")
    if not os.path.exists(items_path):
        raise SystemExit(f"Missing items.json at {items_path}")

    dim = vectors.shape[1]
    ix = FaissIndexer(dim)
    ix.build(vectors, items_path, renorm=renorm)
    ix.save(index_dir)

    print(f"[faiss] Built index for {vectors.shape[0]} vectors (dim={dim})")
    print(f"[faiss] Saved -> {os.path.join(index_dir, 'index.faiss')}")

def main():
    ap = argparse.ArgumentParser(description="Run complete image similarity pipeline and UI")
    ap.add_argument("--images", default="data/images", help="Folder with images to index")
    ap.add_argument("--thumbs", default="data/thumbs", help="Where to store thumbnails")
    ap.add_argument("--index", default="data/index", help="Where to store index files")
    ap.add_argument("--batch", type=int, default=64, help="Batch size for embedding")
    ap.add_argument("--renorm", action="store_true", help="Re-normalize vectors before adding to FAISS (recommended)")
    ap.add_argument("--skip-pipeline", action="store_true", help="Skip ingestion/embedding/indexing and just launch UI")
    args = ap.parse_args()

    # Create directories if they don't exist
    os.makedirs(args.thumbs, exist_ok=True)
    os.makedirs(args.index, exist_ok=True)

    if not args.skip_pipeline:
        print("=== Starting Pipeline ===")
        print("1. Building embeddings...")
        vectors = build_embeddings(args.images, args.thumbs, args.index, args.batch)
        
        print("\n2. Building FAISS index...")
        build_faiss(args.index, vectors, args.renorm)
        print("\n=== Pipeline Complete ===\n")

    print("3. Launching Streamlit UI...")
    # Get the absolute path to app_streamlit.py
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "app_streamlit.py")
    
    # Launch Streamlit
    sys.argv = ["streamlit", "run", ui_path]
    streamlit.web.bootstrap.run(ui_path, "", [], [])

if __name__ == "__main__":
    main()