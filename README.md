# Image Similarity Search
This project is a content-based image search engine.
It uses a pretrained ResNet-50 model to embed images into vectors, stores them in a FAISS index, and provides both a CLI and a Streamlit UI for searching by example.

# Usage
1. Place images under data/images
2. ingest and embed images: ```python -m cli.build_embeddings --images data/images --thumbs data/thumbs --index data/index```
3. Build FIASS index: ```python -m cli.build_faiss --index data/index --renorm```
4. Launch Web UI: ```python -m streamlit run ui/app_streamlit.py```
