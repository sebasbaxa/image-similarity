# Image Similarity Search

A tool for building and searching image similarity using FAISS indexing and ResNet embeddings. This project provides a complete pipeline from image ingestion to interactive visual search through a Streamlit UI.

## Features

- Image ingestion and thumbnail generation
- ResNet-based image embedding
- FAISS indexing for fast similarity search
- Interactive web UI for visual search
- Support for both uploaded images and dataset browsing

## Usage

### Quick Start

1. Place your images in the `data/images` directory
2. Run the complete pipeline:
```bash
python run_app.py
```
This will:
- Process your images
- Generate embeddings
- Build the FAISS index
- Launch the web UI

### Advanced Usage

Run with custom options:
```bash
python run_app.py --images "path/to/images" --thumbs "path/to/thumbs" --index "path/to/index" --batch 32 --renorm
```

Available options:
- `--images`: Input images directory (default: "data/images")
- `--thumbs`: Thumbnail storage directory (default: "data/thumbs")
- `--index`: Index files directory (default: "data/index")
- `--batch`: Batch size for embedding (default: 64)
- `--renorm`: Enable vector renormalization (recommended)
- `--skip-pipeline`: Skip processing and launch UI directly
- `--help`: Show help message

### Running UI Only

If you've already built your index and just want to launch the UI:
```bash
python run_app.py --skip-pipeline
```

## Directory Structure

```
.
├── data/
│   ├── images/     # Source images
│   ├── thumbs/     # Generated thumbnails
│   └── index/      # Index files
├── ui/             # UI components
├── app/            # Core application code
└── run_app.py      # Main entry point
```

## Web Interface

The web UI provides:
- Image upload functionality
- Dataset browsing
- Top-K similar image results
- Adjustable search parameters
- Result visualization

## Tips

1. For best results, ensure your images are in common formats (jpg, png, webp)
2. The `--renorm` flag is recommended for better search accuracy
3. Adjust batch size based on your available memory
4. Use `--skip-pipeline` to avoid reprocessing when just viewing results

## Requirements

- Python 3.7+
- FAISS
- PyTorch
- Streamlit
- PIL
- NumPy