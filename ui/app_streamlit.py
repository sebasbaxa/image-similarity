# ui/app_streamlit.py
from __future__ import annotations
import os
import sys
import json
import time
import hashlib
from typing import List, Dict, Optional

from PIL import Image
import io
import streamlit as st

# Ensure project root is on sys.path so `from app...` works when running from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.search import SearchEngine  # noqa: E402


# ---------- Caching ----------

@st.cache_resource(show_spinner=False)
def load_engine(index_dir: str, emb_dim: int = 2048, load_vectors: bool = True) -> SearchEngine:
    return SearchEngine(index_dir=index_dir, emb_dim=emb_dim, load_vectors=load_vectors)

@st.cache_data(show_spinner=False)
def load_items(index_dir: str) -> List[Dict]:
    p = os.path.join(index_dir, "items.json")
    if not os.path.exists(p):
        return []
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- Session state ----------

def _init_state():
    ss = st.session_state
    ss.setdefault("results", None)
    ss.setdefault("last_query_sig", None)
    ss.setdefault("query_source", "Upload")       # "Upload" or "Dataset"
    ss.setdefault("upload_bytes", None)           # bytes of last uploaded file
    ss.setdefault("upload_name", None)            # display name
    ss.setdefault("dataset_choice", None)         # last selected dataset rel_path
_init_state()


def sig_from_bytes(b: bytes) -> str:
    return "upload:" + hashlib.sha1(b).hexdigest()

def sig_from_path(path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    return "path:" + hashlib.sha1(f"{path}|{mtime}".encode()).hexdigest()


# ---------- UI ----------

st.set_page_config(page_title="Image Similarity Search", layout="wide")
st.title("Image Similarity (FAISS + ResNet)")

with st.sidebar:
    st.subheader("Settings")
    index_dir = st.text_input("Index directory", value="data/index")
    topk = st.slider("Top-K results", min_value=1, max_value=24, value=8)
    use_rerank = st.checkbox("Exact re-rank shortlist", value=True, help="Recompute cosine on top candidates using vectors.npy")
    load_vecs = st.checkbox("Load vectors.npy (enables re-rank)", value=True)

    st.markdown("---")
    source_mode = st.radio("Query source", ["Upload", "Dataset"], index=(0 if st.session_state.query_source == "Upload" else 1), horizontal=True)
    st.session_state.query_source = source_mode

    st.caption("Tip: Put images under `data/images/`, then run ingestion + embedding + FAISS build before using the UI.")

# Load engine/items (cached)
engine: Optional[SearchEngine] = None
items: List[Dict] = []
engine_load_err = None
try:
    engine = load_engine(index_dir, emb_dim=2048, load_vectors=load_vecs)
    items = load_items(index_dir)
except Exception as e:
    engine_load_err = str(e)

query_img: Optional[Image.Image] = None
query_label: str = ""
query_sig: str = ""

# ----------------- Query builders (only the chosen source runs) -----------------

if st.session_state.query_source == "Upload":
    st.subheader("⬆️ Upload image")

    uploader_key = f"uploader::{index_dir}"
    file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp", "bmp", "tiff"],
        key=uploader_key,
    )
    col_u1, col_u2 = st.columns([1,1])
    with col_u1:
        clear_upload = st.button("Clear upload")
    if clear_upload:
        st.session_state.upload_bytes = None
        st.session_state.upload_name = None
        st.session_state.results = None
        st.session_state.last_query_sig = None
        st.rerun()

    # If a new file is uploaded, stash its bytes & name in session state
    if file is not None:
        try:
            data = file.getvalue()
            if data and data != st.session_state.upload_bytes:
                st.session_state.upload_bytes = data
                st.session_state.upload_name = getattr(file, "name", "uploaded_image")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    # Build query from session bytes (robust across reruns)
    if st.session_state.upload_bytes:
        try:
            query_img = Image.open(io.BytesIO(st.session_state.upload_bytes)).convert("RGB")
            query_label = st.session_state.upload_name or "uploaded_image"
            query_sig = sig_from_bytes(st.session_state.upload_bytes)
            st.image(query_img, caption=f"Query: {query_label}", use_container_width=False)
        except Exception as e:
            st.error(f"Failed to open uploaded image: {e}")
    else:
        st.info("Upload an image to search.")

else:
    st.subheader("📁 Pick from dataset")

    if not items:
        st.info("No items.json found or it is empty. Run ingestion first.")
    else:
        rel_paths = [it.get("rel_path") or it.get("path") for it in items]
        default_idx = rel_paths.index(st.session_state.dataset_choice) if st.session_state.dataset_choice in rel_paths else 0
        choice = st.selectbox("Select an image from your indexed dataset", options=rel_paths, index=default_idx, key="picker")
        if choice:
            st.session_state.dataset_choice = choice
            it = next((i for i in items if (i.get("rel_path") or i.get("path")) == choice), None)
            if it:
                try:
                    img_path = it["path"]
                    _img = Image.open(img_path).convert("RGB")
                    _thumb = it.get("thumb_path")

                    query_img = _img
                    query_label = it.get("rel_path") or it["path"]
                    query_sig = sig_from_path(img_path)

                    st.image(_thumb if (_thumb and os.path.exists(_thumb)) else _img,
                             caption=f"Query: {query_label}", use_container_width=False)
                except Exception as e:
                    st.error(f"Failed to open dataset image: {e}")

# ----------------- Controls & execution -----------------

disabled = (engine is None) or (query_img is None)
if engine_load_err:
    st.error(f"Failed to load index: {engine_load_err}")

left, right = st.columns([1, 5])
with left:
    run_clicked = st.button("Search", type="primary", disabled=disabled)
with right:
    auto_run_on_change = st.checkbox("Auto-run when query changes", value=True)

# Clear old results if the query changed
if query_sig and query_sig != st.session_state.last_query_sig:
    st.session_state.results = None

# Recompute condition: clicked OR (query changed and auto-run)
should_run = run_clicked or (auto_run_on_change and query_sig and query_sig != st.session_state.last_query_sig)

# Run search if needed
if should_run and engine and query_img is not None:
    t0 = time.time()
    results = engine.query_by_pil(
        query_img,
        k=topk,
        filter_pred=None,
        rerank_exact=(use_rerank and (engine.vectors is not None)),
        oversample=2,
    )
    dt_ms = (time.time() - t0) * 1000.0

    st.session_state.results = {"items": results, "latency_ms": dt_ms, "label": query_label}
    st.session_state.last_query_sig = query_sig

# Render results (persisted)
res = st.session_state.results
if res and res.get("items"):
    st.subheader(f"Results (Top-{topk}) • {res['latency_ms']:.1f} ms")
    results = res["items"]

    cols_per_row = 4 if len(results) >= 8 else min(3, len(results))
    for i in range(0, len(results), cols_per_row):
        cols = st.columns(cols_per_row, gap="small")
        for c, r in zip(cols, results[i : i + cols_per_row]):
            with c:
                img_src = (
                    r.get("thumb_path")
                    if r.get("thumb_path") and os.path.exists(r["thumb_path"])
                    else r["path"]
                )
                try:
                    st.image(img_src, use_container_width=True)
                except Exception:
                    try:
                        st.image(Image.open(r["path"]).convert("RGB"), use_container_width=True)
                    except Exception as e:
                        st.write(f"⚠️ Unable to display image: {e}")

                cap = (r.get("rel_path") or os.path.basename(r["path"]))
                score = r.get("score", 0.0)
                st.markdown(f"**{cap}**")
                st.caption(f"cosine: {score:.4f}")

    with st.expander("Show raw results JSON"):
        st.json(results)
elif engine and query_img is not None:
    st.info("Click **Search** to run, or enable **Auto-run when query changes**.")
