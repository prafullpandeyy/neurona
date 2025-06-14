from typing import List, Dict
import numpy as np
from tqdm import tqdm
import faiss
import os
import json
import hashlib
from sentence_transformers import SentenceTransformer

# ✅ Extended model registry with high-performance transformer options
MODEL_REGISTRY = {
    "local": "all-MiniLM-L6-v2",
    "base": "paraphrase-MiniLM-L12-v2",
    "distil": "distiluse-base-multilingual-cased-v1",
    "mpnet": "all-mpnet-base-v2"  # ✅ Recommended for semantic search
}

_loaded_models = {}  # model cache


def load_model(name: str = "local") -> SentenceTransformer:
    """
    Load (and cache) a sentence-transformer model by name.
    """
    if name in _loaded_models:
        return _loaded_models[name]

    model_name = MODEL_REGISTRY.get(name, MODEL_REGISTRY["local"])
    print(f"🧠 Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    _loaded_models[name] = model
    return model


def embed_chunks(
    chunks: List[Dict],
    model: SentenceTransformer,
    text_key: str = "chunk_text",
    normalize: bool = True
) -> List[Dict]:
    """
    Generate and normalize embeddings for document chunks.
    """
    texts = [chunk[text_key] for chunk in chunks]
    print(f"🔢 Embedding {len(texts)} chunks...")

    vectors = model.encode(texts, show_progress_bar=True, batch_size=32)
    vectors = np.array(vectors).astype("float32")

    if normalize:
        faiss.normalize_L2(vectors)

    embedded = []
    for i, chunk in enumerate(chunks):
        embedded.append({
            "chunk_index": chunk.get("chunk_index", i),
            "page_number": chunk.get("page_number", -1),
            "chunk_id": _chunk_id(chunk[text_key]),
            "chunk_text": chunk[text_key],
            "vector": vectors[i].tolist(),
            "meta": chunk.get("meta", {})
        })

    return embedded


def embed_query(query: str, model: SentenceTransformer) -> np.ndarray:
    """
    Embed a user query and return a normalized 2D numpy vector for FAISS.
    """
    vector = model.encode([query])[0]
    vector = np.array(vector).astype("float32").reshape(1, -1)
    faiss.normalize_L2(vector)
    return vector


def save_embeddings_json(embedded_chunks: List[Dict], file_path: str):
    """
    Save embedded vectors + metadata as JSON for persistent retrieval.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)
    print(f"✅ Saved {len(embedded_chunks)} embeddings to {file_path}")


def preview_embedding(chunk: Dict):
    """
    Print a short preview of an embedded chunk.
    """
    print(f"\n🧠 [Page {chunk['page_number']}] Chunk {chunk['chunk_index']}")
    print(f"Text Preview: {chunk['chunk_text'][:300]}")
    print(f"Vector (first 5 dims): {chunk['vector'][:5]}")


def _chunk_id(text: str) -> str:
    """
    Generate a stable unique ID from the text for chunk tracking.
    """
    return hashlib.md5(text.encode()).hexdigest()
