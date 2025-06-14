import numpy as np
import faiss
import json
import os
from typing import List, Dict, Tuple


def save_embeddings_json(embedded_chunks: List[Dict], file_path: str):
    """
    Save list of embedded chunks with their vectors and metadata as JSON.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)
    print(f"✅ Saved {len(embedded_chunks)} embeddings to {file_path}")


def load_embeddings_json(file_path: str) -> Tuple[np.ndarray, List[Dict]]:
    """
    Load vector embeddings and metadata from JSON file.
    Returns:
        - matrix: Numpy 2D array of shape (n_chunks, vector_dim)
        - data: Original metadata per chunk
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ Embedding file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        raise ValueError("❌ Embedding file is empty or corrupted.")

    vectors = []
    for chunk in data:
        vector = np.array(chunk.get("vector", []), dtype="float32")
        if vector.size == 0:
            continue
        vectors.append(vector)

    if not vectors:
        raise ValueError("❌ No valid vectors found in embeddings.")

    matrix = np.stack(vectors)
    return matrix, data


def create_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create a FAISS index (Inner Product) with normalized vectors.
    """
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    print(f"🧠 Created FAISS index with {vectors.shape[0]} vectors of dim {vectors.shape[1]}")
    return index


def search_faiss(query_vector: np.ndarray, index: faiss.Index, metadata: List[Dict], top_k: int = 5) -> List[Dict]:
    """
    Perform semantic search with FAISS and return top-k metadata results.
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    faiss.normalize_L2(query_vector)

    D, I = index.search(query_vector, top_k)

    if D.shape[0] == 0 or I.shape[0] == 0:
        raise ValueError("❌ Search failed: FAISS returned empty results.")

    results = []
    for score, idx in zip(D[0], I[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

    return results
