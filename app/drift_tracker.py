import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics.pairwise import cosine_similarity


def load_vectors_from_json(path: str) -> Tuple[np.ndarray, List[Dict]]:
    import json
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    vectors = np.array([chunk["vector"] for chunk in data]).astype("float32")
    return vectors, data


def compute_drift_matrix(vectors_v1: np.ndarray, vectors_v2: np.ndarray) -> np.ndarray:
    """
    Returns a cosine similarity matrix between version1 and version2 vectors.
    """
    sim_matrix = cosine_similarity(vectors_v1, vectors_v2)
    return sim_matrix


def analyze_drift(
    sim_matrix: np.ndarray,
    meta_v1: List[Dict],
    meta_v2: List[Dict],
    threshold: float = 0.85
) -> List[Dict]:
    """
    For each v1 chunk, find its best match in v2 and compute drift.
    """
    drift_report = []
    for i, row in enumerate(sim_matrix):
        best_match_idx = np.argmax(row)
        score = row[best_match_idx]
        drift_score = round(1 - score, 4)

        drift_report.append({
            "chunk_index": meta_v1[i].get("chunk_index", i),
            "page_number": meta_v1[i].get("page_number", "?"),
            "chunk_text": meta_v1[i]["chunk_text"][:250],
            "drift_score": drift_score,
            "matched_page": meta_v2[best_match_idx].get("page_number", "?"),
            "matched_score": round(score, 4)
        })
    return sorted(drift_report, key=lambda x: x["drift_score"], reverse=True)
