from app.vector_store import (
    load_embeddings_json,
    create_faiss_index,
    embed_query,
    search_faiss
)
from app.embedder import load_model
import numpy as np

# ---- CONFIG ----
EMBED_PATH = "data/embeddings/sample_embeddings.json"
MODEL_NAME = "mpnet"
TOP_K = 3
TEST_QUERY = "What is LegitReach?"
# ----------------

def main():
    print("🔍 [Test] Vector Store Validation\n")

    # Step 1: Load
    vectors, metadata = load_embeddings_json(EMBED_PATH)
    print(f"✅ Loaded {len(vectors)} vectors from: {EMBED_PATH}")
    
    # Step 2: Check vector shape
    assert isinstance(vectors, np.ndarray), "Vectors must be numpy array"
    assert vectors.ndim == 2, "Vectors must be 2D"
    print(f"✅ Vectors shape: {vectors.shape}")

    # Step 3: Build FAISS index
    index = create_faiss_index(vectors)
    print(f"✅ FAISS index created with dim = {index.d}")

    # Step 4: Load model
    model = load_model(MODEL_NAME)
    query_vector = embed_query(TEST_QUERY, model)

    # Step 5: Run search
    results = search_faiss(query_vector, index, metadata, top_k=TOP_K)
    assert results, "No results returned!"

    # Step 6: Display
    print(f"\n🔍 Search results for: \"{TEST_QUERY}\"")
    for i, r in enumerate(results, start=1):
        print(f"\n[{i}] Score: {round(r['score'], 4)} | Page: {r['page_number']}")
        print(f"Text: {r['chunk_text'][:250]}...\n{'-'*50}")

    print("\n✅ Vector store is functional.\n")

if __name__ == "__main__":
    main()
