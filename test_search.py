from app.embedder import load_model, embed_query, preview_embedding
from app.vector_store import load_embeddings_json, create_faiss_index, search_faiss

# ---- CONFIGURATION ----
VECTOR_PATH = "data/embeddings/sample_embeddings.json"
MODEL_NAME = "mpnet"  # ✅ Match the model used during embedding
TOP_K = 5
# ------------------------

def main():
    print("📂 Loading embedded memory...")
    try:
        vectors, data = load_embeddings_json(VECTOR_PATH)
        index = create_faiss_index(vectors)
    except FileNotFoundError:
        print(f"❌ File not found: {VECTOR_PATH}")
        return
    except Exception as e:
        print(f"⚠️ Error loading vector store: {e}")
        return

    print("🧠 Loading embedding model...")
    model = load_model(MODEL_NAME)

    # 🔎 Accept user query
    query = input("🔎 Ask Neurona: ").strip()
    if not query:
        print("⚠️ Empty query. Exiting.")
        return

    print("\n🔍 Searching memory...\n")
    query_vector = embed_query(query, model)
    results = search_faiss(query_vector, index, data, top_k=TOP_K)

    if not results:
        print("⚠️ No relevant results found.")
        return

    print(f"\n🔍 Top {len(results)} results for query: “{query}”\n")
    for res in results:
        print(f"🧠 [Page {res['page_number']}] | Score: {round(res['score'], 4)}")
        print(f"Text Preview:\n{res['chunk_text'][:300]}...\n{'-'*60}")

if __name__ == "__main__":
    main()
