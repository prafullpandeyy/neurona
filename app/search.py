from app.vector_store import (
    load_embeddings_json,
    create_faiss_index,
    search_faiss
)
from app.embedder import load_model, embed_query
from typing import List, Dict, Optional
import os


class NeuronaSearchEngine:
    def __init__(
        self,
        model_name: str = "mpnet",
        embedding_path: str = "data/embeddings/sample_embeddings.json"
    ):
        """
        Initializes the search engine with given model and vector store.
        """
        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")
        
        print("📂 Loading embedded vectors...")
        self.vectors, self.metadata = load_embeddings_json(embedding_path)
        self.index = create_faiss_index(self.vectors)

        print(f"🧠 Loading model '{model_name}' for search...")
        self.model = load_model(model_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[Dict]:
        """
        Perform semantic search. Optionally apply a metadata filter function.
        """
        if not query.strip():
            raise ValueError("Query must not be empty.")

        print(f"🔍 Searching top {top_k} matches for: “{query}”")
        query_vector = embed_query(query, self.model)

        results = search_faiss(query_vector, self.index, self.metadata, top_k=top_k)

        if filter_fn:
            results = list(filter(filter_fn, results))

        return results

    def explain_result(self, result: Dict, show_meta: bool = False) -> str:
        """
        Create a human-readable summary of a result block.
        """
        preview = result["chunk_text"][:300].replace("\n", " ").strip()
        page = result.get("page_number", "?")
        score = round(result.get("score", 0), 4)
        meta = result.get("meta", {})

        summary = f"🧠 Page {page} | Score: {score}\nText:\n{preview}..."

        if show_meta and meta:
            summary += f"\nMetadata: {meta}"

        return summary

    def summarize_all(self, results: List[Dict], word_limit: int = 40) -> List[str]:
        """
        Return short previews (for UI list mode).
        """
        return [
            f"[{r['page_number']}] {r['chunk_text'][:word_limit*6]}..."
            for r in results
        ]

    def get_model_name(self) -> str:
        return self.model.__class__.__name__

    def get_index_info(self) -> str:
        return f"FAISS Index: {self.index.ntotal} vectors, dim={self.index.d}"
