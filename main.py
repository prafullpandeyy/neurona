from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.embedder import load_model, embed_chunks, save_embeddings_json
from app.vector_store import (
    load_embeddings_json,
    create_faiss_index,
    search_faiss,
)
from app.search import NeuronaSearchEngine
import os
import uuid
import shutil

# ------------------------------------
# ✅ App setup
# ------------------------------------
app = FastAPI(title="Neurona Backend")

# ✅ Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------
# ✅ Global config
# ------------------------------------
UPLOAD_DIR = "data/uploads"
EMBED_PATH = "data/embeddings/sample_embeddings.json"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(EMBED_PATH), exist_ok=True)

# ✅ Load model once
global_model = load_model("mpnet")


# ------------------------------------
# 📌 Route: /embed
# ------------------------------------
@app.post("/embed")
async def embed_pdf(file: UploadFile = File(...)):
    try:
        # Save uploaded PDF
        file_id = str(uuid.uuid4())
        save_path = os.path.join(UPLOAD_DIR, f"{file_id}.pdf")
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Parse & chunk
        parsed = extract_text_from_pdf(save_path)
        pages = parsed["pages"]
        if not pages:
            raise HTTPException(status_code=400, detail="No text extracted from PDF.")
        chunks = chunk_text(pages, mode="paragraph", max_words=150, overlap=30)

        # Embed and save
        embedded_chunks = embed_chunks(chunks, global_model)
        save_embeddings_json(embedded_chunks, EMBED_PATH)

        return {
            "message": f"✅ File embedded successfully. {len(embedded_chunks)} chunks saved."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed failed: {str(e)}")


# ------------------------------------
# 📌 Route: /search
# ------------------------------------
@app.post("/search")
async def search(query: dict):
    try:
        user_query = query.get("query")
        if not user_query:
            raise HTTPException(status_code=400, detail="Query missing.")

        engine = NeuronaSearchEngine(model_name="mpnet", embedding_path=EMBED_PATH)
        engine.model = global_model  # reuse loaded model

        results = engine.search(user_query, top_k=5)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
