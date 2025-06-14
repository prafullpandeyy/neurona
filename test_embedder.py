from app.parser import extract_text_from_pdf
from app.chunker import chunk_text
from app.embedder import (
    load_model,
    embed_chunks,
    save_embeddings_json,
    preview_embedding
)

# ---- Step 1: Parse PDF ----
pdf_path = "data/uploads/sample.pdf"
parsed = extract_text_from_pdf(pdf_path)
pages = parsed["pages"]
print(f"📄 Parsed {len(pages)} pages.")

# ---- Step 2: Chunk ----
chunks = chunk_text(pages, mode="paragraph", max_words=150, overlap=30)
print(f"🔗 Generated {len(chunks)} chunks.")

# ---- Step 3: Load Model ----
model = load_model("mpnet")  # 🔁 Changed from "local" to "mpnet" for better accuracy

# ---- Step 4: Embed Chunks ----
embedded_chunks = embed_chunks(chunks, model)

# ---- Step 5: Save ----
save_embeddings_json(embedded_chunks, "data/embeddings/sample_embeddings.json")

# ---- Step 6: Preview Output ----
preview_embedding(embedded_chunks[0])
