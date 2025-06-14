from app.parser import extract_text_from_pdf
from app.chunker import chunk_text

file_path = "data/uploads/sample.pdf"

# Step 1: Parse
data = extract_text_from_pdf(file_path)
pages = data["pages"]

# Step 2: Chunk
chunks = chunk_text(pages, mode="paragraph", max_words=150, overlap=30)

print(f"✅ Extracted {len(chunks)} chunks.\n")

# Show preview
for chunk in chunks[:3]:
    print(f"[Page {chunk['page_number']}] Chunk {chunk['chunk_index']}:\n{chunk['chunk_text'][:300]}...\n")
