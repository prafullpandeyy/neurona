import fitz  # PyMuPDF
import os
from typing import Dict, List

def extract_text_from_pdf(file_path: str) -> Dict:
    """
    Extracts structured data from a PDF including:
    - Full text
    - Page-wise text
    - Metadata
    - Outline/bookmarks
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"[ERROR] File does not exist: {file_path}")
    
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        raise RuntimeError(f"[ERROR] Unable to open PDF: {e}")

    metadata = {
        "filename": os.path.basename(file_path),
        "page_count": doc.page_count,
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
        "created": doc.metadata.get("creationDate", ""),
        "mod_date": doc.metadata.get("modDate", ""),
    }

    # Extract text from each page
    pages_text: List[str] = []
    full_text = ""

    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text("text")
        cleaned_text = _clean_text(page_text)
        pages_text.append(cleaned_text)
        full_text += cleaned_text + "\n"

    # Optional: get outline/bookmarks
    try:
        outline = doc.get_toc(simple=True)  # Table of contents if present
    except Exception:
        outline = []

    doc.close()

    return {
        "metadata": metadata,
        "pages": pages_text,
        "full_text": full_text.strip(),
        "outline": outline
    }

def _clean_text(text: str) -> str:
    """ Basic cleaning: remove duplicate line breaks, unnecessary whitespace """
    lines = text.splitlines()
    cleaned = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned)
