import re
import nltk
from typing import List, Dict, Literal
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)

def chunk_text(
    pages: List[str],
    mode: Literal["paragraph", "sentence", "token"] = "paragraph",
    max_words: int = 150,
    overlap: int = 30
) -> List[Dict]:
    """
    Converts page-wise text into clean, context-preserving chunks.
    
    Args:
        pages: List of page texts.
        mode: How to split ('paragraph', 'sentence', 'token').
        max_words: Maximum words per chunk.
        overlap: Number of overlapping words between chunks.

    Returns:
        List of dicts with: chunk_text, page_number, chunk_index
    """

    chunks = []
    chunk_id = 0

    for page_number, page_text in enumerate(pages):
        if not page_text.strip():
            continue

        units = _split_text(page_text, mode)
        merged_chunks = _merge_units(units, max_words, overlap)

        for chunk in merged_chunks:
            chunks.append({
                "chunk_index": chunk_id,
                "page_number": page_number + 1,
                "chunk_text": chunk.strip()
            })
            chunk_id += 1

    return chunks


def _split_text(text: str, mode: str) -> List[str]:
    """Split text into units based on mode"""
    if mode == "paragraph":
        return re.split(r'\n{2,}', text)
    elif mode == "sentence":
        return sent_tokenize(text)
    elif mode == "token":
        return re.findall(r'\S+', text)  # list of tokens
    else:
        raise ValueError(f"Invalid mode: {mode}")


def _merge_units(units: List[str], max_words: int, overlap: int) -> List[str]:
    """
    Merge units (paragraphs/sentences/tokens) into overlapping chunks.
    """
    chunks = []
    current_chunk = []

    words = 0
    i = 0
    while i < len(units):
        unit = units[i]
        unit_word_count = len(unit.split())

        # If unit is too large, break it directly
        if unit_word_count > max_words:
            unit_words = unit.split()
            for j in range(0, unit_word_count, max_words - overlap):
                sub_chunk = " ".join(unit_words[j: j + max_words])
                chunks.append(sub_chunk)
            i += 1
            continue

        # Normal merge
        if words + unit_word_count <= max_words:
            current_chunk.append(unit)
            words += unit_word_count
            i += 1
        else:
            chunks.append(" ".join(current_chunk))
            # Start new chunk with overlap
            overlap_words = " ".join(current_chunk).split()[-overlap:]
            current_chunk = [" ".join(overlap_words)]
            words = len(overlap_words)

    # Final flush
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks
