import json
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


def load_documents(docs_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSON file."""
    with open(docs_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents from {docs_path}")
    return documents


def chunk_text(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Chunk text into overlapping segments by word count.
    Uses word-boundary splitting to preserve context.
    """
    words = text.split()
    if len(words) <= max_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + max_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start += max_size - overlap

    return chunks


def chunk_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process all documents into chunks with metadata.
    Returns a flat list of chunk dicts.
    """
    chunks = []
    for doc_idx, doc in enumerate(documents):
        title = doc.get("title", f"Document {doc_idx}")
        content = doc.get("content", "")

        text_chunks = chunk_text(content)
        for chunk_idx, chunk_text_content in enumerate(text_chunks):
            chunk_id = f"{doc_idx}_{chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "title": title,
                "content": chunk_text_content,
                "source": title,
                "chunk_index": chunk_idx,
                "total_chunks": len(text_chunks),
            })

    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
