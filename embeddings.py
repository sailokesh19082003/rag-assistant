import os
import logging
import numpy as np
from typing import List, Union
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

_model: SentenceTransformer = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (singleton)."""
    global _model
    if _model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _model


def generate_embedding(text: str) -> np.ndarray:
    """Generate a single embedding vector for a text string."""
    model = get_model()
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
    return embedding


def generate_embeddings_batch(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Generate embeddings for a batch of texts (more efficient)."""
    model = get_model()
    logger.info(f"Generating embeddings for {len(texts)} texts in batches of {batch_size}")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(texts) > 10,
    )
    logger.info(f"Generated {len(embeddings)} embeddings, dimension: {embeddings.shape[1]}")
    return embeddings
