import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
TOP_K = int(os.getenv("TOP_K_CHUNKS", "3"))


class InMemoryVectorStore:
    """
    In-memory vector store using cosine similarity for retrieval.
    Stores chunk metadata alongside normalized embedding vectors.
    """

    def __init__(self):
        self._chunks: List[Dict[str, Any]] = []
        self._embeddings: np.ndarray = None  # shape: (N, D)

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: np.ndarray):
        """
        Store chunks and their pre-computed embeddings.
        embeddings must be row-normalized (unit vectors) for cosine similarity.
        """
        self._chunks = chunks
        self._embeddings = embeddings
        logger.info(f"VectorStore: stored {len(chunks)} chunks, embedding dim={embeddings.shape[1]}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = TOP_K,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        """
        Perform cosine similarity search.
        Returns top_k chunks with similarity score above threshold.
        """
        if self._embeddings is None or len(self._chunks) == 0:
            logger.warning("VectorStore is empty — no documents indexed")
            return []

        # query_embedding shape: (D,) → reshape to (1, D)
        query_vec = query_embedding.reshape(1, -1)

        # Cosine similarity: dot product of unit vectors = cos(θ)
        scores = cosine_similarity(query_vec, self._embeddings)[0]  # shape: (N,)

        # Log similarity scores for debugging
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        logger.debug(
            "Top similarity scores: %s",
            [(self._chunks[i]["title"], round(float(scores[i]), 4)) for i in top_indices],
        )

        # Filter by threshold and take top_k
        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score >= threshold:
                result = dict(self._chunks[idx])
                result["similarity_score"] = score
                results.append(result)
            if len(results) >= top_k:
                break

        logger.info(
            f"Search returned {len(results)} chunks (threshold={threshold}, top_k={top_k})"
        )
        return results

    @property
    def total_chunks(self) -> int:
        return len(self._chunks)

    @property
    def is_loaded(self) -> bool:
        return self._embeddings is not None and len(self._chunks) > 0


# Singleton instance
_vectorstore = InMemoryVectorStore()


def get_vectorstore() -> InMemoryVectorStore:
    return _vectorstore
