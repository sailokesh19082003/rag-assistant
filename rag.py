import logging
from typing import Dict, Any

from app.services.embeddings import generate_embedding
from app.services.llm import generate_response
from app.services.conversation import (
    add_message,
    format_history_for_prompt,
)
from app.vectorstore.store import get_vectorstore
from app.prompts.templates import (
    SYSTEM_PROMPT,
    build_rag_prompt,
    FALLBACK_RESPONSE,
)

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = float(__import__("os").getenv("SIMILARITY_THRESHOLD", "0.3"))


def rag_query(session_id: str, user_message: str) -> Dict[str, Any]:
    """
    Full RAG pipeline:
    1. Generate query embedding
    2. Similarity search in vector store
    3. Apply threshold filter
    4. Build grounded prompt with context + history
    5. Call LLM
    6. Store conversation turn
    7. Return structured result
    """
    logger.info(f"[RAG] session={session_id} query='{user_message[:80]}...'")

    # Step 1: Generate query embedding
    query_embedding = generate_embedding(user_message)

    # Step 2: Similarity search
    vectorstore = get_vectorstore()
    retrieved = vectorstore.search(
        query_embedding,
        threshold=SIMILARITY_THRESHOLD,
    )

    similarity_scores = [r["similarity_score"] for r in retrieved]
    logger.info(f"[RAG] Retrieved {len(retrieved)} chunks, scores: {[round(s, 4) for s in similarity_scores]}")

    # Step 3: Threshold check — use fallback if no relevant chunks found
    if not retrieved:
        logger.warning(f"[RAG] No chunks above threshold={SIMILARITY_THRESHOLD} — returning fallback")
        add_message(session_id, "user", user_message)
        add_message(session_id, "assistant", FALLBACK_RESPONSE)
        return {
            "reply": FALLBACK_RESPONSE,
            "tokensUsed": None,
            "retrievedChunks": 0,
            "similarityScores": [],
        }

    # Step 4: Build context from retrieved chunks
    context_parts = []
    for r in retrieved:
        context_parts.append(
            f"[Source: {r['source']}]\n{r['content']}"
        )
    retrieved_context = "\n\n".join(context_parts)

    # Step 5: Get conversation history
    history_str = format_history_for_prompt(session_id)

    # Step 6: Build RAG prompt
    prompt = build_rag_prompt(
        retrieved_context=retrieved_context,
        conversation_history=history_str,
        user_question=user_message,
    )

    logger.debug(f"[RAG] Prompt length: {len(prompt)} chars")

    # Step 7: Call LLM
    reply, tokens_used = generate_response(prompt=prompt, system_prompt=SYSTEM_PROMPT)

    # Step 8: Store conversation turn
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", reply)

    return {
        "reply": reply,
        "tokensUsed": tokens_used,
        "retrievedChunks": len(retrieved),
        "similarityScores": similarity_scores,
    }
