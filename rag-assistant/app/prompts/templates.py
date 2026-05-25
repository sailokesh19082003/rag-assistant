SYSTEM_PROMPT = """You are a helpful and precise customer support assistant.

Your job is to answer user questions using ONLY the context provided below.
Do not use any prior knowledge or make up information.
If the context does not contain enough information to answer the question, say so honestly.
Be concise, friendly, and accurate. Format your answer clearly."""


def build_rag_prompt(
    retrieved_context: str,
    conversation_history: str,
    user_question: str,
) -> str:
    """
    Build the RAG prompt that grounds LLM responses in retrieved context.

    Structure:
    1. Retrieved context (primary source of truth)
    2. Conversation history (for follow-up understanding)
    3. User question
    """
    prompt = f"""Use ONLY the information in the Context below to answer the Question.
If the context does not contain the answer, respond with:
"I could not find enough information in the knowledge base to answer this question."

---

Context:
{retrieved_context}

---

Conversation History:
{conversation_history if conversation_history else "No previous messages."}

---

Question: {user_question}

Answer:"""
    return prompt


def build_fallback_prompt(conversation_history: str, user_question: str) -> str:
    """Prompt used when no relevant context is found."""
    return f"""Conversation History:
{conversation_history if conversation_history else "No previous messages."}

Question: {user_question}

The knowledge base does not contain relevant information for this question.
Politely inform the user that you cannot find an answer in the available knowledge base
and suggest they contact support."""


FALLBACK_RESPONSE = (
    "I could not find enough information in the knowledge base to answer this question. "
    "Please try rephrasing your question, or contact our support team at support@example.com for further assistance."
)
