import logging
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, ErrorResponse
from app.services.rag import rag_query
from app.services.conversation import clear_session
from app.vectorstore.store import get_vectorstore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint to verify the service is running and indexed."""
    vs = get_vectorstore()
    return HealthResponse(
        status="healthy",
        vectorstore_loaded=vs.is_loaded,
        total_chunks=vs.total_chunks,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@router.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    RAG-powered chat endpoint.
    Validates input → retrieves context → calls LLM → returns grounded response.
    """
    # Validate
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message field is required and cannot be empty")
    if not request.sessionId.strip():
        raise HTTPException(status_code=400, detail="sessionId field is required")

    logger.info(f"Chat request: session={request.sessionId}, msg_len={len(request.message)}")

    try:
        result = rag_query(
            session_id=request.sessionId,
            user_message=request.message.strip(),
        )
        return ChatResponse(
            reply=result["reply"],
            tokensUsed=result.get("tokensUsed"),
            retrievedChunks=result["retrievedChunks"],
            similarityScores=result.get("similarityScores"),
        )

    except PermissionError as e:
        logger.error(f"API key error: {e}")
        raise HTTPException(status_code=401, detail=str(e))

    except TimeoutError as e:
        logger.error(f"Timeout: {e}")
        raise HTTPException(status_code=504, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        raise HTTPException(status_code=503, detail=str(e))

    except Exception as e:
        logger.exception(f"Unexpected error in /api/chat: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@router.delete("/api/session/{session_id}", tags=["Chat"])
async def delete_session(session_id: str):
    """Clear conversation history for a session (new chat)."""
    clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
