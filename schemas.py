from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class ChatRequest(BaseModel):
    sessionId: str = Field(..., min_length=1, description="Unique session identifier")
    message: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    reply: str
    tokensUsed: Optional[int] = None
    retrievedChunks: int = 0
    similarityScores: Optional[List[float]] = None


class HealthResponse(BaseModel):
    status: str
    vectorstore_loaded: bool
    total_chunks: int
    timestamp: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


class DocumentChunk(BaseModel):
    chunk_id: str
    title: str
    content: str
    source: str
    chunk_index: int
