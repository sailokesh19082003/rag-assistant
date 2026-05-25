import os
import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Startup: Index documents ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
DOCS_PATH = BASE_DIR / "docs.json"


def index_documents():
    """Load docs → chunk → embed → store in vector store."""
    from app.utils.chunker import load_documents, chunk_documents
    from app.services.embeddings import generate_embeddings_batch
    from app.vectorstore.store import get_vectorstore

    logger.info("=== Starting document indexing ===")
    docs = load_documents(str(DOCS_PATH))
    chunks = chunk_documents(docs)

    texts = [c["content"] for c in chunks]
    embeddings = generate_embeddings_batch(texts)

    vs = get_vectorstore()
    vs.add_chunks(chunks, embeddings)
    logger.info(f"=== Indexing complete: {vs.total_chunks} chunks ready ===")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run indexing at startup."""
    index_documents()
    yield
    logger.info("Shutting down RAG assistant")


# ── FastAPI App ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="GenAI RAG Assistant",
    description="Production-grade RAG-powered chat assistant using Claude + Sentence Transformers",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend static files
frontend_path = BASE_DIR / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the chat UI."""
    index_file = frontend_path / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return JSONResponse({"message": "RAG Assistant API is running. See /docs for API documentation."})


# Register API routes
from app.routes.chat import router
app.include_router(router)


# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
