# 🤖 GenAI RAG Assistant

A production-grade, Retrieval-Augmented Generation (RAG) powered chat assistant built with FastAPI, Sentence Transformers, and Claude (Anthropic).

---

## 📋 Table of Contents

- [Architecture Overview](#architecture-overview)
- [RAG Workflow](#rag-workflow)
- [Embedding Strategy](#embedding-strategy)
- [Similarity Search](#similarity-search)
- [Prompt Design](#prompt-design)
- [Tech Stack](#tech-stack)
- [Setup Instructions](#setup-instructions)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Deployment](#deployment)

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG ASSISTANT                            │
│                                                                 │
│  ┌──────────────┐     ┌──────────────────────────────────────┐ │
│  │   Frontend   │     │           FastAPI Backend            │ │
│  │  (HTML/JS)   │◄───►│                                      │ │
│  └──────────────┘     │  ┌─────────┐  ┌──────────────────┐  │ │
│                        │  │  /health│  │    /api/chat     │  │ │
│                        │  └─────────┘  └────────┬─────────┘  │ │
│                        │                        │             │ │
│                        │            ┌───────────▼──────────┐  │ │
│                        │            │    RAG Service        │  │ │
│                        │            │  ┌─────────────────┐  │  │ │
│                        │            │  │ 1. Embed query  │  │  │ │
│                        │            │  │ 2. Vector search│  │  │ │
│                        │            │  │ 3. Build prompt │  │  │ │
│                        │            │  │ 4. Call LLM     │  │  │ │
│                        │            │  └─────────────────┘  │  │ │
│                        │            └──┬──────────┬──────────┘  │ │
│                        │               │          │             │ │
│  ┌─────────────────┐   │  ┌────────────▼──┐  ┌───▼──────────┐ │ │
│  │   docs.json     │──►│  │ Vector Store  │  │  LLM API     │ │ │
│  │  (Knowledge     │   │  │ (In-Memory    │  │  (Claude /   │ │ │
│  │   Base)         │   │  │  cosine sim.) │  │   OpenAI)    │ │ │
│  └─────────────────┘   │  └───────────────┘  └──────────────┘ │ │
│                        └──────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 RAG Workflow

### Indexing Phase (at startup)

```
docs.json
    │
    ▼
Load Documents (10 docs)
    │
    ▼
Chunk Documents (300-400 words per chunk, 50-word overlap)
    │
    ▼
Generate Embeddings (Sentence Transformers: all-MiniLM-L6-v2)
    │                 → 384-dimensional unit vectors
    ▼
Store in InMemoryVectorStore
    │                 → chunks[] + embeddings matrix (N×384)
    ▼
Ready to serve requests ✓
```

### Query Phase (per user message)

```
User Question
    │
    ▼
Generate Query Embedding (same model, same 384-dim space)
    │
    ▼
Cosine Similarity Search against all stored vectors
    │   scores = cosine_similarity([query_vec], embeddings_matrix)
    ▼
Rank and Filter (top-K=3, threshold=0.3)
    │
    ├─── Score < 0.3 → Fallback: "I could not find enough information..."
    │
    └─── Score ≥ 0.3 → Build RAG Prompt
                            │
                            ▼
                    Inject: Context + History + Question
                            │
                            ▼
                    Call LLM (Claude / OpenAI)
                            │
                            ▼
                    Grounded Response → User
```

---

## 🧮 Embedding Strategy

**Model**: `sentence-transformers/all-MiniLM-L6-v2`

- Produces **384-dimensional** dense vectors
- Pre-trained on 1B+ sentence pairs for semantic similarity
- Runs **locally** — no embedding API key or cost
- Embeddings are **L2-normalized** (unit vectors), making cosine similarity equivalent to dot product

**Why this model?**
- Fast inference (<50ms per query on CPU)
- Strong semantic understanding for Q&A tasks
- No API dependency — works offline
- Proven performance on BEIR benchmark

**Chunking strategy**:
- Max 400 words per chunk with 50-word overlap
- Overlap preserves context at chunk boundaries
- Metadata stored with each chunk: `title`, `source`, `chunk_id`, `chunk_index`

---

## 📐 Similarity Search

**Algorithm**: Cosine Similarity using `scikit-learn`

```python
from sklearn.metrics.pairwise import cosine_similarity

scores = cosine_similarity([query_vector], embeddings_matrix)[0]
# Returns array of shape (N,) with values in [-1, 1]
# Since vectors are unit-normalized: range is [0, 1]
# 0 = completely unrelated, 1 = identical semantics
```

**Why cosine similarity?**
- Scale-invariant: direction matters, not magnitude
- Ideal for semantic embeddings that are already normalized
- Efficient to compute: O(N·D) for N documents of dimension D

**Threshold**: `0.3` (configurable via `SIMILARITY_THRESHOLD` env var)
- Scores above 0.3 = semantically relevant → use as context
- Scores below 0.3 = insufficient signal → return fallback response
- This prevents hallucination when the knowledge base lacks relevant info

**Similarity scores are logged** on every query for transparency and debugging.

---

## ✍️ Prompt Design

The RAG prompt follows a strict structure that keeps the LLM grounded:

```
SYSTEM: You are a helpful and precise customer support assistant.
        Answer using ONLY the provided context.
        If context is insufficient, say so honestly.

USER:
  Context:
  [Source: Reset Password]
  Users can reset their password by navigating to Settings > Security...

  [Source: Data Privacy]
  We take data privacy seriously...

  ---

  Conversation History:
  User: Hello
  Assistant: Hi! How can I help?

  ---

  Question: How do I change my password?

  Answer:
```

**Design Reasoning**:
1. **Context first** — LLM attention focuses on retrieved info before seeing the question
2. **Source labels** — tell the model where each chunk came from
3. **History included** — enables follow-up questions ("what about that setting?")
4. **Explicit constraint** — "ONLY the context" reduces hallucination
5. **Fallback instruction** — explicit instructions for insufficient context
6. **Temperature = 0.2** — low temperature for factual, consistent answers

---

## 🛠 Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| LLM | Claude (claude-sonnet-4-20250514) or OpenAI GPT |
| Embeddings | Sentence Transformers (all-MiniLM-L6-v2) |
| Vector Store | In-Memory (NumPy matrix) |
| Similarity | Cosine Similarity (scikit-learn) |
| Frontend | HTML + CSS + Vanilla JS |
| Session Store | In-Memory (Python dict) |
| Config | python-dotenv |

---

## ⚙️ Setup Instructions

### Prerequisites

- Python 3.10+
- Git
- An Anthropic API key (or OpenAI key)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/rag-assistant.git
cd rag-assistant
```

### 2. Create virtual environment

```bash
python -m venv venv

# macOS/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Note: First run will download the `all-MiniLM-L6-v2` model (~90MB) automatically.

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
ANTHROPIC_API_KEY=sk-ant-...your-key-here...
LLM_PROVIDER=anthropic
```

### 5. Run the server

```bash
# From the project root
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open the app

Navigate to: **http://localhost:8000**

API documentation: **http://localhost:8000/docs**

---

## 📡 API Reference

### POST /api/chat

Send a message and get a RAG-grounded response.

**Request**:
```json
{
  "sessionId": "sess_abc123",
  "message": "How do I reset my password?"
}
```

**Response**:
```json
{
  "reply": "You can reset your password by going to Settings > Security > Change Password...",
  "tokensUsed": 312,
  "retrievedChunks": 2,
  "similarityScores": [0.847, 0.623]
}
```

**Error Response**:
```json
{
  "error": "Message field is required",
  "detail": "message cannot be empty"
}
```

---

### GET /health

Check service status and vector store state.

**Response**:
```json
{
  "status": "healthy",
  "vectorstore_loaded": true,
  "total_chunks": 12,
  "timestamp": "2025-01-01T12:00:00Z"
}
```

---

### DELETE /api/session/{session_id}

Clear conversation history (new chat).

---

## 📁 Project Structure

```
rag-assistant/
│
├── app/
│   ├── main.py                  # FastAPI app, startup indexing, CORS
│   ├── routes/
│   │   └── chat.py              # /api/chat and /health endpoints
│   ├── services/
│   │   ├── rag.py               # Core RAG orchestration pipeline
│   │   ├── embeddings.py        # Sentence transformer embedding service
│   │   ├── llm.py               # LLM API integration (Claude / OpenAI)
│   │   └── conversation.py      # Session-based history management
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response models
│   ├── vectorstore/
│   │   └── store.py             # In-memory vector store + cosine search
│   ├── prompts/
│   │   └── templates.py         # System prompt + RAG prompt builder
│   └── utils/
│       └── chunker.py           # Document loading + chunking
│
├── frontend/
│   ├── index.html               # Chat UI markup
│   ├── styles.css               # Dark-mode design system
│   └── app.js                   # Session mgmt, fetch, rendering
│
├── docs.json                    # Knowledge base (10 documents)
├── requirements.txt             # Python dependencies
├── .env.example                 # Environment variable template
└── README.md                    # This file
```

---

## 🚀 Deployment

### Render (Recommended — Free Tier)

1. Push your repository to GitHub
2. Go to [render.com](https://render.com) and create a **New Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables in the Render dashboard (from your `.env`)
6. Deploy

### Railway

```bash
# Install Railway CLI
npm install -g @railway/cli

railway login
railway init
railway up
```

Add environment variables via the Railway dashboard.

### Docker (optional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t rag-assistant .
docker run -p 8000:8000 --env-file .env rag-assistant
```

---

## 🔑 Key Design Decisions

| Decision | Rationale |
|---|---|
| Local embeddings (no API) | No cost, no rate limits, fast, reproducible |
| In-memory vector store | Simple, zero dependencies, sufficient for 10-50 docs |
| Temperature = 0.2 | Maximally factual, minimal hallucination |
| 50-word chunk overlap | Preserves context at chunk boundaries |
| Threshold = 0.3 | Balances recall vs. precision for small KB |
| Session history (5 turns) | Enables follow-up without blowing context window |

---

## 📸 Screenshots

> Add screenshots of your running application here.

---

## 📄 License

MIT — see LICENSE for details.
