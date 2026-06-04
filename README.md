# Knowledge Base Q&A Bot

A Q&A bot over a small Markdown knowledge base, implemented with two retrieval strategies side by side for comparison.

## Project Structure

```
knowledgebase_qa_bot_ai_jerry/
├── docs/                        # Source Markdown knowledge base
│   ├── refund_policy.md
│   ├── shipping_faq.md
│   └── account_help.md
├── scaffold/
│   ├── common/
│   │   └── static/
│   │       ├── index.html       # Single chatbot UI (served at /)
│   │       └── compare.html     # Side-by-side comparison UI (served at /compare)
│   ├── markdown_kb/             # Strategy A: BM25 keyword search
│   │   └── app/
│   │       ├── main.py
│   │       ├── routes.py
│   │       ├── indexer.py       # Parse sections, BM25 scoring
│   │       ├── retrieval.py     # Query pipeline + SSE streaming
│   │       └── schemas.py
│   └── vector_rag/              # Strategy B: Embeddings + FAISS
│       └── app/
│           ├── main.py
│           ├── routes.py
│           ├── indexer.py       # Chunk, embed, FAISS index
│           ├── retrieval.py     # Query pipeline + SSE streaming
│           └── schemas.py
├── .kb/                         # Generated index files (git-ignored)
│   ├── index.json               # Markdown KB section index
│   └── faiss_index/             # Vector RAG FAISS index
├── QUICKSTART.md                # Step-by-step server startup guide
├── KNOWLEDGE.md                 # Design decisions, pros/cons, Q&A
└── PROMPT.md                    # Original exercise specification
```

## Quick Start

See `QUICKSTART.md` for full instructions. Short version:

**1. Set your OpenAI API key:**
```bash
cp .env.example .env
# Edit .env and add your key
```

**2. Start both servers:**
```bash
# Terminal 1 — Markdown KB on port 8000
cd scaffold/markdown_kb
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Terminal 2 — Vector RAG on port 8001
cd scaffold/vector_rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/python -m uvicorn app.main:app --reload --port 8001
```

**3. Open the comparison UI:**
```bash
open http://localhost:8000/compare
```

Click **Index** in each panel, then start asking questions.

## API

Both strategies expose the same API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check → `{"status": "ok"}` |
| GET | `/` | Single chatbot browser UI |
| GET | `/compare` | Side-by-side comparison UI (Markdown KB only) |
| POST | `/index` | Read `docs/*.md` and build the retrieval index |
| POST | `/chat` | Answer a question, returns full JSON response |
| POST | `/chat/stream` | Answer a question, streams SSE tokens |

### SSE Streaming Event Sequence (`/chat/stream`)

```
event: sources   → retrieved sections/chunks sent first (before the answer)
event: token     → one per LLM output token (repeated)
event: done      → stream complete
event: error     → sent instead if the index has not been built
```

## Retrieval Strategies

| | Markdown KB | Vector RAG |
|--|-------------|------------|
| **Port** | 8000 | 8001 |
| **Retrieval unit** | Section (heading + content) | Chunk (~500 chars) |
| **Algorithm** | BM25 keyword scoring | Cosine/L2 similarity (FAISS) |
| **Score direction** | Higher = better match | Lower = closer (more similar) |
| **Embeddings needed** | No | Yes (OpenAI API) |
| **Persisted index** | `.kb/index.json` | `.kb/faiss_index/` |
| **Handles synonyms** | No | Yes |
| **Debuggability** | High — inspect `index.json` | Low — vectors are opaque |

### Markdown KB Flow
```
docs/*.md → parse headings → Section index → BM25 rank → top-3 sections → LLM → answer
```

### Vector RAG Flow
```
docs/*.md → parse headings → chunk (~500 chars) → embed (OpenAI) → FAISS index
                                                                         ↓
                                        query → embed → nearest chunks → LLM → answer
```

## Persistence

Both servers load their index on startup — no need to call `POST /index` again after a restart.

| Strategy | Index location | Rebuild when |
|----------|---------------|--------------|
| Markdown KB | `.kb/index.json` | After editing `docs/*.md` |
| Vector RAG | `.kb/faiss_index/` | After editing `docs/*.md` (costs API $) |

## Prerequisites

- Python 3.10+
- OpenAI API key (both strategies use it for LLM answer generation; Vector RAG also uses it for embeddings)

## Documentation

| File | Contents |
|------|----------|
| `QUICKSTART.md` | Step-by-step server startup, curl tests |
| `KNOWLEDGE.md` | Strategy comparison, pros/cons, scoring, design Q&A |
| `PROMPT.md` | Original exercise specification and design questions |
