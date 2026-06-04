# Scaffolds

Two independent retrieval strategy implementations that share the same API surface.

## Structure

```
scaffold/
├── common/
│   └── static/
│       ├── index.html       # Single chatbot UI — served at http://localhost:8000/
│       └── compare.html     # Side-by-side comparison UI — served at http://localhost:8000/compare
├── markdown_kb/             # Strategy A: BM25 keyword search, port 8000
└── vector_rag/              # Strategy B: Embeddings + FAISS, port 8001
```

## Starting Each Server

```bash
# Markdown KB — port 8000
cd scaffold/markdown_kb
.venv/bin/python -m uvicorn app.main:app --reload --port 8000

# Vector RAG — port 8001
cd scaffold/vector_rag
.venv/bin/python -m uvicorn app.main:app --reload --port 8001
```

Use `.venv/bin/python -m uvicorn` (not just `uvicorn`) to ensure the venv Python is used, not the system Python.

## Shared API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| GET | `/` | Single chatbot UI (markdown_kb only) |
| GET | `/compare` | Side-by-side comparison UI (markdown_kb only) |
| POST | `/index` | Build the retrieval index from `docs/*.md` |
| POST | `/chat` | Synchronous answer with sources |
| POST | `/chat/stream` | SSE streaming answer with sources |

## Key Differences

| | markdown_kb | vector_rag |
|--|-------------|------------|
| Retrieval | BM25 keyword scoring | FAISS vector similarity |
| Score direction | Higher is better | Lower is better (L2 distance) |
| Embeddings | Not required | Required (OpenAI API) |
| Index | `.kb/index.json` (human-readable) | `.kb/faiss_index/` (binary) |
| Re-index speed | Fast (milliseconds) | Slow (API calls per chunk) |

## First-Time Setup (per scaffold)

```bash
cd scaffold/<markdown_kb or vector_rag>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Only needed once. After that, just activate the venv and start the server.
