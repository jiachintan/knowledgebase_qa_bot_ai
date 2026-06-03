# Quickstart — Run the Server

## 1. Set your OpenAI API key

Copy the example env file and add your key:

```bash
cp .env.example .env
```

Open `.env` and replace the placeholder:

```
OPENAI_API_KEY=sk-your-actual-key-here
```

## 2. Start the server

Choose a retrieval strategy:

### Markdown KB (simpler, no embeddings)

```bash
cd scaffold/markdown_kb
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Vector RAG (uses OpenAI embeddings + FAISS)

```bash
cd scaffold/vector_rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Server runs at `http://localhost:8000`.

## 3. Build the index

```bash
curl -X POST http://localhost:8000/index
```

## 4. Ask a question

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "How long do refunds take?"}'
```

## 5. Health check

```bash
curl http://localhost:8000/health
# -> {"status": "ok"}
```

## Notes

- Run `pip install -r requirements.txt` and create the venv only once.
- Re-run `POST /index` after editing any file in `docs/`.
- Restarting the server does **not** require rebuilding the index — it loads the persisted index on startup.
