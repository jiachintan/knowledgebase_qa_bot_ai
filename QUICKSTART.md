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

## 2. Run both servers (for side-by-side comparison)

Open **two terminals** from the project root.

**Terminal 1 — Markdown KB on port 8000:**

```bash
cd scaffold/markdown_kb
python3 -m venv .venv        # only needed once
source .venv/bin/activate
pip install -r requirements.txt  # only needed once
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Vector RAG on port 8001:**

```bash
cd scaffold/vector_rag
python3 -m venv .venv        # only needed once
source .venv/bin/activate
pip install -r requirements.txt  # only needed once
.venv/bin/python -m uvicorn app.main:app --reload --port 8001
```

**Then open the comparison UI:**

```bash
open http://localhost:8000/compare
```

Click **Index** in each panel to build the index, then start asking questions.

---

## Run a single server only

### Markdown KB only

```bash
cd scaffold/markdown_kb
source .venv/bin/activate
.venv/bin/python -m uvicorn app.main:app --reload --port 8000
```

Open `http://localhost:8000`

### Vector RAG only

```bash
cd scaffold/vector_rag
source .venv/bin/activate
.venv/bin/python -m uvicorn app.main:app --reload --port 8001
```

---

## 3. Build the index

```bash
# Markdown KB
curl -X POST http://localhost:8000/index

# Vector RAG
curl -X POST http://localhost:8001/index
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
curl http://localhost:8001/health
# -> {"status": "ok"}
```

## Notes

- Run `pip install -r requirements.txt` and create the venv only once per scaffold.
- Use `.venv/bin/python -m uvicorn` (not just `uvicorn`) to ensure the venv's Python is used.
- Re-run `POST /index` after editing any file in `docs/`.
- Restarting a server does **not** require rebuilding the index — it loads the persisted index on startup.
- The comparison UI at `/compare` requires **both** servers running simultaneously.
