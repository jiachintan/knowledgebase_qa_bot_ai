from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from .indexer import load_index_json
from .routes import router

app = FastAPI(title="Markdown Knowledge Base Q&A Bot")
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    load_index_json()
