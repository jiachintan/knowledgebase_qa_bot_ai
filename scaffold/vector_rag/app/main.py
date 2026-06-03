from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

from .indexer import load_vector_index
from .routes import router

app = FastAPI(title="Vector RAG Knowledge Base Q&A Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.on_event("startup")
def load_persisted_index():
    try:
        load_vector_index()
    except Exception as exc:
        print(f"[vector_rag] Skipping persisted FAISS load: {exc}", flush=True)
