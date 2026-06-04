from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse

from .indexer import build_index
from .retrieval import query, query_stream
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()

_UI_HTML = (Path(__file__).parents[2] / "common" / "static" / "index.html").read_text(encoding="utf-8")
_COMPARE_HTML = (Path(__file__).parents[2] / "common" / "static" / "compare.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
def ui():
    return _UI_HTML


@router.get("/compare", response_class=HTMLResponse)
def compare_ui():
    return _COMPARE_HTML


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    try:
        files_count, sections_count = build_index()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return query(req.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat generation failed: {exc}") from exc


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    return StreamingResponse(query_stream(req.query), media_type="text/event-stream")
