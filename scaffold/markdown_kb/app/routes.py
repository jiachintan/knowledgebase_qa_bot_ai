from fastapi import APIRouter
from fastapi import HTTPException

from .indexer import build_index
from .retrieval import query
from .schemas import ChatRequest, ChatResponse, IndexResponse

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.post("/index", response_model=IndexResponse)
def index_docs():
    files_count, sections_count = build_index()
    return IndexResponse(files_indexed=files_count, sections_indexed=sections_count)


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        return query(req.query)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat generation failed: {exc}") from exc
