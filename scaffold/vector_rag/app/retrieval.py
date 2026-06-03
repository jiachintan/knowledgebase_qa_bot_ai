import json
import os
from typing import Generator

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


SYSTEM_PROMPT = """You are a knowledge base assistant. Answer questions using ONLY the provided CONTEXT.

Rules:
- Cite every fact using [Source: filename#heading] immediately after the statement.
- If the context does not contain the answer, respond with: "I cannot confirm this from the knowledge base."
- Do not use outside knowledge or guess. If you are unsure, say so."""

_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list) -> str:
    context_parts = []
    for doc, score in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }


def query_stream(question: str) -> Generator[str, None, None]:
    if indexer.vectorstore is None:
        yield f"event: error\ndata: {json.dumps({'message': 'Not indexed yet. Call POST /index first.'})}\n\n"
        return

    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        yield f"event: sources\ndata: {json.dumps([])}\n\n"
        yield f"event: token\ndata: {json.dumps({'text': 'I cannot confirm from the knowledge base.'})}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "heading": doc.metadata.get("heading", "unknown"),
            "score": round(float(score), 3),
            "content": doc.page_content[:240],
        }
        for doc, score in ranked_chunks
    ]
    yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

    prompt = build_prompt(question, ranked_chunks)
    for chunk in get_llm().stream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]):
        if chunk.content:
            yield f"event: token\ndata: {json.dumps({'text': chunk.content})}\n\n"

    yield "event: done\ndata: {}\n\n"
