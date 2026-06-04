# =============================================================================
# retrieval.py — Vector RAG Strategy
# =============================================================================
#
# FLOW OVERVIEW
# -------------
# POST /chat  → query()        — returns a complete JSON response
# POST /chat/stream → query_stream() — streams SSE events token by token
#
# Both functions follow the same pipeline:
#   1. Guard: if the FAISS index is not loaded, return a not-indexed error
#   2. Retrieve: call indexer.search() to embed the query and find the top-3
#      nearest chunks by L2 distance in the FAISS index
#   3. Guard: if no chunks found, return a cannot-confirm response
#   4. Build prompt: inject retrieved chunks as CONTEXT above the QUESTION
#   5. Call LLM: send system prompt + user prompt to gpt-4o-mini
#   6. Return answer + sources (chunk source IDs, distances, content previews)
#
# For streaming, the LLM response is streamed token by token via SSE:
#   → event: sources  (sent first so the UI can show grounding before the answer)
#   → event: token    (one per LLM output chunk, repeated until done)
#   → event: done     (signals the client the stream has ended)
#
# KEY DIFFERENCE vs Markdown KB:
#   - Retrieval uses semantic similarity (embeddings), not keyword overlap (BM25)
#   - This handles synonyms and paraphrases: "money back" finds "refund" content
#   - Score is an L2 distance (lower = more similar), not a BM25 relevance score
# =============================================================================

import json
import os
from typing import Generator

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


# System prompt sent to the LLM on every request.
# Instructs the model to stay grounded in CONTEXT and cite sources inline.
SYSTEM_PROMPT = """You are a knowledge base assistant. Answer questions using ONLY the provided CONTEXT.

Rules:
- Cite every fact using [Source: filename#heading] immediately after the statement.
- If the context does not contain the answer, respond with: "I cannot confirm this from the knowledge base."
- Do not use outside knowledge or guess. If you are unsure, say so."""

# Lazy singleton — instantiated on first request to avoid startup cost
_llm = None


def get_llm():
    """Return a shared ChatOpenAI instance (created once, reused across requests)."""
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            request_timeout=20,
            max_retries=1,
        )
    return _llm


def build_prompt(query: str, ranked_chunks: list) -> str:
    """Assemble the user message sent to the LLM.

    Each retrieved chunk is formatted as:
        [Source: filename#heading-slug]
        <chunk content>

    Chunks are separated by --- so the LLM can distinguish boundaries.
    CONTEXT is placed before QUESTION so the model reads evidence first.

    Note: unlike Markdown KB, the score (L2 distance) is omitted from the
    prompt — lower distance is better, which is counterintuitive for the LLM.
    """
    context_parts = []
    for doc, score in ranked_chunks:
        source = doc.metadata.get("source", "unknown")
        context_parts.append(f"[Source: {source}]\n{doc.page_content}")
    context = "\n\n---\n\n".join(context_parts)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    """Answer a question synchronously using vector retrieval + LLM generation.

    Returns a dict with:
        answer  — the LLM's grounded answer string
        sources — list of {source, heading, score, content} for each retrieved chunk
    """
    # Guard: FAISS index must be loaded before answering
    if indexer.vectorstore is None:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Step 1: Vector retrieval — embed the query and find the top-3 nearest chunks
    ranked_chunks = indexer.search(question, k=3)
    if not ranked_chunks:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    # Step 2: LLM generation — blocked call, waits for full response
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_chunks)),
    ])

    # Step 3: Build source metadata for the API response
    sources = [
        {
            "source": doc.metadata.get("source", "unknown"),   # e.g. "refund_policy.md#refund-timeline"
            "heading": doc.metadata.get("heading", "unknown"),  # heading from metadata if available
            "score": round(float(score), 3),                    # L2 distance (lower = more similar)
            "content": doc.page_content[:240],                  # preview for the UI chip tooltip
        }
        for doc, score in ranked_chunks
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }


def query_stream(question: str) -> Generator[str, None, None]:
    """Answer a question and stream the response as Server-Sent Events (SSE).

    SSE event sequence:
        event: error   — only sent if not indexed (early exit)
        event: sources — sent before any tokens so the UI shows grounding first
        event: token   — one event per LLM output chunk (repeated)
        event: done    — signals the stream is complete

    Each event is formatted as:
        event: <name>\\ndata: <json>\\n\\n
    """
    # Guard: FAISS index must be loaded before answering
    if indexer.vectorstore is None:
        yield f"event: error\ndata: {json.dumps({'message': 'Not indexed yet. Call POST /index first.'})}\n\n"
        return

    # Step 1: Vector retrieval — embed query, find nearest chunks
    ranked_chunks = indexer.search(question, k=3)

    # Guard: no results — send empty sources then a cannot-confirm token
    if not ranked_chunks:
        yield f"event: sources\ndata: {json.dumps([])}\n\n"
        yield f"event: token\ndata: {json.dumps({'text': 'I cannot confirm from the knowledge base.'})}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    # Step 2: Send sources first — UI renders source chips before the answer appears
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

    # Step 3: Stream LLM tokens — each chunk is yielded as it arrives
    prompt = build_prompt(question, ranked_chunks)
    for chunk in get_llm().stream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]):
        if chunk.content:
            yield f"event: token\ndata: {json.dumps({'text': chunk.content})}\n\n"

    # Step 4: Signal completion
    yield "event: done\ndata: {}\n\n"
