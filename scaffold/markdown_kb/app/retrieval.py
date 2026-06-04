# =============================================================================
# retrieval.py — Markdown KB Strategy
# =============================================================================
#
# FLOW OVERVIEW
# -------------
# POST /chat  → query()        — returns a complete JSON response
# POST /chat/stream → query_stream() — streams SSE events token by token
#
# Both functions follow the same pipeline:
#   1. Guard: if the index is empty, return a not-indexed error immediately
#   2. Retrieve: call indexer.search() to get the top-3 BM25-ranked sections
#   3. Guard: if no sections scored > 0, return a cannot-confirm response
#   4. Build prompt: inject retrieved sections as CONTEXT above the QUESTION
#   5. Call LLM: send system prompt + user prompt to gpt-4o-mini
#   6. Return answer + sources (section IDs, scores, content previews)
#
# For streaming, the LLM response is streamed token by token via SSE:
#   → event: sources  (sent first so the UI can show grounding before the answer)
#   → event: token    (one per LLM output chunk, repeated until done)
#   → event: done     (signals the client the stream has ended)
# =============================================================================

import json
import os
from typing import Generator

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


# System prompt sent to the LLM on every request.
# Instructs the model to stay grounded in CONTEXT and cite sources.
SYSTEM_PROMPT = """You are a knowledge base Q&A assistant.
Rules:
1. Only answer using the provided CONTEXT.
2. Cite every fact using [Source: filename#heading] immediately after the statement.
3. If the CONTEXT does not contain the answer, say: "I cannot confirm from the knowledge base."
4. Do not guess, invent policies, or use outside knowledge.
"""

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


def build_prompt(query: str, ranked_sections: list) -> str:
    """Assemble the user message sent to the LLM.

    Each retrieved section is formatted as:
        [Source: filename#heading]
        [BM25 score: X.XX]
        # Heading breadcrumb
        <section content>

    Sections are separated by --- so the LLM can distinguish boundaries.
    CONTEXT is placed before QUESTION so the model reads evidence first.
    """
    context_blocks = []
    for section, score in ranked_sections:
        heading_lines = "\n".join(
            f"{'#' * (idx + 1)} {heading}"
            for idx, heading in enumerate(section.heading_path)
        )
        context_blocks.append(
            f"[Source: {section.id}]\n"
            f"[BM25 score: {score:.2f}]\n"
            f"{heading_lines}\n\n"
            f"{section.content}"
        )

    context = "\n\n---\n\n".join(context_blocks)
    return f"CONTEXT:\n{context}\n\nQUESTION:\n{query}"


def query(question: str) -> dict:
    """Answer a question synchronously using BM25 retrieval + LLM generation.

    Returns a dict with:
        answer  — the LLM's grounded answer string
        sources — list of {source, heading, score, content} for each retrieved section
    """
    # Guard: index must be built before answering
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    # Step 1: BM25 retrieval — returns top-3 sections with score > 0
    ranked_sections = indexer.search(question, k=3)
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    # Step 2: LLM generation — blocked call, waits for full response
    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    # Step 3: Build source metadata for the API response
    sources = [
        {
            "source": section.id,           # e.g. "refund_policy.md#refund-timeline"
            "heading": " > ".join(section.heading_path),  # e.g. "Refunds > Refund Timeline"
            "score": round(score, 3),        # BM25 score for transparency
            "content": section.content[:240],  # preview for the UI chip tooltip
        }
        for section, score in ranked_sections
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
    # Guard: index must be built before answering
    if not indexer.sections:
        yield f"event: error\ndata: {json.dumps({'message': 'Not indexed yet. Call POST /index first.'})}\n\n"
        return

    # Step 1: BM25 retrieval
    ranked_sections = indexer.search(question, k=3)

    # Guard: no results — send empty sources then a cannot-confirm token
    if not ranked_sections:
        yield f"event: sources\ndata: {json.dumps([])}\n\n"
        yield f"event: token\ndata: {json.dumps({'text': 'I cannot confirm from the knowledge base.'})}\n\n"
        yield "event: done\ndata: {}\n\n"
        return

    # Step 2: Send sources first — UI renders source chips before the answer appears
    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]
    yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

    # Step 3: Stream LLM tokens — each chunk is yielded as it arrives
    prompt = build_prompt(question, ranked_sections)
    for chunk in get_llm().stream([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]):
        if chunk.content:
            yield f"event: token\ndata: {json.dumps({'text': chunk.content})}\n\n"

    # Step 4: Signal completion
    yield "event: done\ndata: {}\n\n"
