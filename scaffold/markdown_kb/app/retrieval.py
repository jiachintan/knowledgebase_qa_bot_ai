import os

from langchain.schema import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from . import indexer


SYSTEM_PROMPT = """You are a knowledge base Q&A assistant.
Rules:
1. Only answer using the provided CONTEXT.
2. Cite sources using filename#heading.
3. If the CONTEXT does not contain the answer, say: "I cannot confirm from the knowledge base."
4. Do not guess, invent policies, or use outside knowledge.
"""

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


def build_prompt(query: str, ranked_sections: list) -> str:
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
    if not indexer.sections:
        return {
            "answer": "The knowledge base has not been indexed yet. Call POST /index first.",
            "sources": [],
        }

    ranked_sections = indexer.search(question, k=3)
    if not ranked_sections:
        return {
            "answer": "I cannot confirm from the knowledge base.",
            "sources": [],
        }

    response = get_llm().invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_prompt(question, ranked_sections)),
    ])

    sources = [
        {
            "source": section.id,
            "heading": " > ".join(section.heading_path),
            "score": round(score, 3),
            "content": section.content[:240],
        }
        for section, score in ranked_sections
    ]

    return {
        "answer": response.content,
        "sources": sources,
    }
