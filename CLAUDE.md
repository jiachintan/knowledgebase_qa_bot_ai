### Streaming Interface

For a better user experience, add a streaming endpoint:

```text
POST /chat/stream
```

Recommended approach:

- Use Server-Sent Events (SSE) for a simple one-way token stream
- Send `source` events before token events so the UI can show what context was selected
- Send `token` events as the LLM produces output
- Send a final `done` event when the answer is complete

This is intentionally a stretch goal. The core exercise is still retrieval quality and grounded answer generation.

### Browser UI

Build a tiny browser UI over `/chat` or `/chat/stream`. Show selected sources before the answer, then render streamed tokens as they arrive.

### Multi-Format Import

Karpathy-style knowledge bases often treat Markdown as the canonical knowledge format, not the only input format.

Add an import pipeline:

```text
raw/*.txt or raw/*.html -> docs/*.md -> POST /index -> retrieval index
```

Recommended scope:

- Start with `.txt` and `.html`
- Preserve source filename in front matter or metadata
- Convert headings into Markdown headings
- Keep `docs/*.md` as the human-readable canonical copy
- Rebuild the retrieval index after conversion

Avoid parsing complex PDFs or spreadsheets first. The goal is to teach normalization into clean Markdown, not file parser edge cases.

### Alternative Interfaces

Keep the retrieval logic the same, but expose it through another interface:

```text
CLI: kb index / kb ask
MCP: expose index, search, and chat as agent tools
Web UI: simple chat screen over /chat or /chat/stream
```

This is useful for comparing interface design. The core exercise should still stay focused on indexing, retrieval, grounding, and citation quality.

### Wiki Index Generation

Generate `wiki/index.md` from `.kb/index.json` so humans and agents can browse the available topics without calling the API.

### Answer Filing

Write useful Q&A results back into `wiki/` after review. Keep filed answers source-grounded and preserve citations back to the original Markdown sections.

### Conversation Memory

Add short conversation memory for follow-up questions. Memory should help interpret the query, but it must not override retrieved sources or citation requirements.

### Paraphrase Comparison

Create a small set of paraphrased queries and compare Markdown KB vs Vector RAG. Look for cases where BM25 misses synonyms and cases where vector search retrieves semantically related but wrong chunks.

### additional UI
To make easy to compare could you provide me 2 chatbot, 1 for markdown and 1 for vector to easy see the difference.

- markdown is light yellow background color.
- vector is light blue background color.

- add suggested question, user can drag and drop to chatbox:
-- How long do refunds take?
-- Can I change my email address?
-- Which restaurants are nearby?
-- How long to get money back?