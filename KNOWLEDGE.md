# Knowledge Base Q&A Bot — Design Knowledge

This document covers the design decisions behind this project, compares the two retrieval strategies, and answers the eight design questions from `PROMPT.md`.

---

## Retrieval Strategies Overview

### Strategy A: Markdown KB (BM25 Keyword Search)

```
Markdown files → heading sections → section index → BM25 keyword search → raw Markdown context → LLM answer
```

The knowledge base is parsed into heading-level sections. Each section becomes a unit in a JSON index (`.kb/index.json`). At query time, BM25 — a classical information retrieval algorithm — scores each section by keyword overlap with the query. The top-k sections are injected into the LLM prompt as context.

**How BM25 works:**
BM25 scores a section higher when:
- A query term appears frequently in that section (term frequency)
- The term is rare across all sections (inverse document frequency)
- The section is not abnormally long (length normalization)

It also gives a bonus when query terms appear in section headings.

### Strategy B: Vector RAG (Embeddings + FAISS)

```
Markdown files → chunks → embeddings → FAISS vector index → retrieved chunks → LLM answer
```

Each section is split into smaller chunks (~500 chars). Each chunk is embedded using OpenAI's `text-embedding-3-small` model — a vector of numbers that encodes semantic meaning. All chunk vectors are stored in a FAISS index (`.kb/faiss_index/`). At query time, the query is also embedded and FAISS finds the nearest chunk vectors by cosine similarity. The closest chunks are used as context for the LLM.

---

## Pros and Cons

### Markdown KB

| Pros | Cons |
|------|------|
| No embeddings API calls — fast and cheap to index | Misses synonyms and paraphrases (e.g. "money back" won't match "refund") |
| Index is a human-readable JSON file — easy to inspect and debug | Sensitive to exact wording; keyword stuffing can fool it |
| Retrieval unit is a whole section — rich context, no mid-sentence cuts | Large sections dilute scores across many terms |
| No vector store dependency — runs fully offline after indexing | Heading bonus is hand-tuned and fragile |
| Transparent: you can see exactly which sections were ranked and why | Doesn't understand semantic similarity |
| Instant re-index on doc changes | BM25 score thresholds need manual calibration |

### Vector RAG

| Pros | Cons |
|------|------|
| Handles synonyms, paraphrases, and semantic similarity | Requires OpenAI API for both indexing and querying — costs money |
| "How long to get money back?" correctly retrieves refund content | Embedding is a black box — hard to debug why a chunk was retrieved |
| Scales better to larger corpora with diverse topics | Chunks can cut across sentence boundaries — loses context |
| State-of-the-art retrieval quality on semantic queries | Re-indexing is slow and expensive (all chunks must be re-embedded) |
| FAISS gives sub-linear search on large indexes | Wrong embedding model on a saved index → garbage results |
| Works well when docs use varied vocabulary | Chunk size and overlap require tuning |

---

## Design Questions (from PROMPT.md)

### 1. Which retrieval strategy did you choose, and why?

This project implements **both**, which is the most instructive choice. For production with a small, stable knowledge base (< 50 docs, controlled vocabulary), **Markdown KB** is preferred — it is cheaper, fully inspectable, and faster to iterate. For a larger or more varied corpus where users ask in natural language with synonyms and paraphrases, **Vector RAG** is preferred.

If forced to choose one starting point: **Markdown KB first**. It is simpler to build, debug, and explain. Upgrade to Vector RAG when you have evidence that BM25 is missing relevant sections.

---

### 2. What is the retrieval unit: file, section, or chunk?

- **Markdown KB** uses the **section** (a heading + its content) as the retrieval unit. This preserves structure and gives the LLM rich, coherent context. A section is typically 100–600 words.
- **Vector RAG** uses **chunks** (~500 chars) split from sections. Chunks are smaller than sections, which improves embedding precision but risks cutting sentences mid-thought. `chunk_overlap` can be increased to reduce boundary artifacts.

The **section** is generally the better unit for small knowledge bases because it keeps related information together. Chunks are better for large documents where a single section might be 2,000+ words.

---

### 3. How do you decide what goes into the prompt?

Both strategies inject the **top-3 retrieved results** into the prompt. The decision process:

1. Run retrieval to get ranked results.
2. If no results score above zero (BM25) or the result list is empty (FAISS), skip the LLM call entirely and return a cannot-confirm answer.
3. For the top-3 results, include: the source ID (`filename#heading`), the score, and the full section/chunk content.
4. Separate each context block with `---` so the LLM can distinguish boundaries.
5. Place `CONTEXT:` before `QUESTION:` so the model reads evidence before forming an answer.

The system prompt instructs the LLM to use **only** the provided context, cite sources, and say it cannot confirm if the context is insufficient.

**What to tune:**
- `k=3` is a starting point. Increase for broad queries; decrease to reduce hallucination risk from irrelevant context.
- Add a score threshold: if the best BM25 score < 1.0 or best vector distance > 0.4, treat as no-match.

---

### 4. How do you cite sources so users can inspect the original Markdown?

Sources are cited as `filename#heading-slug`, e.g. `refund_policy.md#refund-timeline`.

- The **filename** points to the file in `docs/`.
- The **heading slug** is a URL-safe version of the heading text, matching GitHub's anchor format.
- The LLM is instructed to cite inline as `[Source: filename#heading]` immediately after each fact.
- The API response also returns a `sources` array with the source ID, heading path, score, and a 240-char content preview.
- In the streaming UI, source chips appear **before** the answer so users see what evidence the bot is using before reading its conclusion.

This means a user can open `docs/refund_policy.md` and search for the heading to verify any claim.

---

### 5. What should happen when retrieval finds weak or irrelevant results?

**Current behavior:** If retrieval returns zero results, the bot returns a hard-coded cannot-confirm message without calling the LLM. This is correct — no context means no answer.

**Recommended improvement — score threshold:**

Even when results exist, they may be irrelevant. "Which restaurants are nearby?" will retrieve some sections because BM25 finds partial keyword matches, but those sections won't contain useful information.

Add a threshold:
- **BM25**: if the top score is below ~1.5, treat as no-match.
- **Vector RAG**: FAISS returns L2 distance; if the smallest distance is above ~0.8, treat as no-match.

When below threshold, skip the LLM call and return:
```json
{"answer": "I cannot confirm from the knowledge base.", "sources": []}
```

This prevents the LLM from hallucinating an answer from weak context. Track how often this happens — it tells you what topics your knowledge base is missing.

---

### 6. When would you switch from Markdown KB to Vector RAG?

Switch when you observe:

| Signal | Reason |
|--------|---------|
| Users phrase questions differently from how docs are written | BM25 misses synonyms; embeddings handle paraphrases |
| Knowledge base has 50+ files with diverse topics | BM25 IDF becomes less reliable; FAISS scales better |
| Queries require semantic understanding across sections | Vector search retrieves semantically related content |
| Multilingual docs or queries | Embeddings handle cross-lingual similarity; BM25 does not |
| Retrieval quality is provably poor (user feedback, A/B test) | Evidence-driven upgrade |

Do **not** switch just because Vector RAG sounds more sophisticated. If BM25 is working, keep it.

---

### 7. When would you switch from Vector RAG back to a Markdown index?

Switch back when:

| Signal | Reason |
|--------|---------|
| Embedding API costs are too high | Re-embedding on every doc change is expensive |
| Retrieval results are hard to explain to stakeholders | BM25 scores are interpretable; vectors are not |
| Knowledge base is small and stable (< 20 docs, controlled vocabulary) | BM25 is fast, free, and accurate enough |
| Need to run fully offline or on-premise | No embedding API dependency |
| Chunk boundaries are causing context loss | Sections are richer retrieval units than chunks |
| Debugging is too difficult | You can inspect `.kb/index.json` directly |

Vector RAG adds operational complexity (API keys, embedding costs, FAISS serialization, model versioning). Only keep it if the quality improvement is measurable and worth the cost.

---

### 8. If the knowledge base grows from 10 files to 100,000 files, what changes?

#### Markdown KB at scale

| Component | What breaks | Fix |
|-----------|-------------|-----|
| `.kb/index.json` | JSON file becomes gigabytes; slow to load at startup | Replace with SQLite FTS5 or Elasticsearch |
| BM25 in-memory search | Iterating all sections per query is O(n) | Use an inverted index (Elasticsearch, Typesense, Meilisearch) |
| Doc parsing | Still fast — parsing is per-file | Add a file watcher for incremental updates |

At ~5,000 sections, in-memory BM25 slows noticeably. At 100,000 files, you need a dedicated search engine.

#### Vector RAG at scale

| Component | What breaks | Fix |
|-----------|-------------|-----|
| FAISS flat index | L2 search is O(n) — slow at 1M+ vectors | Switch to FAISS HNSW or IVF for approximate nearest neighbor |
| Re-embedding on change | Embedding 100k files takes hours and real cost | Incremental embedding: only embed changed/new files |
| In-memory FAISS | Gigabytes of vectors won't fit in RAM | Use Pinecone, Weaviate, Qdrant, or pgvector |
| Single embedding model | Model may be deprecated | Version the model in metadata; re-embed on model change |

#### Common changes at 100,000 files

- **Incremental indexing**: only re-index changed files, not the entire corpus.
- **Metadata filtering**: pre-filter by category, date, or tag before retrieval.
- **Hybrid search**: combine BM25 and vector scores (reciprocal rank fusion) for best recall and precision.
- **Async indexing**: indexing becomes a background job, not a synchronous API call.
- **Monitoring**: track retrieval quality, latency, and fallback rates continuously.

---

## Summary Table

| Dimension | Markdown KB | Vector RAG |
|-----------|-------------|------------|
| Retrieval unit | Section (heading + content) | Chunk (~500 chars) |
| Algorithm | BM25 keyword scoring | Cosine similarity on embeddings |
| Cost to index | Free | OpenAI API ($) |
| Cost per query | Free | OpenAI API ($) |
| Synonym handling | No | Yes |
| Debuggability | High — inspect `.kb/index.json` | Low — vectors are opaque |
| Best for | Small, stable, controlled-vocab KB | Large, varied, semantic queries |
| Scale ceiling | ~5k sections in-memory | ~1M chunks with HNSW index |
| Offline capable | Yes | No (needs embedding API) |
| Re-index speed | Fast (milliseconds) | Slow (seconds to minutes) |
