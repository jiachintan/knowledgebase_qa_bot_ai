# =============================================================================
# indexer.py — Vector RAG Strategy
# =============================================================================
#
# FLOW OVERVIEW
# -------------
# POST /index triggers build_index():
#   1. Glob all *.md files from docs/
#   2. load_markdown_sections() splits each file into heading-level Documents
#   3. splitter.split_documents() chunks each section into ~500-char pieces
#   4. FAISS.from_documents() embeds every chunk via OpenAI and builds the index
#   5. save_vector_index() persists the FAISS index + metadata to .kb/faiss_index/
#
# On server startup, load_vector_index() reloads the FAISS index from disk so
# POST /index does not need to be called again after a restart.
#
# At query time (retrieval.py calls search()):
#   6. The query string is embedded by FAISS (via the same OpenAI model)
#   7. similarity_search_with_score() returns the k nearest chunks by L2 distance
#
# RETRIEVAL UNIT: Chunk (~500 chars, split from a heading section)
# ALGORITHM:      Dense vector similarity (cosine/L2) via FAISS flat index
# PERSISTENCE:    .kb/faiss_index/  (index.faiss + index.pkl + metadata.json)
# EMBEDDING MODEL: text-embedding-3-small (stored in metadata for version safety)
# =============================================================================

import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# Paths resolved relative to this file so the server can run from any CWD.
DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"

# Embedding model name — stored in metadata.json so a model change is detected on reload
EMBEDDING_MODEL = "text-embedding-3-small"

# Matches Markdown headings: "## My Heading" → group(1)="##", group(2)="My Heading"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Text splitter for chunking sections before embedding.
# chunk_size=500 chars balances embedding precision vs. context richness.
# separators prefer paragraph breaks → line breaks → sentence → word.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " "],
)

# In-memory FAISS vectorstore — populated by build_index() or load_vector_index()
vectorstore: FAISS | None = None

# Lazy singleton for the OpenAI embeddings client
_embeddings = None

files_indexed = 0
sections_indexed = 0  # actually chunk count after splitting


def slugify(text: str) -> str:
    """Convert heading text to a URL-safe anchor, e.g. "Refund Timeline" → "refund-timeline"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
    """Return a shared OpenAIEmbeddings instance (created once, reused across requests).

    Raises RuntimeError if OPENAI_API_KEY is not set — embeddings require the API.
    """
    global _embeddings
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in the server environment")
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            request_timeout=20,
            max_retries=1,
        )
    return _embeddings


def load_markdown_sections(path: Path) -> list[Document]:
    """Parse a Markdown file into heading-level LangChain Documents.

    Each heading starts a new section. Content lines accumulate until the
    next heading, then flush() creates a Document with:
        page_content = "<heading line>\\n\\n<content>"
        metadata     = {"source": "filename.md#heading-slug"}

    The source metadata becomes the citation key in the LLM response.

    Example:
        ## Refund Timeline
        Refunds are processed within 5–10 business days.
        →  Document(
               page_content="## Refund Timeline\\n\\nRefunds are processed...",
               metadata={"source": "refund_policy.md#refund-timeline"}
           )
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    sections: list[Document] = []
    current_heading = ""
    current_slug = ""
    current_lines: list[str] = []

    def flush():
        content = "".join(current_lines).strip()
        if content:
            source = f"{path.name}#{current_slug}" if current_slug else path.name
            sections.append(Document(
                page_content=f"{current_heading}\n\n{content}" if current_heading else content,
                metadata={"source": source}
            ))

    for line in lines:
        m = HEADING_RE.match(line.rstrip())
        if m:
            flush()
            current_heading = line.rstrip()
            current_slug = slugify(m.group(2))
            current_lines = []
        else:
            current_lines.append(line)

    flush()
    return sections


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build the FAISS vector index from all *.md files in docs/.

    Steps:
        1. Parse each file into heading-level Documents
        2. Split Documents into ~500-char chunks
        3. Embed all chunks via OpenAI (one API call per batch)
        4. Store vectors in FAISS flat index (exact L2 search)
        5. Persist to .kb/faiss_index/ for server restarts

    Returns (files_indexed, chunks_indexed).
    Note: this call is slow and costs money — each chunk requires an embedding API call.
    """
    global vectorstore, files_indexed, sections_indexed

    md_files = sorted(docs_dir.glob("*.md"))
    all_sections: list[Document] = []
    for md_file in md_files:
        all_sections.extend(load_markdown_sections(md_file))

    # Split sections into smaller chunks for more precise embedding retrieval
    chunks = splitter.split_documents(all_sections)

    # Embed all chunks and build the FAISS index in one call
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    files_indexed = len(md_files)
    sections_indexed = len(chunks)
    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    """Persist the FAISS index and metadata to disk.

    Saves:
        index.faiss   — the FAISS binary index (vectors)
        index.pkl     — the docstore mapping (chunk text + metadata)
        metadata.json — embedding model name + index stats for version checking

    Clears the existing directory first to avoid stale files from a previous run.
    """
    if vectorstore is None:
        return
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {
        "embedding_model": EMBEDDING_MODEL,
        "files_indexed": files_indexed,
        "sections_indexed": sections_indexed,
    }
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    """Load the persisted FAISS index from disk on server startup.

    Safety checks before loading:
        1. Both index.faiss and index.pkl must exist (incomplete save → skip)
        2. metadata.json must exist (missing → possibly corrupted → skip)
        3. Embedding model in metadata must match EMBEDDING_MODEL constant
           (model mismatch → vectors are incompatible → skip)

    Returns (files_indexed, sections_indexed), or (0, 0) if load is skipped.
    """
    global vectorstore, files_indexed, sections_indexed
    if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
        return 0, 0
    meta_path = index_dir / "metadata.json"
    if not meta_path.exists():
        # Metadata missing — can't verify model compatibility, skip to be safe
        return 0, 0
    meta = json.loads(meta_path.read_text())
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        # Vectors were built with a different model — they are not comparable
        return 0, 0
    files_indexed = meta.get("files_indexed", 0)
    sections_indexed = meta.get("sections_indexed", 0)
    # allow_dangerous_deserialization=True is safe here because this index
    # was created by this application on the local machine.
    vectorstore = FAISS.load_local(str(index_dir), get_embeddings(), allow_dangerous_deserialization=True)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    """Find the k most semantically similar chunks for the given query.

    The query is embedded using the same OpenAI model as the index.
    FAISS returns chunks sorted by L2 distance (lower = more similar).

    Returns an empty list if the index has not been loaded yet.
    """
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
