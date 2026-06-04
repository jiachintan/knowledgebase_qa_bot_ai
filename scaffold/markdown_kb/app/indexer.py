# =============================================================================
# indexer.py — Markdown KB Strategy
# =============================================================================
#
# FLOW OVERVIEW
# -------------
# POST /index triggers build_index():
#   1. Glob all *.md files from docs/
#   2. parse_markdown() splits each file into heading-level Sections
#   3. rebuild_stats() computes BM25 statistics (doc freq, avg length)
#   4. write_index_json() persists the index to .kb/index.json
#
# On server startup, load_index_json() reloads the index from disk so
# POST /index does not need to be called again after a restart.
#
# At query time (retrieval.py calls search()):
#   5. tokenize() lowercases and removes stop words from the query
#   6. bm25_score() ranks every section against the query tokens
#   7. Top-k sections with score > 0 are returned to the caller
#
# RETRIEVAL UNIT: Section (one Markdown heading + its content)
# ALGORITHM:      BM25 — classical keyword scoring, no embeddings needed
# PERSISTENCE:    .kb/index.json (human-readable, inspectable)
# =============================================================================

import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


# Paths resolved relative to this file so the server can run from any CWD.
DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"

# Matches Markdown headings: "## My Heading" → group(1)="##", group(2)="My Heading"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Matches lowercase alphanumeric tokens for BM25 indexing
TOKEN_RE = re.compile(r"[a-z0-9]+")

# Common words excluded from BM25 scoring to reduce noise
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "from",
    "how",
    "i",
    "is",
    "it",
    "my",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
}


@dataclass
class Section:
    """One heading-level unit from a Markdown file.

    id           — citation key, e.g. "refund_policy.md#refund-timeline"
    file         — source filename, e.g. "refund_policy.md"
    heading      — the heading text, e.g. "Refund Timeline"
    heading_path — breadcrumb from root to this heading, e.g. ["Refunds", "Refund Timeline"]
    content      — raw Markdown text under this heading (no heading line)
    tokens       — pre-tokenized words used for BM25 scoring
    """

    id: str
    file: str
    heading: str
    heading_path: list[str]
    content: str
    tokens: list[str]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file": self.file,
            "heading": self.heading,
            "heading_path": self.heading_path,
            "content": self.content,
            "tokens": self.tokens,
        }


# In-memory index — populated by build_index() or load_index_json() at startup
sections: list[Section] = []

# doc_freq[term] = number of sections containing that term (for IDF calculation)
doc_freq: Counter[str] = Counter()

# Average number of tokens per section (for BM25 length normalization)
avg_doc_len = 0.0

files_indexed = 0


def slugify(text: str) -> str:
    """Convert heading text to a URL-safe anchor, e.g. "Refund Timeline" → "refund-timeline"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, and remove stop words."""
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
    """Parse a Markdown file into a list of Sections, one per heading.

    Each time a heading line is encountered, the accumulated content lines
    for the previous heading are flushed into a Section. heading_path tracks
    the ancestor headings so the LLM gets breadcrumb context.

    Example for refund_policy.md:
        ## Refunds
        ### Refund Timeline
            → Section(id="refund_policy.md#refund-timeline",
                       heading_path=["Refunds", "Refund Timeline"], ...)
    """
    parsed: list[Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_heading = path.stem.replace("_", " ").title()
    current_level = 1
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if not content:
            current_lines = []
            return
        heading_path = [title for _, title in heading_stack] or [current_heading]
        section_id = f"{path.name}#{slugify(current_heading)}"
        # Include heading path in the tokenized text so heading terms boost BM25 scores
        full_text = "\n".join([*heading_path, content])
        parsed.append(
            Section(
                id=section_id,
                file=path.name,
                heading=current_heading,
                heading_path=heading_path,
                content=content,
                tokens=tokenize(full_text),
            )
        )
        current_lines = []

    for line in path.read_text(encoding="utf-8").splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            # Keep only ancestors with a smaller heading level (e.g. drop ## when seeing ##)
            heading_stack = [(level, title) for level, title in heading_stack if level < current_level]
            heading_stack.append((current_level, current_heading))
        else:
            current_lines.append(line)

    flush()
    return parsed


def write_index_json(index_path: Path = INDEX_PATH) -> None:
    """Persist the in-memory index to .kb/index.json.

    The file is human-readable — open it to inspect what was indexed,
    verify section boundaries, or debug retrieval issues.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sections": [section.to_dict() for section in sections],
        "stats": {
            "files_indexed": files_indexed,
            "sections_indexed": len(sections),
            "avg_doc_len": avg_doc_len,
        },
    }
    index_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def rebuild_stats() -> None:
    """Recompute BM25 statistics from the current sections list.

    Must be called after sections is populated or reloaded so that
    doc_freq and avg_doc_len are consistent with the active index.
    """
    global doc_freq, avg_doc_len, files_indexed

    files_indexed = len({section.file for section in sections})
    doc_freq = Counter()
    for section in sections:
        # Count each term once per section (set) for IDF, not total occurrences
        doc_freq.update(set(section.tokens))
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
    """Load the persisted index from .kb/index.json on server startup.

    This means POST /index only needs to be called once. Restarting the
    server reloads the existing index automatically.

    Returns (files_indexed, sections_indexed), or (0, 0) if no index exists.
    """
    global sections

    if not index_path.exists():
        return 0, 0

    payload = json.loads(index_path.read_text(encoding="utf-8"))
    sections = [
        Section(
            id=item["id"],
            file=item["file"],
            heading=item["heading"],
            heading_path=item["heading_path"],
            content=item["content"],
            tokens=item["tokens"],
        )
        for item in payload.get("sections", [])
    ]
    rebuild_stats()
    return files_indexed, len(sections)


def build_index(docs_dir: Path = DOCS_DIR) -> tuple[int, int]:
    """Build the BM25 index from all *.md files in docs/.

    Called by POST /index. Replaces any existing in-memory index and
    overwrites .kb/index.json.

    Returns (files_indexed, sections_indexed).
    """
    global sections, doc_freq, avg_doc_len, files_indexed

    markdown_files = sorted(docs_dir.glob("*.md"))
    new_sections: list[Section] = []
    for path in markdown_files:
        new_sections.extend(parse_markdown(path))

    sections = new_sections
    rebuild_stats()
    write_index_json()
    return files_indexed, len(sections)


def bm25_score(query_tokens: list[str], section: Section, k1: float = 1.5, b: float = 0.75) -> float:
    """Compute the BM25 relevance score for a section against query tokens.

    BM25 formula per term:
        IDF * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * (doc_len / avg_doc_len)))

    Where:
        IDF  = log(1 + (N - df + 0.5) / (df + 0.5))  — rare terms score higher
        tf   = term frequency in this section
        k1   = term frequency saturation (1.5 = moderate saturation)
        b    = length normalization strength (0.75 = standard)

    A heading bonus (+1.5) is added when any query term appears in the
    heading path, rewarding sections whose titles match the question.
    """
    if not sections or not section.tokens:
        return 0.0

    counts = Counter(section.tokens)
    score = 0.0
    for term in query_tokens:
        if term not in counts:
            continue
        n_docs = len(sections)
        idf = math.log(1 + (n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
        tf = counts[term]
        length_norm = 1 - b + b * (len(section.tokens) / avg_doc_len)
        score += idf * ((tf * (k1 + 1)) / (tf + k1 * length_norm))

    # Heading match bonus: boosts sections whose heading directly names the topic
    heading_text = " ".join(section.heading_path).lower()
    if any(term in heading_text for term in query_tokens):
        score += 1.5

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    """Rank all sections against the query using BM25 and return the top-k.

    Only sections with a score > 0 are returned — a zero score means no
    query token appeared in the section at all.
    """
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
