import math
import re
from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_PATH = Path(__file__).resolve().parents[3] / ".kb" / "index.json"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
TOKEN_RE = re.compile(r"[a-z0-9]+")
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


sections: list[Section] = []
doc_freq: Counter[str] = Counter()
avg_doc_len = 0.0
files_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOP_WORDS]


def parse_markdown(path: Path) -> list[Section]:
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
            heading_stack = [(level, title) for level, title in heading_stack if level < current_level]
            heading_stack.append((current_level, current_heading))
        else:
            current_lines.append(line)

    flush()
    return parsed


def write_index_json(index_path: Path = INDEX_PATH) -> None:
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
    global doc_freq, avg_doc_len, files_indexed

    files_indexed = len({section.file for section in sections})
    doc_freq = Counter()
    for section in sections:
        doc_freq.update(set(section.tokens))
    avg_doc_len = sum(len(s.tokens) for s in sections) / len(sections) if sections else 0.0


def load_index_json(index_path: Path = INDEX_PATH) -> tuple[int, int]:
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

    heading_text = " ".join(section.heading_path).lower()
    if any(term in heading_text for term in query_tokens):
        score += 1.5

    return score


def search(query: str, k: int = 3) -> list[tuple[Section, float]]:
    query_tokens = tokenize(query)
    ranked = [
        (section, bm25_score(query_tokens, section))
        for section in sections
    ]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return [(section, score) for section, score in ranked[:k] if score > 0]
