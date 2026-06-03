import json
import os
import re
import shutil
from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


DOCS_DIR = Path(__file__).resolve().parents[3] / "docs"
INDEX_DIR = Path(__file__).resolve().parents[3] / ".kb" / "faiss_index"
EMBEDDING_MODEL = "text-embedding-3-small"
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=0,
    separators=["\n\n", "\n", ". ", " "],
)

vectorstore: FAISS | None = None
_embeddings = None
files_indexed = 0
sections_indexed = 0


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"


def get_embeddings():
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
            sections.append(Document(page_content=f"{current_heading}\n\n{content}" if current_heading else content, metadata={"source": source}))

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
    global vectorstore, files_indexed, sections_indexed

    md_files = sorted(docs_dir.glob("*.md"))
    all_sections: list[Document] = []
    for md_file in md_files:
        all_sections.extend(load_markdown_sections(md_file))

    chunks = splitter.split_documents(all_sections)
    vectorstore = FAISS.from_documents(chunks, get_embeddings())
    files_indexed = len(md_files)
    sections_indexed = len(chunks)
    save_vector_index()
    return files_indexed, sections_indexed


def save_vector_index(index_dir: Path = INDEX_DIR) -> None:
    if vectorstore is None:
        return
    if index_dir.exists():
        shutil.rmtree(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_dir))
    metadata = {"embedding_model": EMBEDDING_MODEL, "files_indexed": files_indexed, "sections_indexed": sections_indexed}
    (index_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))


def load_vector_index(index_dir: Path = INDEX_DIR) -> tuple[int, int]:
    global vectorstore, files_indexed, sections_indexed
    if not (index_dir / "index.faiss").exists() or not (index_dir / "index.pkl").exists():
        return 0, 0
    meta_path = index_dir / "metadata.json"
    if not meta_path.exists():
        return 0, 0
    meta = json.loads(meta_path.read_text())
    if meta.get("embedding_model") != EMBEDDING_MODEL:
        return 0, 0
    files_indexed = meta.get("files_indexed", 0)
    sections_indexed = meta.get("sections_indexed", 0)
    vectorstore = FAISS.load_local(str(index_dir), get_embeddings(), allow_dangerous_deserialization=True)
    return files_indexed, sections_indexed


def search(query: str, k: int = 3) -> list[tuple[Document, float]]:
    if vectorstore is None:
        return []
    return vectorstore.similarity_search_with_score(query, k=k)
