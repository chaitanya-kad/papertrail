"""Document loading, chunking, and tokenization for PaperTrail."""

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass(frozen=True)
class Document:
    """A text unit extracted from a source file (a PDF page, or a text file)."""

    source: str
    text: str
    page: int | None = None


@dataclass(frozen=True)
class Passage:
    """A searchable window of text with a stable source reference."""

    passage_id: str
    source: str
    number: int
    text: str
    page: int | None = None


def load_documents(directory: Path, *, source_prefix: str = "") -> list[Document]:
    """Load Markdown, plain text, and PDF files recursively from a directory."""
    if not directory.exists():
        return []

    documents: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        relative_source = path.relative_to(directory).as_posix()
        if source_prefix:
            relative_source = f"{source_prefix.rstrip('/')}/{relative_source}"
        suffix = path.suffix.lower()

        if suffix in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(Document(source=relative_source, text=text))
        elif suffix == ".pdf":
            documents.extend(_read_pdf(path, relative_source))
    return documents


def _read_pdf(path: Path, source: str) -> list[Document]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "PDF support needs pypdf. Install it with `python -m pip install -e '.[pdf]'`."
        ) from exc

    pages: list[Document] = []
    for page_number, page in enumerate(PdfReader(str(path)).pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append(Document(source=source, text=text, page=page_number))
    return pages


def split_into_passages(
    documents: list[Document], *, max_words: int = 180, overlap: int = 30
) -> list[Passage]:
    """Split document text into overlapping word windows while preserving source pages."""
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    if overlap < 0 or overlap >= max_words:
        raise ValueError("overlap must be between 0 and max_words - 1")

    passages: list[Passage] = []
    step = max_words - overlap
    source_counts: dict[str, int] = {}

    for document in documents:
        words = document.text.split()
        if not words:
            continue
        for start in range(0, len(words), step):
            window = words[start : start + max_words]
            if not window:
                break
            number = source_counts.get(document.source, 0) + 1
            source_counts[document.source] = number
            page_suffix = f"-p{document.page}" if document.page is not None else ""
            passage_id = f"{document.source}{page_suffix}-c{number}"
            passages.append(
                Passage(
                    passage_id=passage_id,
                    source=document.source,
                    number=number,
                    text=" ".join(window),
                    page=document.page,
                )
            )
            if start + max_words >= len(words):
                break
    return passages


def tokenize(text: str) -> list[str]:
    """Lowercase and extract words, preserving simple contractions and numbers."""
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())
