"""Corpus loading and chunking.

Every chunk carries a stable `chunk_id` of the form `<document>#<n>`. That id
is the backbone of the whole harness: the agent is required to cite it, and
the citation grader checks the cited chunk actually contains the claim. An
answer that is correct but cites a chunk which does not support it is still
counted as ungrounded, because at scale that is indistinguishable from a
lucky guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document: str
    text: str
    ordinal: int

    def render(self) -> str:
        """How the chunk appears in the model's context window."""
        return f"[{self.chunk_id}]\n{self.text}"


@dataclass(frozen=True)
class Document:
    name: str
    text: str


def load_corpus(directory: str | Path) -> list[Document]:
    """Load every .md and .txt file in a directory, sorted by name.

    Sorted so that chunk ids are stable between runs on different machines;
    an unstable id would make two eval reports incomparable, which defeats
    the point of keeping them.
    """
    path = Path(directory)
    if not path.is_dir():
        raise FileNotFoundError(f"Corpus directory not found: {path}")

    documents: list[Document] = []
    for file in sorted(path.iterdir()):
        if file.suffix.lower() not in {".md", ".txt"}:
            continue
        text = file.read_text(encoding="utf-8").strip()
        if text:
            documents.append(Document(name=file.stem, text=text))

    if not documents:
        raise ValueError(
            f"No .md or .txt files with content found in {path}. "
            "The harness cannot evaluate grounding against an empty corpus."
        )
    return documents


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_document(
    document: Document, target_chars: int = 700, overlap_chars: int = 100
) -> list[Chunk]:
    """Split on paragraph boundaries, packing up to `target_chars`.

    Paragraph-aligned rather than fixed-width because a chunk cut mid-sentence
    produces a citation a human reviewer cannot verify by eye, and the point of
    the citation is that it can be checked.
    """
    if target_chars <= 0:
        raise ValueError("target_chars must be positive")
    if overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("overlap_chars must be >= 0 and < target_chars")

    paragraphs = _split_paragraphs(document.text)
    chunks: list[Chunk] = []
    buffer: list[str] = []
    buffer_len = 0

    def flush() -> None:
        nonlocal buffer, buffer_len
        if not buffer:
            return
        text = "\n\n".join(buffer)
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.name}#{ordinal}",
                document=document.name,
                text=text,
                ordinal=ordinal,
            )
        )
        if overlap_chars and len(text) > overlap_chars:
            tail = text[-overlap_chars:]
            buffer = [tail]
            buffer_len = len(tail)
        else:
            buffer = []
            buffer_len = 0

    for paragraph in paragraphs:
        if buffer_len and buffer_len + len(paragraph) > target_chars:
            flush()
        buffer.append(paragraph)
        buffer_len += len(paragraph)

    # Final flush without carrying overlap forward.
    if buffer:
        text = "\n\n".join(buffer)
        ordinal = len(chunks)
        chunks.append(
            Chunk(
                chunk_id=f"{document.name}#{ordinal}",
                document=document.name,
                text=text,
                ordinal=ordinal,
            )
        )

    return chunks


def build_chunks(
    directory: str | Path, target_chars: int = 700, overlap_chars: int = 100
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document in load_corpus(directory):
        chunks.extend(chunk_document(document, target_chars, overlap_chars))
    return chunks
