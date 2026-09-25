"""Load documents (PDF or text) and split them into overlapping chunks."""

import logging
from pathlib import Path

from pypdf import PdfReader

from rag.config import CHUNK_OVERLAP, CHUNK_SIZE

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt", ".md"})
_WHITESPACE = (" ", "\n", "\t", "\r")


def load_text(path: str | Path) -> str:
    """Read a PDF or text file and return its text.

    Raises ValueError for unsupported file types or when no text can be extracted,
    so an unreadable file fails loudly instead of being indexed as nothing.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {path.name}")

    if suffix == ".pdf":
        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        empty = sum(1 for page in pages if not page.strip())
        if empty:
            logger.warning(
                "%s: %d of %d pages had no extractable text", path.name, empty, len(pages)
            )
        text = "\n".join(pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\N{REPLACEMENT CHARACTER}" in text:
            logger.warning("%s: some bytes were not valid UTF-8 and were replaced", path.name)

    if not text.strip():
        raise ValueError(f"No text extracted from {path.name}")
    return text


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of at most `size` characters, overlapping by about `overlap`.

    Chunk edges are moved to whitespace so words and figures like "$45,183,036"
    are never cut in half. Line breaks are kept, so section headings stay readable.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    chunks: list[str] = []
    start, length = 0, len(text)
    while start < length:
        end = min(start + size, length)
        if end < length:
            cut = _last_whitespace(text, start, end)
            if cut > start:
                end = cut

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= length:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        elif not text[next_start - 1].isspace():
            space = _first_whitespace(text, next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start
    return chunks


def _last_whitespace(text: str, lo: int, hi: int) -> int:
    return max(text.rfind(ch, lo, hi) for ch in _WHITESPACE)


def _first_whitespace(text: str, lo: int, hi: int) -> int:
    positions = [pos for ch in _WHITESPACE if (pos := text.find(ch, lo, hi)) != -1]
    return min(positions, default=-1)
