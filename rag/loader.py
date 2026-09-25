"""Load documents (PDF or text) and split them into overlapping chunks."""

import hashlib
import logging
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES: frozenset[str] = frozenset({".pdf", ".txt", ".md"})
_WHITESPACE = (" ", "\n", "\t", "\r")


class DocumentError(ValueError):
    """A document can't be used: unsupported type, unreadable, or no extractable text."""


def document_id(path: str | Path) -> str:
    """Return the SHA-256 of the file's bytes: the same content always gets the same ID."""
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_text(path: str | Path, name: str | None = None) -> str:
    """Read a PDF or text file and return its text.

    Raises DocumentError for unsupported, unreadable or empty files, so a bad file fails
    loudly instead of being indexed as nothing. `name` is used in messages instead of the
    file's own name (uploads are read from temp files with meaningless names).
    """
    path = Path(path)
    name = name or path.name
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"Unsupported file type: {name}")

    if suffix == ".pdf":
        try:
            pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
        except PdfReadError as exc:
            raise DocumentError(f"Could not read PDF {name}: {exc}") from exc
        empty = sum(1 for page in pages if not page.strip())
        if empty:
            logger.warning("%s: %d of %d pages had no extractable text", name, empty, len(pages))
        text = "\n".join(pages)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if "\N{REPLACEMENT CHARACTER}" in text:
            logger.warning("%s: some bytes were not valid UTF-8 and were replaced", name)

    if not text.strip():
        raise DocumentError(f"No text extracted from {name}")
    return text


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
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
