"""High-level ingest and ask functions used by the CLI (and later the API)."""

from dataclasses import dataclass, field
from pathlib import Path

from rag.config import TOP_K
from rag.generator import generate
from rag.loader import chunk_text, load_text
from rag.prompts import build_baseline_messages, build_rag_messages
from rag.store import Hit, add_chunks, search

EMPTY_INDEX_ANSWER = "No documents indexed yet."


@dataclass(frozen=True)
class Answer:
    """A generated answer plus the chunks it was based on, most relevant first."""

    text: str
    sources: list[Hit] = field(default_factory=list)


def ingest(path: str | Path) -> int:
    """Load, chunk, embed and store one file. Returns the number of chunks stored."""
    path = Path(path)
    return add_chunks(chunk_text(load_text(path)), source=path.name)


def ask(question: str, use_rag: bool = True, top_k: int = TOP_K) -> Answer:
    """Answer a question with retrieval (default), or as the no-retrieval baseline."""
    if not use_rag:
        return Answer(text=generate(build_baseline_messages(question)))

    hits = search(question, top_k)
    if not hits:
        return Answer(text=EMPTY_INDEX_ANSWER)

    answer = generate(build_rag_messages(question, [hit.text for hit in hits]))
    return Answer(text=answer, sources=hits)
