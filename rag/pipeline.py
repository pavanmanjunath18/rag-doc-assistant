"""Ingest documents and answer questions about them. Used by the CLI and the API."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

from rag.interfaces import Embedder, Generator, Hit, VectorStore
from rag.loader import chunk_text, load_text
from rag.prompts import build_baseline_messages, build_rag_messages

logger = logging.getLogger(__name__)

EMPTY_INDEX_ANSWER = "No documents indexed yet."


@dataclass(frozen=True)
class Answer:
    """A generated answer plus the chunks it was based on, most relevant first."""

    text: str
    sources: list[Hit] = field(default_factory=list)


class RagPipeline:
    """Retrieval-augmented Q&A over stored documents.

    The embedder, store and generator are passed in rather than created here, so tests can
    use fakes and other providers can be swapped in without changing this class.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        generator: Generator,
        *,
        chunk_size: int,
        chunk_overlap: int,
        top_k: int,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._generator = generator
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._top_k = top_k

    def ingest(self, path: Path, source: str | None = None) -> int:
        """Load, chunk, embed and store one file. Returns the number of chunks stored.

        `source` is the name shown in citations; it defaults to the file name.
        """
        name = source or path.name
        chunks = chunk_text(load_text(path), self._chunk_size, self._chunk_overlap)
        self._store.add(name, chunks, self._embedder.embed(chunks))
        logger.info("Indexed %d chunks from %s", len(chunks), name)
        return len(chunks)

    def ask(self, question: str, use_rag: bool = True, top_k: int | None = None) -> Answer:
        """Answer a question with retrieval (default), or as the no-retrieval baseline."""
        if not use_rag:
            return Answer(text=self._generator.generate(build_baseline_messages(question)))

        [query_vector] = self._embedder.embed([question])
        hits = self._store.search(query_vector, top_k or self._top_k)
        if not hits:
            return Answer(text=EMPTY_INDEX_ANSWER)

        messages = build_rag_messages(question, [hit.text for hit in hits])
        return Answer(text=self._generator.generate(messages), sources=hits)

    def chunk_count(self) -> int:
        """Return how many chunks are indexed."""
        return self._store.count()
