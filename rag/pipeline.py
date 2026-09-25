"""Ingest documents and answer questions about them. Used by the CLI and the API."""

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rag.interfaces import Embedder, Generator, Hit, VectorStore
from rag.loader import chunk_text, document_id, load_text
from rag.prompts import build_baseline_messages, build_rag_messages

logger = logging.getLogger(__name__)

EMPTY_INDEX_ANSWER = "No documents indexed yet."


class IngestStatus(StrEnum):
    INDEXED = "indexed"  # new document
    REPLACED = "replaced"  # new version of a document with the same name; old chunks removed
    UNCHANGED = "unchanged"  # these exact bytes were already indexed; nothing was done


@dataclass(frozen=True)
class IngestResult:
    doc_id: str
    source: str
    chunks: int
    status: IngestStatus


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

    def ingest(self, path: Path, source: str | None = None) -> IngestResult:
        """Index one file, idempotently.

        The document ID is a hash of the file's bytes. If those bytes are already indexed
        (under any name), nothing is done. Otherwise the new version is stored first, and
        only then are older versions with the same name deleted, so a failure part-way
        leaves the previous version searchable.

        `source` is the name shown in citations; it defaults to the file name.
        """
        name = source or path.name
        doc_id = document_id(path)

        existing = self._store.find(doc_id)
        if existing is not None:
            logger.info("Skipped %s: same content already indexed as %s", name, existing.source)
            return IngestResult(doc_id, existing.source, existing.chunks, IngestStatus.UNCHANGED)

        chunks = chunk_text(load_text(path, name), self._chunk_size, self._chunk_overlap)
        self._store.add(doc_id, name, chunks, self._embedder.embed(chunks))
        removed = self._store.delete_stale_versions(name, doc_id)
        status = IngestStatus.REPLACED if removed else IngestStatus.INDEXED
        logger.info("Indexed %d chunks from %s (%s)", len(chunks), name, status)
        return IngestResult(doc_id, name, len(chunks), status)

    def retrieve(self, question: str, top_k: int | None = None) -> list[Hit]:
        """Return the chunks most relevant to `question`, best first."""
        [query_vector] = self._embedder.embed([question])
        return self._store.search(query_vector, top_k or self._top_k)

    def ask(self, question: str, use_rag: bool = True, top_k: int | None = None) -> Answer:
        """Answer a question with retrieval (default), or as the no-retrieval baseline."""
        if not use_rag:
            return Answer(text=self._generator.generate(build_baseline_messages(question)))

        hits = self.retrieve(question, top_k)
        if not hits:
            return Answer(text=EMPTY_INDEX_ANSWER)

        messages = build_rag_messages(question, [hit.text for hit in hits])
        return Answer(text=self._generator.generate(messages), sources=hits)

    def chunk_count(self) -> int:
        """Return how many chunks are indexed."""
        return self._store.count()
