"""Chroma vector store: persists chunks with their embeddings and searches by similarity."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.interfaces import Hit

logger = logging.getLogger(__name__)


class ChromaStore:
    """Vector store backed by a local, on-disk Chroma collection using cosine distance."""

    def __init__(self, path: Path, collection: str) -> None:
        client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}
        )

    def add(self, source: str, chunks: list[str], embeddings: list[list[float]]) -> None:
        """Store the chunks of one source document with their embeddings."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self._collection.upsert(
            ids=[f"{source}:{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[{"source": source, "chunk": i} for i in range(len(chunks))],
        )
        logger.info("Stored %d chunks from %s", len(chunks), source)

    def search(self, embedding: list[float], top_k: int) -> list[Hit]:
        """Return up to `top_k` chunks closest to `embedding`, most relevant first."""
        if self.count() == 0:
            return []
        res = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
        return [
            Hit(text=doc, source=str(meta["source"]), chunk=int(meta["chunk"]), score=1 - dist)
            for doc, meta, dist in rows
        ]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()
