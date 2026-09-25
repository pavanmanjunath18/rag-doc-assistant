"""Chroma vector store: persists chunks with their embeddings and searches by similarity."""

import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from rag.interfaces import Hit, StoredDocument

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

    def add(
        self, doc_id: str, source: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        """Store the chunks of one document version (`doc_id`) under the name `source`.

        Chunk IDs are `<doc_id>:<index>`, so storing the same version twice overwrites
        rather than duplicates.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        self._collection.upsert(
            ids=[f"{doc_id}:{i}" for i in range(len(chunks))],
            embeddings=embeddings,
            documents=chunks,
            metadatas=[
                {"doc_id": doc_id, "source": source, "chunk": i} for i in range(len(chunks))
            ],
        )
        logger.info("Stored %d chunks from %s (%s)", len(chunks), source, doc_id[:12])

    def find(self, doc_id: str) -> StoredDocument | None:
        """Return the stored document with this ID, or None if it isn't stored."""
        res = self._collection.get(where={"doc_id": doc_id}, include=["metadatas"])
        if not res["ids"]:
            return None
        return StoredDocument(source=str(res["metadatas"][0]["source"]), chunks=len(res["ids"]))

    def delete_stale_versions(self, source: str, current_doc_id: str) -> int:
        """Delete chunks stored under `source` that belong to any other document ID.

        Chunks without a `doc_id` (written before content hashing existed) count as stale.
        """
        res = self._collection.get(where={"source": source}, include=["metadatas"])
        stale = [
            chunk_id
            for chunk_id, meta in zip(res["ids"], res["metadatas"], strict=True)
            if meta.get("doc_id") != current_doc_id
        ]
        if stale:
            self._collection.delete(ids=stale)
            logger.info("Deleted %d stale chunks from %s", len(stale), source)
        return len(stale)

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
