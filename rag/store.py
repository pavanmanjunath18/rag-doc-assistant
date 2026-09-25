"""Chroma vector store: upsert chunks and search by similarity."""

import logging
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from rag.config import CHROMA_DIR, COLLECTION
from rag.embeddings import embed

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is cosine similarity (higher is more relevant)."""

    text: str
    source: str
    chunk: int
    score: float


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Open (or create) the persistent Chroma collection, using cosine distance."""
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


def add_chunks(chunks: list[str], source: str) -> int:
    """Embed and store chunks for one source file. Returns the number of chunks stored."""
    ids = [f"{source}:{i}" for i in range(len(chunks))]
    metadatas = [{"source": source, "chunk": i} for i in range(len(chunks))]
    get_collection().upsert(
        ids=ids, embeddings=embed(chunks), documents=chunks, metadatas=metadatas
    )
    logger.info("Stored %d chunks from %s", len(chunks), source)
    return len(chunks)


def search(query: str, top_k: int) -> list[Hit]:
    """Return the `top_k` chunks closest to `query`, most relevant first."""
    res = get_collection().query(query_embeddings=embed([query]), n_results=top_k)
    rows = zip(res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True)
    return [
        Hit(text=doc, source=str(meta["source"]), chunk=int(meta["chunk"]), score=1 - dist)
        for doc, meta, dist in rows
    ]
