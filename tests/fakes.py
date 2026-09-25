"""Fake embedder, store and generator, so tests exercise real pipeline and API code
without downloading or loading any model."""

import math
import re
import zlib
from dataclasses import dataclass

from rag.interfaces import Hit, StoredDocument
from rag.prompts import Message


class FakeEmbedder:
    """Bag-of-words hashing into a small vector: texts sharing words get similar vectors."""

    dim = 64

    def __init__(self) -> None:
        self.texts_embedded = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.texts_embedded += len(texts)
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


@dataclass(frozen=True)
class Row:
    doc_id: str
    source: str
    chunk: int
    text: str
    vector: list[float]


class InMemoryStore:
    """Vector store kept in a dict, keyed like the real store (`doc_id:index`)."""

    def __init__(self) -> None:
        self.rows: dict[str, Row] = {}
        self.searches = 0

    def add(
        self, doc_id: str, source: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        for i, (text, vec) in enumerate(zip(chunks, embeddings, strict=True)):
            self.rows[f"{doc_id}:{i}"] = Row(doc_id, source, i, text, vec)

    def find(self, doc_id: str) -> StoredDocument | None:
        rows = [row for row in self.rows.values() if row.doc_id == doc_id]
        return StoredDocument(rows[0].source, len(rows)) if rows else None

    def delete_stale_versions(self, source: str, current_doc_id: str) -> int:
        stale = [
            key
            for key, row in self.rows.items()
            if row.source == source and row.doc_id != current_doc_id
        ]
        for key in stale:
            del self.rows[key]
        return len(stale)

    def search(self, embedding: list[float], top_k: int) -> list[Hit]:
        self.searches += 1
        scored = [
            Hit(
                text=row.text, source=row.source, chunk=row.chunk, score=_dot(embedding, row.vector)
            )
            for row in self.rows.values()
        ]
        return sorted(scored, key=lambda hit: hit.score, reverse=True)[:top_k]

    def count(self) -> int:
        return len(self.rows)


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class FakeGenerator:
    """Records every prompt it receives and returns a fixed answer."""

    answer = "fake answer"

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.calls.append(messages)
        return self.answer
