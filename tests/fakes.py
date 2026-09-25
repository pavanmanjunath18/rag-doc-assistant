"""Fake embedder, store and generator, so tests exercise real pipeline and API code
without downloading or loading any model."""

import math
import re
import zlib

from rag.interfaces import Hit
from rag.prompts import Message


class FakeEmbedder:
    """Bag-of-words hashing into a small vector: texts sharing words get similar vectors."""

    dim = 64

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self.dim
            for word in re.findall(r"\w+", text.lower()):
                vec[zlib.crc32(word.encode()) % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in vec)) or 1.0
            vectors.append([x / norm for x in vec])
        return vectors


class InMemoryStore:
    """Vector store kept in a dict, keyed like the real store so re-adding overwrites."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[str, int, str, list[float]]] = {}
        self.searches = 0

    def add(self, source: str, chunks: list[str], embeddings: list[list[float]]) -> None:
        for i, (text, vec) in enumerate(zip(chunks, embeddings, strict=True)):
            self.rows[f"{source}:{i}"] = (source, i, text, vec)

    def search(self, embedding: list[float], top_k: int) -> list[Hit]:
        self.searches += 1
        scored = [
            Hit(text=text, source=source, chunk=i, score=_dot(embedding, vec))
            for source, i, text, vec in self.rows.values()
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
