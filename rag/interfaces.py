"""Small interfaces between the pipeline and the things it depends on.

The pipeline only talks to these, so implementations can be swapped: tests use in-memory
fakes, and a cloud version could plug in a hosted model or vector database without
touching the pipeline.
"""

from dataclasses import dataclass
from typing import Protocol

from rag.prompts import Message


@dataclass(frozen=True)
class Hit:
    """One retrieved chunk. `score` is cosine similarity (higher is more relevant)."""

    text: str
    source: str
    chunk: int
    score: float


@dataclass(frozen=True)
class StoredDocument:
    """A document already in the store: the name it's cited under and its chunk count."""

    source: str
    chunks: int


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one unit-length vector per text."""
        ...


class VectorStore(Protocol):
    def add(
        self, doc_id: str, source: str, chunks: list[str], embeddings: list[list[float]]
    ) -> None:
        """Store the chunks of one document version (`doc_id`) under the name `source`."""
        ...

    def find(self, doc_id: str) -> StoredDocument | None:
        """Return the stored document with this ID, or None if it isn't stored."""
        ...

    def delete_stale_versions(self, source: str, current_doc_id: str) -> int:
        """Delete chunks stored under `source` that belong to any other document ID.

        Returns the number of chunks deleted.
        """
        ...

    def search(self, embedding: list[float], top_k: int) -> list[Hit]:
        """Return up to `top_k` chunks closest to `embedding`, most relevant first."""
        ...

    def count(self) -> int:
        """Return the number of stored chunks."""
        ...


class Generator(Protocol):
    def generate(self, messages: list[Message]) -> str:
        """Return the model's reply to a list of chat messages."""
        ...
