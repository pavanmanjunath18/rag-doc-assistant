"""Embedding model backed by sentence-transformers (runs locally)."""

import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedder:
    """Embedder that loads a sentence-transformers model once, when constructed."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts as unit-length vectors, so cosine similarity is a plain dot product."""
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return vectors.tolist()
