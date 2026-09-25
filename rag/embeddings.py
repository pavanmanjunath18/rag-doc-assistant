"""Embedding model wrapper. Loaded once and reused."""

import logging
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from rag.config import EMBED_MODEL

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    """Load the embedding model on first use and reuse it afterwards."""
    logger.info("Loading embedding model %s", EMBED_MODEL)
    return SentenceTransformer(EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Embed texts as unit-length vectors, so cosine similarity is a plain dot product."""
    vectors = get_embedder().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()
