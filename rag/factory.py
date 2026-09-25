"""Wire the pipeline together with the real local models and store."""

from rag.config import Settings
from rag.embeddings import SentenceTransformerEmbedder
from rag.generator import HuggingFaceGenerator
from rag.pipeline import RagPipeline
from rag.store import ChromaStore


def build_pipeline(settings: Settings) -> RagPipeline:
    """Build the pipeline from settings. Loads both models, which takes a few seconds."""
    return RagPipeline(
        embedder=SentenceTransformerEmbedder(settings.embed_model),
        store=ChromaStore(settings.chroma_dir, settings.collection),
        generator=HuggingFaceGenerator(settings.gen_model, settings.max_new_tokens),
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )
