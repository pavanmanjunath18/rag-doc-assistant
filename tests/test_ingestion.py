"""Idempotent ingestion: same bytes are a no-op, a changed file replaces its old chunks.

Each scenario runs twice: against the in-memory fake store (fast, checks pipeline logic)
and against the real Chroma store (checks the actual queries and deletes).
"""

from pathlib import Path

import pytest

from rag.interfaces import VectorStore
from rag.loader import document_id
from rag.pipeline import IngestStatus, RagPipeline
from rag.store import ChromaStore
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore

# Chunk size 60 so the long version splits into several chunks and the short one doesn't.
LONG_VERSION = (
    "INDEBTEDNESS senior notes 14.5 billion outstanding.\n\n"
    "REVOLVER a 3 billion credit facility, undrawn.\n\n"
    "CONTENT liabilities of about 5.7 billion."
)
SHORT_VERSION = "INDEBTEDNESS senior notes 16.0 billion outstanding."


@pytest.fixture(params=["memory", "chroma"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> VectorStore:
    if request.param == "memory":
        return InMemoryStore()
    return ChromaStore(tmp_path / "chroma", "test")


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def pipeline(store: VectorStore, embedder: FakeEmbedder) -> RagPipeline:
    return RagPipeline(embedder, store, FakeGenerator(), chunk_size=60, chunk_overlap=10, top_k=10)


def write(folder: Path, name: str, text: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    path.write_text(text, encoding="utf-8")
    return path


def all_text(pipeline: RagPipeline) -> str:
    return " ".join(hit.text for hit in pipeline.ask("senior notes billion").sources)


def test_document_id_depends_only_on_content(tmp_path: Path) -> None:
    a = write(tmp_path / "a", "report.txt", LONG_VERSION)
    b = write(tmp_path / "b", "renamed.txt", LONG_VERSION)
    c = write(tmp_path / "c", "report.txt", SHORT_VERSION)
    assert document_id(a) == document_id(b)
    assert document_id(a) != document_id(c)
    assert len(document_id(a)) == 64


def test_reuploading_the_same_file_is_a_no_op(
    tmp_path: Path, pipeline: RagPipeline, embedder: FakeEmbedder
) -> None:
    path = write(tmp_path, "report.txt", LONG_VERSION)
    first = pipeline.ingest(path)
    embedded_after_first = embedder.texts_embedded

    second = pipeline.ingest(path)

    assert first.status is IngestStatus.INDEXED
    assert second.status is IngestStatus.UNCHANGED
    assert second.doc_id == first.doc_id
    assert second.chunks == first.chunks > 1
    assert pipeline.chunk_count() == first.chunks
    assert embedder.texts_embedded == embedded_after_first  # nothing re-embedded


def test_same_content_under_another_name_is_not_duplicated(
    tmp_path: Path, pipeline: RagPipeline
) -> None:
    first = pipeline.ingest(write(tmp_path / "a", "report.txt", LONG_VERSION))
    second = pipeline.ingest(write(tmp_path / "b", "copy_of_report.txt", LONG_VERSION))
    assert second.status is IngestStatus.UNCHANGED
    assert second.source == "report.txt"
    assert pipeline.chunk_count() == first.chunks


def test_changed_file_with_same_name_replaces_old_chunks(
    tmp_path: Path, pipeline: RagPipeline
) -> None:
    old = pipeline.ingest(write(tmp_path / "v1", "report.txt", LONG_VERSION))
    new = pipeline.ingest(write(tmp_path / "v2", "report.txt", SHORT_VERSION))

    assert old.chunks > new.chunks == 1
    assert new.status is IngestStatus.REPLACED
    assert new.doc_id != old.doc_id
    assert pipeline.chunk_count() == 1
    text = all_text(pipeline)
    assert "16.0 billion" in text
    assert "14.5" not in text and "REVOLVER" not in text and "CONTENT" not in text


def test_replacing_one_file_leaves_other_files_alone(tmp_path: Path, pipeline: RagPipeline) -> None:
    other = pipeline.ingest(write(tmp_path / "x", "other.txt", "DIVIDENDS none paid ever."))
    pipeline.ingest(write(tmp_path / "v1", "report.txt", LONG_VERSION))
    pipeline.ingest(write(tmp_path / "v2", "report.txt", SHORT_VERSION))
    assert pipeline.chunk_count() == other.chunks + 1


def test_going_back_to_an_earlier_version_reindexes_it(
    tmp_path: Path, pipeline: RagPipeline
) -> None:
    v1 = pipeline.ingest(write(tmp_path / "v1", "report.txt", LONG_VERSION))
    pipeline.ingest(write(tmp_path / "v2", "report.txt", SHORT_VERSION))
    back = pipeline.ingest(write(tmp_path / "v3", "report.txt", LONG_VERSION))
    assert back.status is IngestStatus.REPLACED
    assert back.doc_id == v1.doc_id
    assert pipeline.chunk_count() == v1.chunks


def test_chunks_from_before_content_hashing_are_cleaned_up(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "chroma", "test")
    # Stage 1/2 rows: ids "<name>:<i>" and no doc_id in the metadata.
    store._collection.upsert(
        ids=["report.txt:0", "report.txt:1"],
        embeddings=FakeEmbedder().embed(["old one", "old two"]),
        documents=["old one", "old two"],
        metadatas=[{"source": "report.txt", "chunk": 0}, {"source": "report.txt", "chunk": 1}],
    )
    pipeline = RagPipeline(
        FakeEmbedder(), store, FakeGenerator(), chunk_size=60, chunk_overlap=10, top_k=10
    )

    result = pipeline.ingest(write(tmp_path, "report.txt", SHORT_VERSION))

    assert result.status is IngestStatus.REPLACED
    assert store.count() == 1
