"""Tests for the real Chroma store, fed with fake embeddings (no model needed)."""

from pathlib import Path

import pytest

from rag.store import ChromaStore
from tests.fakes import FakeEmbedder

EMBED = FakeEmbedder().embed


@pytest.fixture
def store(tmp_path: Path) -> ChromaStore:
    return ChromaStore(tmp_path / "chroma", "test")


def test_empty_store_returns_no_hits(store: ChromaStore) -> None:
    assert store.count() == 0
    assert store.search(EMBED(["anything"])[0], top_k=3) == []


def test_search_ranks_the_closest_chunk_first(store: ChromaStore) -> None:
    chunks = ["senior notes outstanding debt", "total revenues grew", "never paid dividends"]
    store.add("10k.txt", chunks, EMBED(chunks))

    hits = store.search(EMBED(["senior notes debt"])[0], top_k=2)

    assert store.count() == 3
    assert [(h.source, h.chunk, h.text) for h in hits][0] == ("10k.txt", 0, chunks[0])
    assert len(hits) == 2
    assert hits[0].score > hits[1].score


def test_top_k_larger_than_store_returns_everything(store: ChromaStore) -> None:
    store.add("a.txt", ["only chunk"], EMBED(["only chunk"]))
    assert len(store.search(EMBED(["only chunk"])[0], top_k=10)) == 1


def test_data_persists_across_instances(tmp_path: Path) -> None:
    ChromaStore(tmp_path / "chroma", "test").add("a.txt", ["kept"], EMBED(["kept"]))
    assert ChromaStore(tmp_path / "chroma", "test").count() == 1


def test_mismatched_lengths_are_rejected(store: ChromaStore) -> None:
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.add("a.txt", ["one", "two"], EMBED(["one"]))
