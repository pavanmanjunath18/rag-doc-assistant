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
    store.add("id1", "10k.txt", chunks, EMBED(chunks))

    hits = store.search(EMBED(["senior notes debt"])[0], top_k=2)

    assert store.count() == 3
    assert [(h.source, h.chunk, h.text) for h in hits][0] == ("10k.txt", 0, chunks[0])
    assert len(hits) == 2
    assert hits[0].score > hits[1].score


def test_top_k_larger_than_store_returns_everything(store: ChromaStore) -> None:
    store.add("id1", "a.txt", ["only chunk"], EMBED(["only chunk"]))
    assert len(store.search(EMBED(["only chunk"])[0], top_k=10)) == 1


def test_data_persists_across_instances(tmp_path: Path) -> None:
    ChromaStore(tmp_path / "chroma", "test").add("id1", "a.txt", ["kept"], EMBED(["kept"]))
    assert ChromaStore(tmp_path / "chroma", "test").count() == 1


def test_mismatched_lengths_are_rejected(store: ChromaStore) -> None:
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        store.add("id1", "a.txt", ["one", "two"], EMBED(["one"]))


def test_find_returns_source_and_chunk_count(store: ChromaStore) -> None:
    store.add("id1", "a.txt", ["one", "two"], EMBED(["one", "two"]))
    found = store.find("id1")
    assert found is not None
    assert (found.source, found.chunks) == ("a.txt", 2)
    assert store.find("missing") is None


def test_delete_stale_versions_keeps_current_and_other_sources(store: ChromaStore) -> None:
    store.add("old", "a.txt", ["v1 one", "v1 two"], EMBED(["v1 one", "v1 two"]))
    store.add("new", "a.txt", ["v2"], EMBED(["v2"]))
    store.add("other", "b.txt", ["b"], EMBED(["b"]))
    assert store.delete_stale_versions("a.txt", "new") == 2
    assert store.find("old") is None
    assert store.find("new") is not None and store.find("other") is not None
    assert store.count() == 2
