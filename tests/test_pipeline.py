"""Tests for RagPipeline using fake components (no models loaded)."""

from pathlib import Path

from rag.pipeline import EMPTY_INDEX_ANSWER, RagPipeline
from tests.fakes import FakeGenerator, InMemoryStore

DEBT = "INDEBTEDNESS\nNetflix had $14.5 billion of senior notes outstanding."
REVENUE = "REVENUES\nTotal revenues were $45.2 billion, up 16 percent."
DIVIDENDS = "DIVIDEND POLICY\nNetflix has never paid cash dividends on its stock."


def write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_stores_chunks_under_the_source_name(
    tmp_path: Path, pipeline: RagPipeline, store: InMemoryStore
) -> None:
    assert pipeline.ingest(write(tmp_path, "tmp123.txt", DEBT), source="risk.txt") == 1
    assert [row[0] for row in store.rows.values()] == ["risk.txt"]
    assert pipeline.chunk_count() == 1


def test_most_relevant_chunk_is_first_and_reaches_the_prompt_in_order(
    tmp_path: Path, pipeline: RagPipeline, generator: FakeGenerator
) -> None:
    for name, text in [("revenue.txt", REVENUE), ("debt.txt", DEBT), ("div.txt", DIVIDENDS)]:
        pipeline.ingest(write(tmp_path, name, text))

    answer = pipeline.ask("How much in senior notes did Netflix have outstanding?")

    assert answer.text == FakeGenerator.answer
    assert answer.sources[0].source == "debt.txt"
    assert [hit.score for hit in answer.sources] == sorted(
        (hit.score for hit in answer.sources), reverse=True
    )
    prompt = generator.calls[0][0]["content"]
    positions = [prompt.index(hit.text) for hit in answer.sources]
    assert positions == sorted(positions)


def test_top_k_limits_sources(tmp_path: Path, pipeline: RagPipeline) -> None:
    for name, text in [("revenue.txt", REVENUE), ("debt.txt", DEBT), ("div.txt", DIVIDENDS)]:
        pipeline.ingest(write(tmp_path, name, text))
    assert len(pipeline.ask("Netflix", top_k=1).sources) == 1


def test_baseline_skips_retrieval(
    pipeline: RagPipeline, store: InMemoryStore, generator: FakeGenerator
) -> None:
    answer = pipeline.ask("What is X?", use_rag=False)
    assert answer.sources == []
    assert store.searches == 0
    assert generator.calls == [[{"role": "user", "content": "What is X?"}]]


def test_empty_index_does_not_call_the_model(
    pipeline: RagPipeline, generator: FakeGenerator
) -> None:
    assert pipeline.ask("Anything?").text == EMPTY_INDEX_ANSWER
    assert generator.calls == []
