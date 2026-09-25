"""Tests for the ask() flow with retrieval and generation replaced by fakes (no models loaded)."""

import pytest

from rag import pipeline
from rag.prompts import Message
from rag.store import Hit

HITS = [
    Hit(text="senior notes $14.5 billion", source="02_risk_factors.txt", chunk=7, score=0.61),
    Hit(text="revenues $45.2 billion", source="05_financial_results.txt", chunk=0, score=0.40),
    Hit(text="interest on senior notes", source="05_financial_results.txt", chunk=5, score=0.38),
]


@pytest.fixture
def seen_messages(monkeypatch: pytest.MonkeyPatch) -> list[list[Message]]:
    calls: list[list[Message]] = []

    def fake_generate(messages: list[Message]) -> str:
        calls.append(messages)
        return "fake answer"

    monkeypatch.setattr(pipeline, "generate", fake_generate)
    return calls


def test_sources_keep_rank_order_and_duplicates(
    monkeypatch: pytest.MonkeyPatch, seen_messages: list[list[Message]]
) -> None:
    monkeypatch.setattr(pipeline, "search", lambda question, top_k: HITS)
    answer = pipeline.ask("How much debt does Netflix have?")
    assert answer.text == "fake answer"
    assert answer.sources == HITS


def test_retrieved_text_reaches_the_prompt_in_order(
    monkeypatch: pytest.MonkeyPatch, seen_messages: list[list[Message]]
) -> None:
    monkeypatch.setattr(pipeline, "search", lambda question, top_k: HITS)
    pipeline.ask("How much debt does Netflix have?")
    prompt = seen_messages[0][0]["content"]
    positions = [prompt.index(hit.text) for hit in HITS]
    assert positions == sorted(positions)


def test_baseline_skips_retrieval(
    monkeypatch: pytest.MonkeyPatch, seen_messages: list[list[Message]]
) -> None:
    def fail_search(question: str, top_k: int) -> list[Hit]:
        raise AssertionError("baseline must not search")

    monkeypatch.setattr(pipeline, "search", fail_search)
    answer = pipeline.ask("What is X?", use_rag=False)
    assert answer.sources == []
    assert seen_messages == [[{"role": "user", "content": "What is X?"}]]


def test_empty_index_does_not_call_the_model(
    monkeypatch: pytest.MonkeyPatch, seen_messages: list[list[Message]]
) -> None:
    monkeypatch.setattr(pipeline, "search", lambda question, top_k: [])
    answer = pipeline.ask("Anything?")
    assert answer.text == pipeline.EMPTY_INDEX_ANSWER
    assert seen_messages == []
