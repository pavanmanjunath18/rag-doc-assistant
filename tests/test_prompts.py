"""Tests for prompt construction (no models needed)."""

from rag.prompts import NO_ANSWER, build_baseline_messages, build_rag_messages


def rag_prompt(
    question: str = "How much debt does Netflix have?", contexts: list[str] | None = None
) -> str:
    messages = build_rag_messages(question, contexts or ["chunk A", "chunk B"])
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    return messages[0]["content"]


def test_rag_prompt_has_no_combining_guardrail() -> None:
    assert "do not add figures together to make a new total" in rag_prompt()


def test_rag_prompt_avoids_guardrail_wording_that_caused_refusals() -> None:
    # "do not give a total unless the context states one" made the model refuse
    # "How much debt does Netflix have?" even with the answer in context (docs/decisions.md, D5).
    assert "unless the context states one" not in rag_prompt()


def test_rag_prompt_keeps_synonym_hint_from_notebook() -> None:
    assert '"debt" may appear as "notes", "indebtedness", or "liabilities"' in rag_prompt()


def test_rag_prompt_uses_soft_refusal() -> None:
    prompt = rag_prompt()
    assert f'Only if you find nothing relevant at all, say "{NO_ANSWER}"' in prompt
    assert "ONLY" not in prompt


def test_rag_prompt_is_not_tied_to_one_company() -> None:
    assert "about the provided documents" in rag_prompt()
    assert "Netflix's annual report" not in rag_prompt()


def test_contexts_keep_rank_order() -> None:
    prompt = rag_prompt(contexts=["first chunk", "second chunk", "third chunk"])
    assert "CONTEXT:\nfirst chunk\n\nsecond chunk\n\nthird chunk\n\nQUESTION:" in prompt


def test_question_is_last_before_answer() -> None:
    assert rag_prompt(question="What is X?").endswith("QUESTION: What is X?\n\nANSWER:")


def test_baseline_is_bare_question() -> None:
    assert build_baseline_messages("What is X?") == [{"role": "user", "content": "What is X?"}]
