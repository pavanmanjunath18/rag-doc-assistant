"""Tests for the evaluation grader and for the question data itself (no models needed)."""

import json
from pathlib import Path

import pytest

from rag.config import Settings
from rag.evaluation import (
    EvalQuestion,
    Grade,
    grade,
    is_refusal,
    load_questions,
    retrieval_hit,
    tally,
)
from rag.interfaces import Hit
from rag.loader import chunk_text
from rag.prompts import NO_ANSWER

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS = load_questions(ROOT / "eval/questions.json")
NOTEBOOK = json.loads((ROOT / "eval/notebook_answers.json").read_text(encoding="utf-8"))["answers"]
BY_ID = {q.id: q for q in QUESTIONS}


def question(**overrides: object) -> EvalQuestion:
    base = {
        "id": "q",
        "set": "held_out",
        "question": "How much debt?",
        "expected_source": "debt.txt",
        "evidence": "$14.5 billion of senior notes",
        "key_facts": [["14.5 billion"]],
    }
    return EvalQuestion(**{**base, **overrides})


# --- the question data ------------------------------------------------------------------


def test_ids_are_unique_and_sets_are_known() -> None:
    assert len({q.id for q in QUESTIONS}) == len(QUESTIONS)
    assert {q.set for q in QUESTIONS} == {"benchmark", "held_out"}


def test_benchmark_is_exactly_the_notebook_questions() -> None:
    benchmark = {q.id: q.question for q in QUESTIONS if q.set == "benchmark"}
    assert benchmark == {qid: a["question"] for qid, a in NOTEBOOK.items()}


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q.evidence], ids=lambda q: q.id)
def test_evidence_is_verbatim_and_fits_in_one_chunk(q: EvalQuestion) -> None:
    text = (ROOT / "data/corpus" / q.expected_source).read_text(encoding="utf-8")
    assert q.evidence in text
    s = Settings()
    assert any(q.evidence in chunk for chunk in chunk_text(text, s.chunk_size, s.chunk_overlap))


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q.key_facts], ids=lambda q: q.id)
def test_key_facts_appear_in_the_source(q: EvalQuestion) -> None:
    text = (ROOT / "data/corpus" / q.expected_source).read_text(encoding="utf-8").lower()
    for group in q.key_facts:
        if q.id == "revolver" and group is q.key_facts[1]:
            continue  # "none borrowed" is phrased freely by the model
        assert any(alt.lower() in text for alt in group), group


def test_unanswerable_question_is_really_unanswerable() -> None:
    corpus = " ".join(p.read_text().lower() for p in (ROOT / "data/corpus").glob("*.txt"))
    assert "free cash flow" not in corpus


# --- grading rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("Netflix had $14.5 billion in senior notes.", Grade.CORRECT),
        ("**$14.5 Billion** of notes", Grade.CORRECT),
        ("About 14 billion.", Grade.WRONG),
        (NO_ANSWER, Grade.REFUSED),
    ],
)
def test_single_fact(answer: str, expected: Grade) -> None:
    assert grade(answer, question()) is expected


def test_forbidden_figure_is_wrong_even_with_the_right_fact() -> None:
    q = question(forbidden=["17.5"])
    assert grade("$14.5 billion + $3 billion = $17.5 billion", q) is Grade.WRONG


def test_some_but_not_all_facts_is_partial() -> None:
    q = question(key_facts=[["3 billion"], ["not borrowed", "none"]])
    assert grade("A $3 billion facility.", q) is Grade.PARTIAL
    assert grade("A $3 billion facility; none has been borrowed.", q) is Grade.CORRECT


def test_refusal_expected() -> None:
    q = question(key_facts=[], refusal_expected=True, evidence=None, expected_source=None)
    assert grade(NO_ANSWER, q) is Grade.CORRECT
    assert grade("Free cash flow was $2 billion.", q) is Grade.WRONG


def test_refusal_ok_counts_a_refusal_as_correct() -> None:
    q = question(refusal_ok=True, forbidden=["17.5"])
    assert grade(NO_ANSWER, q) is Grade.CORRECT
    assert grade("Total debt is $17.5 billion.", q) is Grade.WRONG


@pytest.mark.parametrize(
    "answer",
    [NO_ANSWER, "The context doesn't contain that.", "I'm sorry, I can't answer this question."],
)
def test_refusal_markers(answer: str) -> None:
    assert is_refusal(answer)


def test_normal_answers_are_not_refusals() -> None:
    assert not is_refusal("Netflix has never declared or paid any cash dividends.")


def test_tally_includes_zero_counts() -> None:
    assert tally([Grade.CORRECT, Grade.CORRECT, Grade.WRONG]) == {
        Grade.CORRECT: 2,
        Grade.PARTIAL: 0,
        Grade.REFUSED: 0,
        Grade.WRONG: 1,
    }


# --- retrieval hit@k --------------------------------------------------------------------


def hit(source: str, text: str) -> Hit:
    return Hit(text=text, source=source, chunk=0, score=0.5)


def test_hit_at_k_needs_right_file_and_evidence_within_k() -> None:
    q = question()
    hits = [
        hit("other.txt", "$14.5 billion of senior notes"),  # right text, wrong file
        hit("debt.txt", "unrelated"),
        hit("debt.txt", "we had $14.5  billion of senior notes"),  # whitespace differs
    ]
    assert retrieval_hit(hits, q, 1) is False
    assert retrieval_hit(hits, q, 2) is False
    assert retrieval_hit(hits, q, 3) is True


def test_hit_is_none_for_unanswerable_questions() -> None:
    q = question(evidence=None, expected_source=None)
    assert retrieval_hit([hit("debt.txt", "anything")], q, 3) is None


# --- the grader applied to the notebook's own answers -----------------------------------


def test_notebook_answers_rescored() -> None:
    rag = {qid: grade(a["rag"], BY_ID[qid]) for qid, a in NOTEBOOK.items()}
    assert rag == {
        "debt": Grade.WRONG,  # the $17.5B sum
        "revenue": Grade.CORRECT,
        "dividends": Grade.CORRECT,
        "employees": Grade.CORRECT,
        "cybersecurity": Grade.WRONG,  # names the Senior Director, not the Audit Committee
        "stock_split": Grade.CORRECT,
        "wbd_fee": Grade.CORRECT,
        "operating_margin": Grade.WRONG,  # "up three points", no 29.5%
    }


def test_summing_that_was_cut_off_still_counts_as_wrong() -> None:
    # Real RAG answer from the first Stage 5 run: the token limit stopped it before the total,
    # so the forbidden sums never appeared and the grader wrongly marked it correct.
    truncated = (
        "Adding these components together gives us:\n\n"
        "\\[ \\text{Total Debt} = \\$14.5\\, \\text{billion} + \\$3\\, \\text{billion} + \\$42.2"
    )
    assert grade(truncated, BY_ID["total_debt_trap"]) is Grade.WRONG
    assert grade("Senior notes: $14.5 billion.", BY_ID["total_debt_trap"]) is Grade.CORRECT
