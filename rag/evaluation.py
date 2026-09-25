"""Evaluation helpers: load questions, grade answers, check retrieval.

Grading is rule-based on purpose: every verdict can be traced to a string in
eval/questions.json. See docs/decisions.md (D31) for why not an LLM judge.
"""

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from rag.interfaces import Hit

# Phrases that mean "I can't answer this". The first is the RAG prompt's own wording.
REFUSAL_MARKERS = (
    "does not contain this information",
    "does not contain",
    "doesn't contain",
    "no information",
    "not provided",
    "not mentioned",
    "not specified",
    "cannot find",
    "can't find",
    "unable to find",
    "not available",
    "can't answer",
    "cannot answer",
    "i don't have",
    "i do not have",
)


class Grade(StrEnum):
    CORRECT = "correct"
    PARTIAL = "partial"
    REFUSED = "refused"
    WRONG = "wrong"


@dataclass(frozen=True)
class EvalQuestion:
    id: str
    set: str
    question: str
    expected_source: str | None
    evidence: str | None
    key_facts: list[list[str]]
    forbidden: list[str] = field(default_factory=list)
    refusal_expected: bool = False
    refusal_ok: bool = False
    note: str = ""


def load_questions(path: Path) -> list[EvalQuestion]:
    """Read eval/questions.json into EvalQuestion objects."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**q) for q in data["questions"]]


def normalize(text: str) -> str:
    """Lowercase, drop markdown bold, unify quotes and dashes, collapse whitespace."""
    text = text.lower().replace("**", "")
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def is_refusal(answer: str) -> bool:
    """True if the answer declines to answer rather than giving one."""
    text = normalize(answer)
    return any(marker in text for marker in REFUSAL_MARKERS)


def grade(answer: str, q: EvalQuestion) -> Grade:
    """Grade one answer against the question's key facts, forbidden strings and refusal rules."""
    text = normalize(answer)
    refused = is_refusal(answer)

    if q.refusal_expected:
        return Grade.CORRECT if refused else Grade.WRONG
    if any(normalize(bad) in text for bad in q.forbidden):
        return Grade.WRONG

    matched = sum(1 for group in q.key_facts if any(normalize(alt) in text for alt in group))
    if matched == len(q.key_facts):
        return Grade.CORRECT
    if refused:
        return Grade.CORRECT if q.refusal_ok else Grade.REFUSED
    return Grade.PARTIAL if matched else Grade.WRONG


def retrieval_hit(hits: list[Hit], q: EvalQuestion, k: int) -> bool | None:
    """Did a top-k chunk from the expected file contain the evidence? None if not applicable."""
    if q.evidence is None or q.expected_source is None:
        return None
    evidence = normalize(q.evidence)
    return any(
        hit.source == q.expected_source and evidence in normalize(hit.text) for hit in hits[:k]
    )


def tally(grades: list[Grade]) -> dict[Grade, int]:
    """Count grades, including zeros, in a fixed order."""
    counts = Counter(grades)
    return {g: counts.get(g, 0) for g in Grade}
