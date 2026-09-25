"""Prompt templates, kept separate from the model so they can be tested without loading it.

The RAG template comes from the proof notebook, with two changes:
- "Netflix's annual report" became "the provided documents", since users upload any file.
- One guardrail line was added after the notebook's model summed senior notes and an
  undrawn credit facility into a "total debt" figure that the document never states.
  Its wording matters: a stricter version made the model refuse answerable questions
  (see docs/decisions.md, D5).
"""

Message = dict[str, str]

NO_ANSWER = "The context does not contain this information."

RAG_PROMPT_TEMPLATE = f"""You are a helpful assistant answering questions about the provided documents.
Use the context below to answer. The answer may be phrased differently in the context than in the question — for example, "debt" may appear as "notes", "indebtedness", or "liabilities".
Only if you find nothing relevant at all, say "{NO_ANSWER}"
Quote each figure exactly as it appears in the context, and do not add figures together to make a new total.

CONTEXT:
{{context}}

QUESTION: {{question}}

ANSWER:"""  # noqa: E501


def build_rag_messages(question: str, contexts: list[str]) -> list[Message]:
    """Build the chat messages for a retrieval-augmented answer.

    Contexts are joined in rank order (most relevant first), separated by blank lines.
    """
    prompt = RAG_PROMPT_TEMPLATE.format(context="\n\n".join(contexts), question=question)
    return [{"role": "user", "content": prompt}]


def build_baseline_messages(question: str) -> list[Message]:
    """Build the chat messages for the no-retrieval baseline: the bare question, nothing else."""
    return [{"role": "user", "content": question}]
