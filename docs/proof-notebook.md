# The proof notebook (RAG_project.ipynb)

Before this repo there was a Colab notebook that tested the core idea on five sections of
Netflix's FY2025 10-K. It is the source of truth for the settings in this repo. This page
records what it did and what it showed, so the notebook itself isn't needed to follow along.

## Setup

| Part | Choice |
|---|---|
| Documents | 5 hand-prepared `.txt` files (business, risk factors, cybersecurity, stock, financial results), now in `data/corpus/` |
| Chunking | Plain character slices, 800 characters, 150 overlap. 37 chunks in total |
| Embeddings | `all-MiniLM-L6-v2` (384 dimensions) |
| Vector store | Chroma, in memory, default settings |
| Retrieval | Top 3 chunks |
| Generator | `Qwen/Qwen2.5-1.5B-Instruct`, fp16 on a Colab GPU (`cuda:0`), 200 new tokens, Qwen's default sampling |
| Baseline | Same model, the bare question, no retrieval |

## Benchmark: 8 questions, baseline vs RAG

Grading below was done while reviewing the notebook output against the corpus text. It will
be formalised (expected answers, automatic scoring) in Stage 5.

| # | Question | Baseline (no retrieval) | RAG |
|---|---|---|---|
| 1 | How much debt does Netflix have? | Wrong: says debt is "not publicly disclosed" | **Wrong**: adds $14.5B notes + $3B revolver = "$17.5B". The revolver was undrawn and no total is stated |
| 2 | What was Netflix's total revenue in 2025? | Refuses (no data) | Correct: $45.2B |
| 3 | Does Netflix pay dividends? | Partly right ("no regular dividends") then invents "dividend equivalent units" | Correct |
| 4 | How many employees does Netflix have? | Wrong: about 75,000 | Correct: about 16,000 |
| 5 | Who oversees cybersecurity risk at Netflix? | Invented (names Reed Hastings, a CISO, etc.) | **Partial**: names the Senior Director who leads the program; the text says the Audit Committee oversees cybersecurity risk |
| 6 | What happened with the Netflix stock split? | Refuses, calls it "political" | Correct: 10-for-1, completed Nov 14, 2025 |
| 7 | What is the WBD transaction termination fee? | Refuses, calls it "phishing" | Correct: $5.8B |
| 8 | What is Netflix's operating margin? | No figure | **Partial**: "up about 3 points", misses 29.5% |

Summary: baseline 0/8 fully correct; RAG 5/8 correct, 2 partial, 1 wrong.

Latency on the Colab GPU: baseline 1.2 to 10.4 s, RAG 1.2 to 13.8 s. Most RAG answers were
faster than the baseline because grounded answers are shorter.

## Failures visible in the notebook

**Vocabulary mismatch plus a strict prompt (documented in the notebook, cell 11).**
"How much debt does Netflix have?" retrieved the right chunk ($14.5B senior notes) but the
model said the context didn't contain the answer. The question says "debt", the text says
"indebtedness" and "senior notes", and the prompt told the model to refuse unless the context
had the answer. Fix: soften the refusal ("only if you find nothing relevant at all") and add
a synonym hint to the prompt.

**Over-synthesis of debt figures (cells 12 and 13).** After that fix, the model answered but
added the senior notes and the undrawn revolving credit facility into a "total debt" that the
document never states. The notebook did not fix this. This repo adds a guardrail line to the
prompt (see [decisions.md](decisions.md), D5). When this repo re-ran the notebook's exact
prompt on a Mac (float16, 1 greedy + 5 sampled runs), the $17.5B sum did not reproduce, so
whether the guardrail prevents it is still open and is measured in Stage 5. Stage 5 result: the benchmark debt question is fixed, but a held-out question asking for total debt including the WBD financing still made the model start summing (see [stage 5 notes](stages/stage-5-evaluation.md)).

**Stale cells.** The cells ran out of order (execution counts 7, 25, 8, ..., 22, 23, 24, 19),
and cell 9 shows the revised prompt in its code but the old refusal in its output. Output on
screen did not match the code above it. This is one reason the evaluation in this repo is a
script that runs top to bottom in a fresh process.

## Not covered by the notebook

The PDF extraction failure and the CPU-vs-GPU runtime failure happened during the proof
project but are not recorded in the notebook. They will be written up from memory in Stage 8.
