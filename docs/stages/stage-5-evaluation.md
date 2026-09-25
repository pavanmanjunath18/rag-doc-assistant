# Stage 5: evaluation harness

## Goal

Measure the system instead of eyeballing it: baseline (no retrieval) vs RAG, retrieval
hit@k, and answer correctness, on the notebook's 8 questions. Reproduce the notebook's
findings where possible and say plainly where results differ.

## What was built

| File | What it is |
|---|---|
| `eval/questions.json` | 17 questions: the notebook's 8 (`benchmark`) and 9 new ones (`held_out`). Each has the expected source file, a verbatim evidence sentence, key facts, forbidden strings, and refusal rules |
| `eval/notebook_answers.json` | The notebook's recorded baseline and RAG answers, extracted from the `.ipynb` file (not retyped) |
| `rag/evaluation.py` | Loading, grading (`correct` / `partial` / `refused` / `wrong`), refusal detection, hit@k |
| `eval/run_eval.py` | Builds a fresh index, asks every question both ways, re-scores the notebook's answers, writes the results |
| `eval/results.md`, `eval/results.json` | The latest results (generated, not hand-edited) |
| `tests/test_evaluation.py` | Grader rules, plus checks on the question data itself |
| `rag/pipeline.py` | New `retrieve()` method, so retrieval can be measured separately from generation |

## How grading works

For each answer (full rules in D31):
1. If the question needs a refusal (the free-cash-flow question), only a refusal is correct.
2. Any forbidden string (for example "17.5" or "+ $" on the debt questions) makes it wrong.
3. If every key-fact group is matched, it's correct. Some groups matched: partial.
4. A refusal is `refused`, or correct if the question allows it (the total-debt trap).

Retrieval hit@k: did a chunk *from the expected file* containing the *evidence sentence* rank
in the top k? The tests check that each evidence sentence exists verbatim and fits inside a
single chunk, so a miss is a real retrieval miss and not a data error.

## Results

From [eval/results.md](../../eval/results.md): Qwen2.5-1.5B in float16 on an Apple GPU,
greedy decoding, top 3 chunks.

| Set | Mode | Correct | Refused | Wrong |
|---|---|---|---|---|
| Benchmark (8) | No retrieval | 1 | 3 | 4 |
| Benchmark (8) | **RAG** | **6** | 0 | 2 |
| Benchmark (8) | Notebook RAG, re-scored | 5 | 0 | 3 |
| Held-out (9) | No retrieval | 2 | 3 | 4 |
| Held-out (9) | **RAG** | **4** | 3 | 2 |

| Set | hit@1 | hit@3 | hit@5 |
|---|---|---|---|
| Benchmark | 6/8 | 6/8 | 7/8 |
| Held-out (8 answerable) | 3/8 | 5/8 | 5/8 |

Median time per answer with the model loaded: RAG 1.9 s, baseline 4.0 s.

## What reproduced, and what didn't

**Reproduced:**
- Retrieval is what makes the model useful. Without it the model invents numbers (debt "about
  $3 billion as of 2021"), refuses for odd reasons, or says it has no data. With it, 6 of 8
  benchmark answers are correct.
- Grounded answers are shorter and faster than ungrounded ones.

**Better than the notebook:**
- The debt question is now answered correctly ($14.5B of senior notes, no sum). The notebook
  answered $17.5B. Re-scored with the same grader, the notebook RAG gets 5 of 8 and this repo 6 of 8.

**Different from my earlier hand grading:** In Stage 1b I graded the notebook's cybersecurity
and operating-margin answers as "partial". The automatic grader calls them wrong: one names
the wrong body and the other never gives 29.5%. So the notebook is 5 correct and 3 wrong here,
not 5, 2 and 1. The grader is stricter, and I'm keeping it strict.

**New findings the notebook couldn't show:**
1. **Retrieval is the main bottleneck, not generation.** On the held-out set, every refusal
   (technology spend, UCAN revenue, interest expense) comes from a question where the right
   chunk was *not* in the top 5. The model saw the wrong context and correctly declined
   rather than inventing a number, which is the failure mode you want. The generic
   "financial highlights" chunk of `05_financial_results.txt` outranks the specific line
   items for almost every financial question.
2. **A bigger k isn't the fix.** Held-out hit@5 equals hit@3. It would fix operating margin
   (right chunk at rank 4 or 5) but not the others. The ranking itself needs work.
3. **The guardrail reduces over-synthesis but doesn't stop it.** Asked directly for "total
   debt, including the financing for the WBD transaction", the model began adding $14.5B + $3B +
   $42.2B and was only cut off by the token limit (D5, D34).
4. **One real generation failure:** for "How many shares did Netflix repurchase in Q4 2025?"
   the right chunk was retrieved, but the model decided Q4 2025 "corresponds to September
   2023" and answered with December's figure (3,994,670 instead of 18,882,245).
5. **Cybersecurity is a ranking miss.** The Audit Committee sentence is in chunk 5 of the
   cybersecurity file, the only chunk of that file that wasn't in the top 5.

**Things to be careful about when quoting these numbers:**
- The baseline's two "correct" held-out answers are both refusals on questions where declining
  counts as correct. The baseline didn't actually answer anything correctly there.
- 17 questions on one 5-section document is a small sample. These are directional results,
  not benchmarks.

## A grader bug I caught

The first run marked the total-debt trap **correct**. The answer said "Adding these
components together gives us: $14.5 billion + $3 billion + $42.2" and was cut off before the
total, so no forbidden number appeared. I only noticed because I read the full answers behind
every held-out grade. Summing phrases are now forbidden on the debt questions, and a test uses
that exact answer (D34). The re-run changed only that one grade, and all 17 answers were
identical across the two runs.

## How to run it

```bash
pytest tests/test_evaluation.py     # grader + question data checks, no models
python -m eval.run_eval             # about 2 minutes on an Apple GPU; rewrites eval/results.*
```

## Interview talking points

- **"How do you know RAG helps?"** Same model, same questions, with and without retrieval: 1
  of 8 correct vs 6 of 8 on the benchmark. And on 9 questions I never tuned on, 4 of 9, which
  is lower and I report it.
- **"Why a held-out set?"** I chose the guardrail wording by looking at the benchmark answers,
  so the benchmark flatters it. The held-out set showed the guardrail doesn't stop the model
  from summing when the question asks for a total.
- **"What would you improve next?"** Retrieval ranking, because that's where most misses are.
  Add keyword search (BM25) next to embeddings for exact terms like "interest expense", or split
  the dense financial section into smaller chunks, then re-run this eval to see if hit@3 goes up.
- **"How do you trust your grader?"** It's rules in a JSON file, with unit tests, and I read
  the answers behind the grades. That's how I found it passing a half-finished sum.

## Known limitations and what's next

- Small, single-document question set; one model; one machine.
- The grader checks listed facts only. It won't catch every extra invented detail.
- Retrieval improvements are identified but not built (they're outside the plan).

## Handover notes

- `eval/results.md` is generated. Change `questions.json` or the code, then re-run; don't
  edit the results by hand.
- Adding a question: give it a verbatim `evidence` sentence from the expected file.
  `pytest tests/test_evaluation.py` will tell you if it isn't verbatim or spans two chunks.
- The eval uses a temporary index, so it's safe to run while the API server is using yours.
