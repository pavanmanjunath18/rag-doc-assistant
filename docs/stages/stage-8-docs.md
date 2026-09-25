# Stage 8: README and docs

## Goal

A README someone can read in five minutes and trust: what it is, how it's built, how to run
it, what the API does, how well it works (with numbers), why it's built this way, and what
went wrong along the way.

## What the README covers

| Section | Where the content comes from |
|---|---|
| Summary paragraph | The project as built; the two headline numbers come from `eval/results.md` |
| Architecture (Mermaid) | `api/app.py`, `rag/pipeline.py`, `rag/jobs.py`, `rag/interfaces.py` |
| Quick start | Commands that were run while building each stage |
| API reference | `api/schemas.py` and the status codes tested in `tests/test_api.py` |
| Evaluation table | `eval/results.md` from commit `3f7565a` (identical across three runs) |
| Design decisions | A short version of `docs/decisions.md` with D-numbers to follow up |
| Failures I hit | The proof notebook (cells 9 to 13) and the debugging done in Stages 1b to 7 |
| Limitations | Stage 5 findings and the known gaps listed in each stage note |

## Rules I followed

- Every number in the README can be traced to a file in the repo or a run described in a
  stage note. No "fast", "accurate" or "production-ready" without a measurement behind it.
- Results that aren't flattering (held-out 4 of 9, the guardrail still summing, 39 s per answer
  in Docker) are in the README, not only in the detailed notes.
- The AWS version is described as planned, not built.

## How to keep the docs honest when things change

- Changed the prompt, chunking or models? Re-run `python -m eval.run_eval`, commit the new
  `eval/results.md`, then update the README table from it.
- Added a setting? `tests/test_config.py` fails until it's in `.env.example`.
- Made a design choice with a tradeoff? Add the next D-number to `docs/decisions.md`, and a
  line in the stage note that introduced it.

## Interview talking points

- **"Walk me through the project."** Use the README top to bottom: the summary, the diagram,
  one request through the pipeline, the evaluation table, then one failure story.
- **"What are you least happy with?"** Retrieval ranking. Held-out hit@3 is 5 of 8, and
  that's where most wrong or declined answers come from. The fix I'd try first is hybrid
  keyword + embedding search, measured with the same evaluation.
