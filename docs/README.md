# Docs

Start here if you are new to the project (or explaining it to someone). For a quick
overview read the [main README](../README.md) first, then the stage notes in order.

| Doc | What it is |
|---|---|
| [proof-notebook.md](proof-notebook.md) | What the original Colab notebook did, its benchmark results, and the failures it showed. This is the baseline everything is compared against. |
| [decisions.md](decisions.md) | Every design decision, with the alternatives and tradeoffs. Numbered (D1, D2, ...) so other docs can point to them. |
| [stages/](stages/) | One set of notes per build stage: what changed, why, how to run it, and how to explain it. Read them in order. |

## Stage notes

1. [Stage 1b: align with the proof notebook](stages/stage-1b-align-with-notebook.md)
2. [Stage 2: FastAPI service](stages/stage-2-fastapi-service.md)
3. [Stage 3: idempotent ingestion](stages/stage-3-idempotent-ingestion.md)
4. [Stage 4: background ingestion jobs](stages/stage-4-background-jobs.md)
5. [Stage 5: evaluation harness](stages/stage-5-evaluation.md)
6. [Stage 6: React frontend](stages/stage-6-frontend.md)
7. [Stage 7: Docker and CI](stages/stage-7-packaging.md)
8. [Stage 8: README and docs](stages/stage-8-docs.md)
