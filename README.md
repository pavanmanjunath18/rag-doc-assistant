# RAG Document Assistant

Ask questions about your own documents (PDFs, text) and get answers grounded in them, with sources.
Runs fully locally: MiniLM embeddings, Chroma vector store, and a small open model (Qwen2.5-1.5B) for generation.

> Work in progress: packaging (Docker, CI) comes next.

## Quick start

Requires Python 3.11+.

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env

python cli.py ingest data/corpus/
python cli.py ask "How much debt does Netflix have?"
python cli.py ask "How much debt does Netflix have?" --no-rag   # baseline
```

## API

```bash
uvicorn --factory api.app:create_app --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Status and number of indexed chunks |
| `POST /documents` | Upload a `.pdf`, `.txt` or `.md` file (multipart field `file`). Returns `202` with a `job_id`; indexing runs in the background. Re-uploading identical content is a no-op; a changed file with the same name replaces its old chunks |
| `GET /jobs/{job_id}` | Job status (`queued`, `running`, `succeeded`, `failed`), the error if it failed, the result if it succeeded |
| `POST /query` | `{"question": "...", "top_k": 3, "use_rag": true}` returns an answer with its sources |

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Web UI

```bash
cd frontend && npm install && npm run dev    # http://localhost:5173 (API must be on :8000)
```

Upload a document, watch its ingestion job, ask a question, and expand each cited source to
see the exact chunk the answer came from.

## Evaluation

```bash
python -m eval.run_eval    # writes eval/results.md
```

| Question set | No retrieval | RAG | Retrieval hit@3 |
|---|---|---|---|
| Benchmark (8 questions from the proof notebook) | 1/8 correct | 6/8 correct | 6/8 |
| Held-out (9 new questions, not used for tuning) | 2/9 correct* | 4/9 correct | 5/8 |

\* both are refusals on questions where declining counts as correct. Full results and
caveats: [eval/results.md](eval/results.md), [stage 5 notes](docs/stages/stage-5-evaluation.md).

## Tests and linting

```bash
pytest
ruff format --check . && ruff check .
```

## Docs

See [docs/](docs/README.md): the proof notebook's results, the design decisions log, and
notes for each build stage.
