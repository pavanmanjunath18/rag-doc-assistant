# RAG Document Assistant

Ask questions about your own documents (PDFs, text) and get answers grounded in them, with sources.
Runs fully locally: MiniLM embeddings, Chroma vector store, and a small open model (Qwen2.5-1.5B) for generation.

> Work in progress. Background ingestion, evaluation, and a web UI are coming in later stages.

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
| `POST /documents` | Upload a `.pdf`, `.txt` or `.md` file (multipart field `file`) and index it. Re-uploading identical content is a no-op; a changed file with the same name replaces its old chunks |
| `POST /query` | `{"question": "...", "top_k": 3, "use_rag": true}` returns an answer with its sources |

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Tests and linting

```bash
pytest
ruff format --check . && ruff check .
```

## Docs

See [docs/](docs/README.md): the proof notebook's results, the design decisions log, and
notes for each build stage.
