# RAG Document Assistant

Ask questions about your own documents (PDFs, text) and get answers grounded in them, with sources.
Runs fully locally: MiniLM embeddings, Chroma vector store, and a small open model (Qwen2.5-1.5B) for generation.

> Work in progress. API, background ingestion, evaluation, and a web UI are coming in later stages.

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

## Tests and linting

```bash
pytest
ruff format --check . && ruff check .
```

## Docs

See [docs/](docs/README.md): the proof notebook's results, the design decisions log, and
notes for each build stage.
