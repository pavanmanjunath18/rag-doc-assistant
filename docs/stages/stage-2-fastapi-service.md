# Stage 2: FastAPI service

## Goal

Put the pipeline behind an HTTP API with three endpoints: upload a document, ask a question,
check health. Validate every input, return clear errors, and load the models only once.

## Starting point

After Stage 1b the pipeline worked from the CLI, but it was built from global functions that
loaded models on first use (`lru_cache`). You couldn't test anything above the prompt
without loading a 3 GB model, and there was no way to swap parts.

## What changed

| File | Change | Why |
|---|---|---|
| `rag/interfaces.py` (new) | `Embedder`, `VectorStore`, `Generator` protocols, and `Hit` | D13 |
| `rag/pipeline.py` | `RagPipeline` class that receives its three parts; `ingest(path, source=)` | D13 |
| `rag/factory.py` (new) | `build_pipeline(settings)`: the only place real models are created | D13 |
| `rag/embeddings.py`, `rag/generator.py`, `rag/store.py` | Became classes (`SentenceTransformerEmbedder`, `HuggingFaceGenerator`, `ChromaStore`); store takes vectors; generator has a lock; Chroma telemetry off | D14, D17 |
| `rag/config.py` | `Settings` dataclass with `from_env()`; new `UPLOAD_DIR`, `MAX_UPLOAD_MB` | D20 |
| `rag/loader.py` | `DocumentError` for bad documents (incl. corrupt PDFs); chunk size/overlap passed in | Clean 422s |
| `rag/logs.py` (new) | Logging setup shared by CLI and API | |
| `api/app.py` (new) | `create_app()`, lifespan model loading, the three endpoints | D15, D16, D17 |
| `api/schemas.py` (new) | Pydantic request/response models | D19 |
| `api/uploads.py` (new) | Filename sanitizing, size-limited save, content check | D18 |
| `cli.py` | Builds the pipeline via the factory (after parsing args, so `--help` is instant) | |
| `tests/` | Fakes + fixtures; new API, upload, store and config tests | 84 tests, 0.5 s |

## The endpoints

| Method and path | Does | Success | Errors |
|---|---|---|---|
| `GET /health` | Is it up, how many chunks are indexed | 200 `{"status": "ok", "chunks_indexed": 10}` | |
| `POST /documents` | Upload one file (multipart field `file`) and index it | 201 `{"filename": "02_risk_factors.txt", "chunks": 10}` | 400 empty, 413 too big, 415 wrong type, 422 no text / unreadable |
| `POST /query` | JSON `{"question": ..., "top_k": 1-10 (optional), "use_rag": true}` | 200 `{"answer": ..., "sources": [{"source", "chunk", "score", "text"}]}` | 422 invalid body |

Interactive docs are generated automatically at `http://127.0.0.1:8000/docs`.

## How a request flows

**Upload** (`POST /documents`):
1. `sanitize_filename` turns `../../Q3 report.pdf` into `Q3_report.pdf`.
2. The extension is checked against the allowlist.
3. `save_upload` copies the body into a temp file in `uploads/`, 1 MB at a time, and stops
   if it passes the limit.
4. `check_content` looks at the first bytes (a PDF must start with `%PDF-`).
5. `pipeline.ingest(temp_file, source="Q3_report.pdf")` loads, chunks, embeds and stores it.
6. The temp file is renamed to `uploads/Q3_report.pdf`. On any error it's deleted instead.

**Query** (`POST /query`): Pydantic validates the body (non-blank question, top_k 1 to 10,
no unknown fields), then `pipeline.ask()` runs the same steps as the CLI: embed the question,
search the store, build the prompt, generate. The response includes each source's text, so
the UI can show citations.

## How to run it

```bash
pip install -r requirements-dev.txt
pytest                                           # 84 tests, no models needed
uvicorn --factory api.app:create_app --port 8000 # loads both models, then serves
```

In another terminal:

```bash
curl -F "file=@data/corpus/02_risk_factors.txt" http://127.0.0.1:8000/documents
curl -H 'Content-Type: application/json' -d '{"question": "How much debt does Netflix have?"}' http://127.0.0.1:8000/query
```

## Verified with the real models

Run against a fresh index with uvicorn and curl (not just the test client):

- Startup loaded MiniLM and Qwen (`mps:0`, float16) before accepting requests.
- Uploading `02_risk_factors.txt` returned 201 with 10 chunks.
- "How much debt does Netflix have?" answered "$14.5 billion of senior notes" with
  `02_risk_factors.txt` chunk 7 ranked first (score 0.64).
- A fake `.pdf`, a `.exe` and a blank question were rejected with 415, 415 and 422.
- The upload folder held only the saved file afterwards, with no temp files.
- The CLI still works after the refactor (WBD fee question: "$5.8 billion").

## Numbers to remember

- 3 endpoints; 84 tests in about 0.5 s with no model loaded.
- Upload limit 20 MB by default (`MAX_UPLOAD_MB`); filenames capped at 100 characters.
- `top_k` accepted range 1 to 10; questions up to 1000 characters.

## Interview talking points

- **"How did you make it testable?"** The pipeline receives its embedder, store and generator
  instead of creating them. Tests pass fakes: a word-hashing embedder, a dict-based store and
  a generator that records its prompts. So the API tests run real FastAPI and real pipeline
  code in under a second.
- **"Why `def` and not `async def`?"** The model calls block. In an `async def` they'd freeze
  the event loop and every other request. With `def`, FastAPI runs them in a thread pool.
  There's one model in memory, so a lock makes generation one at a time.
- **"How do you validate uploads?"** In layers: sanitize the name, allowlist the extension,
  cap the size while streaming, check the actual bytes, then write to a temp file and only
  move it into place once indexing worked. And I can name the gap: the framework reads the
  whole body before my code runs, so the real size limit belongs in a proxy or gateway.
- **"How would this move to AWS?"** The interfaces are the seam: a Bedrock `Generator` and a
  managed vector store implement the same three methods, and a Lambda handler becomes another
  thin layer like `api/`.

## Known limitations and what's next

- Re-uploading a file with the same name overwrites the file, but chunk IDs are
  `name:index`, so a shorter new version leaves old chunks behind (Stage 3).
- Upload indexes before responding, so a big PDF holds the request open (Stage 4).
- No authentication or rate limiting. Out of scope for a local tool; note it if asked.
- No CORS yet; it'll be added with the frontend in Stage 6.

## Handover notes

- `create_app(settings, pipeline)` is the entry point. Pass a pipeline to skip model loading
  (that's what `tests/conftest.py` does).
- Uvicorn needs `--factory` because `create_app` is a function, not a module-level `app`.
  That keeps importing the module free of side effects.
- Uploaded files are kept in `UPLOAD_DIR` (default `uploads/`, git-ignored). Deleting that
  folder does not remove indexed chunks; delete `chroma_db/` too for a full reset.
- Temp files are named `.upload-*` in the upload folder. Leftovers only happen if the process
  is killed mid-upload and are safe to delete.
