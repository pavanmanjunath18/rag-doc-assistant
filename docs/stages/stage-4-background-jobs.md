# Stage 4: background ingestion jobs

## Goal

Stop large uploads from blocking the request. Upload should return straight away with a job
ID, indexing should happen in the background, and `GET /jobs/{id}` should report `queued`,
`running`, `succeeded` or `failed` (with the error).

## Starting point

`POST /documents` parsed, embedded and stored the file before responding. The 10-K sections
take under a second, but a 200-page PDF would hold the HTTP connection open for as long as
embedding takes, and a client or proxy timeout would lose the result.

## What changed

| File | Change | Why |
|---|---|---|
| `rag/jobs.py` (new) | `JobStore` (SQLite table), `IngestWorker` (one background thread + queue), `JobStatus` | D26 |
| `api/app.py` | Upload returns 202 + `Location`; new `GET /jobs/{id}`; lifespan starts/stops the worker and cleans up after a restart | D27, D28, D29 |
| `api/schemas.py` | `JobAccepted`, `JobResponse`, `IngestOutcome` | D29 |
| `api/uploads.py` | `remove_stale_temp_files` | D28 |
| `rag/loader.py`, `rag/pipeline.py` | Error messages use the uploaded file's name, not the temp file's | D30 |
| `rag/config.py`, `.env.example`, `.gitignore` | `JOBS_DB` setting (default `jobs.sqlite3`, git-ignored) | |
| `tests/test_jobs.py` (new) | Job table transitions; worker states, ordering, errors, cleanup | |
| `tests/test_api.py` | Upload then poll; failed jobs; restart recovery; 404 | 118 tests total |

## How an upload flows now

```
POST /documents
  sanitize name, check extension, save with size limit, sniff content   -> 4xx right away if bad
  jobs.create(name)                                   status = queued
  worker.submit(task)                                 (returns immediately)
  <- 202 {"job_id": ..., "status": "queued"}   Location: /jobs/<id>

worker thread (one job at a time):
  status = running
  pipeline.ingest(temp_file, source=name)             hash, skip or chunk + embed + store
  keep the file in uploads/ (unless unchanged)
  status = succeeded + result     or     failed + error
  delete the temp file

GET /jobs/<id>  ->  {"status": ..., "error": ..., "result": {...}}
```

Job states:

| Status | Meaning | Set by |
|---|---|---|
| `queued` | Accepted, waiting for the worker | `POST /documents` |
| `running` | The worker is ingesting it | worker |
| `succeeded` | Done; `result` has `document_id`, `filename`, `chunks`, `status` (`indexed` / `replaced` / `unchanged`) | worker |
| `failed` | `error` says why: unreadable document, unexpected error, or a server restart | worker, or startup |

## Verified with the real models

Against a running server with an empty index:

- `02_risk_factors.txt`: 202, then `running` to `succeeded` in 0.49 s, 10 chunks, `indexed`.
- `05_financial_results.txt`: 202, `succeeded`, 7 chunks.
- `02_risk_factors.txt` again: `succeeded` with status `unchanged`, same `document_id`.
- A broken PDF: 202 (it looks like a PDF), then `failed`: "Could not read PDF broken.pdf:
  Stream has ended unexpectedly".
- `GET /jobs/nope`: 404.
- Querying afterwards worked as before; `/health` showed 17 chunks.

## How to run it

```bash
uvicorn --factory api.app:create_app --port 8000
curl -i -F "file=@data/corpus/02_risk_factors.txt" http://127.0.0.1:8000/documents
curl http://127.0.0.1:8000/jobs/<job_id from the response>
```

## Numbers to remember

- One worker thread, jobs run in submission order.
- Job IDs are random UUIDs (32 hex characters); document IDs are content hashes (64).
- 118 tests; the worker tests run real threads and passed three runs in a row.

## Interview talking points

- **"Why not just use FastAPI BackgroundTasks?"** It keeps no status you can poll, forgets
  everything on restart, and runs tasks in parallel, which would bring back the race where two
  uploads of the same name interleave. A SQLite table and one worker thread fix all three with
  no extra dependencies.
- **"What happens if the server dies mid-job?"** On startup, anything still `queued` or
  `running` is marked `failed` with a message to re-upload, and leftover temp files are deleted.
  Re-uploading is safe because ingestion is idempotent.
- **"Which errors are synchronous and which are async?"** Anything cheap to check (name, type,
  size, first bytes) is rejected in the request. Anything that needs parsing is reported on the
  job. So a client never waits for a job just to learn it uploaded a `.exe`.
- **"How would this scale?"** Move the queue out of the process (Redis/SQS) and run several
  workers. On AWS, S3 upload triggers plus Step Functions would replace the worker thread,
  and the job table would move to DynamoDB. The pipeline itself doesn't change.
- **"How do you test background threads?"** A stand-in pipeline that blocks on an `Event`, so
  the test can observe `running` and `queued` at a precise moment, and `wait_until_idle()`
  (a `queue.join()`) instead of sleeps.

## Known limitations and what's next

- One process, one worker, unbounded in-memory queue. Fine for local use.
- Job rows are never cleaned up. A small table for a local tool; a real service would expire
  old rows.
- No endpoint to list jobs. The UI (Stage 6) tracks the IDs it created.

## Handover notes

- Job state lives in `jobs.sqlite3` (set by `JOBS_DB`). Deleting it only loses job history,
  not indexed documents.
- `IngestWorker.stop()` finishes already-queued jobs before exiting, so shutdown can take as
  long as the queue.
- The CLI still ingests synchronously; jobs only exist in the API.
