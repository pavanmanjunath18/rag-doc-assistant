# Stage 3: idempotent ingestion

## Goal

Make uploads safe to repeat. The same file twice should do nothing, and a changed file with
the same name should replace its old chunks completely (no stale chunks).

## Starting point

Chunk IDs were `<filename>:<index>`. Two problems:
- **Stale chunks.** Upload a 10-chunk `report.pdf`, then a 4-chunk new version. Chunks 0 to 3
  are overwritten, but 4 to 9 from the old version stay and can still be retrieved and
  quoted as if current.
- **Wasted work.** Uploading the identical file again re-parsed and re-embedded everything.

## What changed

| File | Change | Why |
|---|---|---|
| `rag/loader.py` | `document_id(path)`: SHA-256 of the file bytes | D22 |
| `rag/interfaces.py` | `VectorStore` gains `find(doc_id)` and `delete_stale_versions(source, current_doc_id)`; `add` takes `doc_id`; new `StoredDocument` | D22, D24 |
| `rag/store.py` | Chunk IDs `<doc_id>:<index>`, `doc_id` in metadata; the two new methods | D22, D24 |
| `rag/pipeline.py` | `ingest` returns `IngestResult` with status `indexed` / `replaced` / `unchanged` | D23, D24 |
| `api/app.py`, `api/schemas.py` | Response adds `document_id` and `status`; 200 for unchanged, 201 otherwise; an unchanged upload doesn't overwrite the saved file | D25 |
| `cli.py` | Prints the status per file | |
| `tests/test_ingestion.py` (new) | Every idempotency scenario, run against both the fake store and real Chroma | |
| `tests/test_api.py`, `tests/test_store.py` | API re-upload and replace tests; `find` / `delete_stale_versions` tests | 100 tests total |

## How ingest decides what to do

```
doc_id = sha256(file bytes)
if store.find(doc_id):            -> "unchanged"  (nothing parsed, nothing embedded)
else:
    chunk, embed, store.add(doc_id, name, ...)
    removed = store.delete_stale_versions(name, keep=doc_id)
    -> "replaced" if removed else "indexed"
```

| You upload | Result |
|---|---|
| `report.txt` for the first time | `indexed` (201) |
| The identical `report.txt` again | `unchanged` (200), nothing re-embedded |
| The identical bytes as `copy.txt` | `unchanged` (200), still cited as `report.txt` |
| An edited `report.txt` | `replaced` (201), only the new chunks remain |
| The original `report.txt` again after editing | `replaced` (201), back to the original chunks |

## Verified on the real index

Your `chroma_db/` still had the 35 Stage 1b chunks with old-style IDs. Running
`python cli.py ingest data/corpus/`:

- First run: all 5 files `replaced` (old IDs had no `doc_id`, so they counted as stale).
- Second run: all 5 `unchanged`.
- Final count: 35 chunks, 0 old-style IDs left.

## How to run it

```bash
pytest tests/test_ingestion.py -v      # each scenario twice: fake store and real Chroma
python cli.py ingest data/corpus/      # run twice: second time says "unchanged"
```

## Numbers to remember

- Document ID: 64 hex characters (SHA-256).
- 100 tests; the idempotency tests run against both stores.

## Interview talking points

- **"How do you make ingestion idempotent?"** The document ID is a hash of the content, not
  the file name. If that hash is already stored, I skip the file before parsing or
  embedding. If the name is known but the hash is new, it's a new version: I store it,
  then delete every chunk under that name with a different hash.
- **"Why add before delete?"** So a failure never leaves the document missing. The cost is a
  brief moment where both versions exist, and if the process crashed in between, the next
  upload of that name cleans it up.
- **"What's a stale chunk and why does it matter?"** Keying chunks by file name and index means
  a shorter new version leaves the old version's tail chunks behind. They still get retrieved,
  so the model can quote numbers that are no longer in the document. For a financial Q&A tool
  that's a correctness bug, not a tidiness issue.
- **"How did you migrate existing data?"** I didn't need a separate migration. Old chunks have
  no `doc_id`, so the same stale-version delete removes them on the next ingest. I checked it
  on the real index before and after.

## Known limitations and what's next

- Two simultaneous uploads with the same name could interleave; Stage 4's one-at-a-time job
  queue removes this.
- Byte-level hashing treats a re-exported but textually identical PDF as a new version (it gets
  re-indexed as `replaced`, which is correct, just redundant).
- There is no endpoint to delete a document yet. It wasn't in the plan; the store method that
  would back it (`delete_stale_versions`) is already there.

## Handover notes

- The Chroma metadata per chunk is `{"doc_id", "source", "chunk"}`. Anything that writes to the
  collection directly must include `doc_id`, or its chunks will be treated as stale.
- `InMemoryStore` in `tests/fakes.py` mirrors the Chroma behaviour. If you change one, change
  the other; `tests/test_ingestion.py` runs the same scenarios against both to catch drift.
