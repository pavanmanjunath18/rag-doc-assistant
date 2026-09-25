# Design decisions

Each entry: what was decided, why, what else was considered, and what it costs.
"Notebook" means the proof notebook, summarised in [proof-notebook.md](proof-notebook.md).

## D1. Chunk size 800 characters, overlap 150 (Stage 1b)

**Decision:** Keep the notebook's values.
**Why:** They produced correct answers on 5 of 8 benchmark questions, and keeping them makes
this repo's results comparable with the notebook.
**Alternatives:** Token-based chunking; sentence or paragraph splitting.
**Tradeoff:** Characters are not tokens, so chunk length in tokens varies a little. Fine for a
model with a large context window and only 3 chunks per prompt.

## D2. Chunk edges snap to whitespace; line breaks are kept (Stage 1b)

**Decision:** Unlike the notebook's raw slicing, a chunk never ends or starts mid-word, and
text is not whitespace-flattened.
**Why:** Raw slicing can cut a figure in half (for example `$45,183,0` | `36`), which is a real
risk for financial questions. Keeping line breaks keeps headings like `INDEBTEDNESS` visible
to the model.
**Alternatives:** Match the notebook exactly; paragraph-first splitting. Paragraph-first was
considered to stop sections mixing, but the notebook's debt error came from two figures in
the same paragraph, so it would not have helped.
**Tradeoff:** The corpus gives 35 chunks instead of the notebook's 37, and the overlap is
sometimes a few characters under 150.

## D3. Retrieve the top 3 chunks (Stage 1b)

**Decision:** Keep the notebook's `TOP_K=3` (the Stage 1 scaffold had 5).
**Why:** Comparable results; less text for a 1.5B model to get confused by.
**Tradeoff:** The notebook's operating-margin answer missed 29.5%, which may be a retrieval
miss that a larger k would fix. Stage 5 measures hit@k instead of guessing.
**Stage 5 result:** For operating margin, the right chunk ranks 4th or 5th (hit@3 no, hit@5
yes), so k=5 would fix that one question. But on the held-out set hit@5 equals hit@3 (5 of
8), so a bigger k isn't the main fix; ranking is.

## D4. RAG prompt: the notebook's template, made document-agnostic (Stage 1b)

**Decision:** Use the notebook's final prompt: a soft refusal ("only if you find nothing
relevant at all") and a synonym hint ("debt" may appear as "notes", "indebtedness",
"liabilities"), sent as a single user message. "Netflix's annual report" became "the provided
documents".
**Why:** The Stage 1 scaffold used a strict "ONLY the context, otherwise say you don't know"
prompt, which is the style that caused the notebook's first failure. The app accepts any
upload, so the prompt should not name one company.
**Tradeoff:** The synonym hint is finance-flavoured. It's kept as an example of the idea and
does no harm on other documents, but it is not tuned for them.

## D5. Guardrail against combining figures (Stage 1b)

**Decision:** Add one line to the prompt: "Quote each figure exactly as it appears in the
context, and do not add figures together to make a new total."
**Why:** The notebook's model summed senior notes and an undrawn credit facility into a
"total debt" that the document never states.
**Tradeoff:** This discourages legitimate arithmetic questions ("what is revenue minus
costs?"). For a document Q&A tool, a wrong confident number is worse than no calculation.

**How the wording was chosen (first manual run, Stage 1b).** The first version said "Report
figures exactly as they are stated. Do not add, combine, or calculate figures, and do not give
a total unless the context states one." With it, the model answered "How much debt does
Netflix have?" with "The context does not contain this information", even though the top
chunk contained the $14.5B figure. Testing one change at a time (same retrieved chunks,
Qwen2.5-1.5B, float16 on an Apple GPU):

- Removing the line fixed the refusal (6 of 6 runs with it refused, 0 of 6 without, across
  greedy and 5 sampled seeds). Changing "provided documents" back to "Netflix's annual report"
  made no difference. So the guardrail itself caused it, most likely because "don't give a
  total unless stated" collides with "how much", which reads as asking for a total.
- Removing the line entirely made the model refuse the WBD termination-fee question instead.
- Four wordings were compared on the 8 benchmark questions plus "What is Netflix's total
  debt?" (one greedy run each). The chosen wording was the only one with no refusals and no
  wrong answers: 6 correct, 2 partial (cybersecurity oversight, operating margin). A shorter
  wording ("Quote figures exactly... Do not add figures together or calculate new ones.")
  refused the debt question and garbled the stock-split answer.
- All wordings answered "What is Netflix's total debt?" with the refusal message rather than
  a sum. That is acceptable, since no total is stated.

**Limits of this evidence:** one model, mostly single greedy runs, and the wording was picked
after seeing results on the same questions, so it is tuned to this benchmark. The notebook's
$17.5B sum did not reproduce in 6 runs without the guardrail either, so this run can't show
that the guardrail prevents it. Stage 5 should measure both on questions not used here.

**Stage 5 result:** On the benchmark debt question the answer is now the $14.5B of senior
notes with no sum. But on a held-out question that asks directly for "total debt, including
the financing for the WBD transaction", the model started adding $14.5B + $3B + $42.2B and was
only stopped by the token limit. So the guardrail reduces over-synthesis but does not prevent
it when the question itself asks for a total. See [eval/results.md](../eval/results.md).

## D6. Greedy decoding (Stage 1b)

**Decision:** `do_sample=False`. The notebook used Qwen's default sampling.
**Why:** The same question gives the same answer, which evaluation needs. With sampling, a
changed answer could be the prompt, the retrieval, or just randomness.
**Tradeoff:** Stage 5 results may not match the notebook answer for answer. Differences will
be reported, not hidden.

## D7. 200 new tokens (Stage 1b)

**Decision:** Keep the notebook's `MAX_NEW_TOKENS=200` (Stage 1 had 256). Configurable.
**Tradeoff:** Long answers get cut off (one notebook baseline answer was). RAG answers were
short, so this didn't affect them.

## D8. Normalised embeddings with cosine distance (Stage 1)

**Decision:** Keep Stage 1's `normalize_embeddings=True` and a cosine Chroma collection. The
notebook used defaults (L2 distance).
**Why:** `all-MiniLM-L6-v2` already outputs unit-length vectors, and on unit vectors L2 and
cosine give the same ranking, so retrieval matches the notebook. Cosine gives a readable
score (1 = identical meaning), which is shown next to each source.

## D9. Baseline is the bare question (Stage 1b)

**Decision:** No system prompt for the baseline, matching the notebook (Stage 1 added
"Answer the question concisely.").
**Note:** With no system message, Qwen's chat template inserts its own default one ("You are
Qwen, created by Alibaba Cloud..."). That is why a baseline answer mentions Alibaba Cloud.

## D10. Sources in rank order, one per chunk, with scores (Stage 1b)

**Decision:** Return every retrieved chunk in the order it was ranked, with file, chunk number
and score. Stage 1 returned a sorted, de-duplicated set of names.
**Why:** Sorting hid which chunk was most relevant, and de-duplicating hid that two chunks came
from the same file. The UI (Stage 6) needs this to show citations.

## D11. Prompts live in their own module (Stage 1b)

**Decision:** `rag/prompts.py` builds messages and imports nothing heavy.
**Why:** Prompt wording is the thing most likely to change and most worth testing, and the tests
shouldn't need a 3 GB model.

## D12. Pick the device explicitly; float16 on any GPU (Stage 1b)

**Decision:** `pick_device()` chooses an NVIDIA GPU (`cuda`), then an Apple GPU (`mps`), then
CPU. GPUs load the model in float16, CPUs in float32. The model is moved there with `.to()`
instead of `device_map="auto"`, and the `accelerate` dependency was dropped.
**Why:** The scaffold used float16 only on CUDA, so on a Mac the 1.5B model loaded in float32
(about 6.2 GB). On a 16 GB Mac already deep in swap, `device_map="auto"` quietly offloaded
some weights to disk ("Some parameters are on the meta device because they were offloaded to
the disk"), and in another shell the same load crashed with a segmentation fault (exit 139).
In float16 on `mps` the model takes 3.09 GB, loads in about 5 s, and a full CLI question runs
in about 15 s including loading both models. This is the same class of problem as the proof
project's CPU-vs-GPU failure: where the model runs, and in what precision, is part of
correctness, not just speed.
**Tradeoff:** No automatic splitting of a model that doesn't fit on one device. That doesn't
matter for a 1.5B model, and a clear out-of-memory error is easier to debug than silent disk
offload.

## D13. Small interfaces, passed in (Stage 2)

**Decision:** `rag/interfaces.py` defines three `Protocol`s: `Embedder`, `VectorStore` and
`Generator`. `RagPipeline` receives one of each in its constructor. `rag/factory.py` is the one
place that builds the real ones (MiniLM, Chroma, Qwen).
**Why:** Stage 1 used global `lru_cache` loaders, so any code that touched the pipeline loaded
a 3 GB model. Now tests pass fakes (84 tests in under a second, no model), and a later AWS
version can pass a Bedrock generator or a different vector store without editing the pipeline.
**Alternatives:** Abstract base classes (need inheritance; a `Protocol` just needs the right
methods); a dependency-injection framework (overkill for three objects).
**Tradeoff:** A little more code and one more file than calling the models directly.

## D14. The store takes vectors; it doesn't embed (Stage 2)

**Decision:** `VectorStore.add` and `search` take embeddings. The pipeline does the
embedding. In Stage 1 the store called the embedder itself.
**Why:** Each part does one job, so either can be replaced alone (for example, keep MiniLM
but move to a hosted vector database).

## D15. The API is a thin layer in its own package (Stage 2)

**Decision:** `api/` holds only HTTP concerns: routes, request/response models, upload
checks. It calls the same `RagPipeline` the CLI uses. `rag/` never imports FastAPI.
**Why:** The core logic is tested once and reused by both front ends. A future Lambda handler
would be a third thin layer on the same core.

## D16. Load models once at startup and fail fast (Stage 2)

**Decision:** The FastAPI `lifespan` hook builds the pipeline before the server accepts
requests. If a model fails to load, the server doesn't start.
**Why:** Loading per request would add seconds to every call. Failing at startup means that
`/health` returning `ok` actually means ready. Catching the error and starting anyway would
only move the failure to the first request.
**Tradeoff:** Startup takes a few seconds (both models load before the port opens).

## D17. Plain `def` endpoints, one generation at a time (Stage 2)

**Decision:** Endpoints are `def`, not `async def`, and `HuggingFaceGenerator` holds a lock
around `generate()`.
**Why:** Embedding and generation block the CPU/GPU. FastAPI runs `def` endpoints in a thread
pool, so a slow answer doesn't freeze `/health` or other requests. `async def` with blocking
calls inside would stall the whole server. There is one model instance shared by all threads,
so the lock stops two requests from running it at the same time.
**Tradeoff:** Answers are produced one at a time. Fine for a local tool; at real scale you'd
use a model server that batches requests, or a hosted model.

## D18. Upload checks in layers (Stage 2)

**Decision:** Every upload goes through, in order:
1. Filename sanitized: directory parts dropped (blocks `../../` path traversal), only
   `A-Z a-z 0-9 . _ -` kept, length capped at 100 with the extension kept.
2. Extension allowlist: `.pdf`, `.txt`, `.md` (otherwise 415).
3. Size limit while copying: read in 1 MB blocks, stop once past `MAX_UPLOAD_MB` (413).
   Empty files are rejected (400).
4. Content check: a `.pdf` must start with `%PDF-`, and a text file must not contain NUL bytes
   (415). The name alone proves nothing.
5. Written to a temp file first, indexed, then renamed into place. Any failure deletes the
   temp file, so a rejected upload leaves nothing behind.
6. Unreadable or textless documents get a 422 with the loader's message.

**Known gap:** Starlette parses the multipart body (spooling it to a temp file) *before* our
code runs, so the size limit stops us from storing a huge file but not from receiving it. In
production this limit belongs in front of the app too (a reverse proxy's body-size limit, or an
API gateway's payload limit). The content check is also a sanity check, not a malware scan.

## D19. Status codes and error format (Stage 2)

**Decision:** 201 for a created document, 400 empty file, 413 too large, 415 wrong type,
422 invalid request or unreadable document. Errors use FastAPI's standard `{"detail": ...}`.
`QueryRequest` rejects unknown fields, so a typo like `topk` is an error instead of being
silently ignored.

## D20. Settings as one object (Stage 2)

**Decision:** `Settings` is a frozen dataclass built by `Settings.from_env()`. It replaced the
module-level constants of Stage 1.
**Why:** Tests create their own `Settings` (temp folders, 1 MB upload limit) without touching
environment variables. A bad value such as `TOP_K=three` fails at startup with the variable's
name in the message.

## D21. Asking with nothing indexed is not an error (Stage 2)

**Decision:** `/query` on an empty index returns 200 with "No documents indexed yet." and no
sources, and the model isn't called.
**Why:** The request itself is valid, and the UI can show the message as-is.
**Tradeoff:** A client has to read the text to tell this apart from a real answer. If that
becomes a problem, add a field rather than an error code.

## D22. Document ID = SHA-256 of the file's bytes (Stage 3)

**Decision:** `document_id(path)` hashes the raw bytes. Chunk IDs are `<doc_id>:<index>`, and
each chunk's metadata stores `doc_id`, `source` (the file name) and `chunk`.
**Why:** The same bytes always get the same ID, so "have I seen this?" is one lookup, done
before any parsing or embedding.
**Alternatives:** The file name (the Stage 1 approach: can't tell versions apart, which caused
stale chunks); a hash of the extracted text (would also match a re-saved PDF with identical
text, but needs parsing first and changes if the PDF library changes); a random UUID (no
dedup at all).
**Tradeoff:** Byte-level: re-exporting the same PDF can change a timestamp inside it and give a
new ID. That just re-indexes it as a replacement, which is wasted work, not wrong results.

## D23. Same content under a different name is a no-op (Stage 3)

**Decision:** If the bytes are already indexed under another name, nothing is stored, and the
response reports the existing name with status `unchanged`.
**Why:** Indexing it again would put identical chunks in the store twice, and they would crowd
out other results in the top 3.
**Tradeoff:** Citations keep showing the first name the content was uploaded under.

## D24. Store the new version first, then delete the old one (Stage 3)

**Decision:** For a changed file with a known name: embed and store the new chunks, then
delete every chunk under that name whose `doc_id` isn't the new one.
**Why:** If embedding or storing fails, the old version is still there and searchable. The
opposite order would leave the document missing after a failure.
**Tradeoff:** Chroma has no multi-step transactions. For a moment both versions are searchable,
and if the process died between the two steps both would stay until that name is uploaded
again (which deletes everything except the current version, so it heals itself).
**Bonus:** Chunks written by Stages 1 and 2 have no `doc_id`, so they count as stale and are
cleaned up automatically on the next ingest. Verified on the real index: 35 old-format chunks
became 35 new ones, 0 left behind.

## D25. 200 for "already indexed", 201 for "indexed" (Stage 3, replaced by D29 in Stage 4)

**Decision:** `POST /documents` returns `status` (`indexed`, `replaced` or `unchanged`) and
`document_id`. The HTTP code is 201 when something was stored, 200 when nothing changed.
**Why:** Clients (and the UI) can tell a duplicate upload from a real one without comparing
chunk counts.
**Known gap:** Two uploads of the same file name at the same moment could interleave their
store and delete steps. Stage 4 runs ingestion jobs one at a time, which removes that race.

## D26. Background jobs: a SQLite table and one worker thread (Stage 4)

**Decision:** `POST /documents` records a job in a SQLite table (`rag/jobs.py`) and hands
the work to a single background thread, which runs jobs one at a time and updates the row:
`queued` then `running` then `succeeded` or `failed` (with the error text).
**Why:** A long PDF no longer holds the HTTP request open. SQLite comes with Python, needs no
extra server, and keeps job history across restarts. One worker is enough: there's one
embedding model in memory, and running jobs in order removes the Stage 3 race where two
uploads of the same file name could interleave their add and delete steps.
**Alternatives:**
- FastAPI `BackgroundTasks`: simplest, but it keeps no status to poll, loses everything on
  restart, and runs tasks concurrently in the thread pool (so the race comes back).
- Celery or RQ with Redis: real retries and multiple workers, but an extra service to run and
  explain, for a one-person local tool.
- A hosted workflow service: the planned AWS version (Step Functions) fills this role there.
**Tradeoff:** Single process only. The in-memory queue isn't bounded, so a flood of uploads
just waits in line. Scaling out would mean moving the queue out of the process.

## D27. Cheap checks in the request, expensive work in the job (Stage 4)

**Decision:** Filename, extension, size and content-sniffing still run before responding
(400/413/415 immediately). Parsing, embedding and storing happen in the job, so a PDF that
turns out to be unreadable is reported as a failed job with the reason.
**Why:** The client gets instant feedback for mistakes that cost nothing to detect, and a
job is only created for files that are plausibly valid.

## D28. After a restart, unfinished jobs fail loudly (Stage 4)

**Decision:** At startup, any job still `queued` or `running` is marked `failed` with "The
server restarted before this job finished. Upload the file again.", and leftover
`.upload-*` temp files are deleted.
**Why:** Otherwise those jobs would say `running` forever. Automatic retry was considered,
but the temp file may be half-written and the user may no longer care, so asking for a
re-upload is simpler and honest. Because ingestion is idempotent (D22), re-uploading is safe.

## D29. `202 Accepted` plus a job to poll (Stage 4)

**Decision:** Upload returns 202 with `job_id` and a `Location: /jobs/{id}` header.
`GET /jobs/{id}` returns the status, timestamps, the error if it failed, and the ingest result
(`document_id`, name cited under, chunk count, `indexed` / `replaced` / `unchanged`) if it
succeeded. Unknown IDs return 404. This replaces the 200/201 distinction from D25.

## D30. Errors name the user's file, not the temp file (Stage 4)

**Decision:** `load_text(path, name)` takes the display name to use in messages, and the
pipeline passes the upload's real name.
**Why:** A Stage 4 test caught that failed uploads reported names like `.upload-c2plpu6s.pdf`.
The bug was there since Stage 2, but those tests only checked the status code, not the
message. The API tests now check that the error contains the user's file name and not the
temp name.

## D31. A rule-based grader, not an LLM judge (Stage 5)

**Decision:** Each question in `eval/questions.json` lists key facts (every group must match),
forbidden strings (any match is wrong), and whether a refusal is required or acceptable.
`rag/evaluation.py` applies those rules to normalized text. Grades are `correct`, `partial`,
`refused` or `wrong`.
**Why:** Every verdict traces back to a string in a file anyone can read and argue with. It's
deterministic and free. An LLM judge would need a stronger model than the one being graded
(an API key and cost), and its verdicts would themselves need checking.
**Tradeoff:** It only checks for the listed facts, not everything else in the answer. For
example, a baseline answer that says Netflix doesn't pay dividends and then invents
"dividend equivalent units" still counts as correct. It can also be fooled by phrasing it
doesn't anticipate, which happened once: see D34.

## D32. A held-out question set (Stage 5)

**Decision:** Besides the 8 notebook questions, 9 new questions were written in Stage 5 and
never used to tune the prompt. They include a total-debt trap and one question the documents
can't answer. Results are reported per set.
**Why:** The guardrail wording was chosen by looking at the benchmark questions (D5). Scoring
only on those would overstate how well it works.

## D33. Evaluation builds its own index and records how it ran (Stage 5)

**Decision:** `python -m eval.run_eval` indexes `data/corpus` into a temporary folder, runs
every question with and without retrieval, and writes `eval/results.md` (readable) and
`eval/results.json` (everything, including full answers). The results record the date, git
commit, device, model and settings.
**Why:** Results don't depend on whatever is in your own `chroma_db/`, and anyone can see
exactly what produced a number. This is the fix for the notebook's stale-cell problem: a
script that runs top to bottom in a fresh process. Two runs gave identical answers for all
17 questions, which also confirms greedy decoding makes runs repeatable (D6).
**Also:** The notebook's own recorded answers (`eval/notebook_answers.json`, extracted from
the notebook file) are re-scored with the same grader, so the comparison with the notebook
uses one ruler.

## D34. Summing language counts as wrong (Stage 5)

**Decision:** For the two debt questions, phrases like "adding these", "+ $" and "sum of"
are forbidden, as well as the computed totals.
**Why:** In the first run, the total-debt-trap answer began "Adding these components
together gives us: $14.5 billion + $3 billion + $42.2" and was cut off by the 200-token limit
before writing a total. No forbidden number appeared, so the grader called it correct. That
was a false pass, and it hid the most important held-out finding. A test now uses that exact
answer.

## D35. React with no component library (Stage 6)

**Decision:** Vite's `react-ts` template, four components, one plain CSS file. No UI kit,
no Tailwind, no state library.
**Why:** The UI is two forms, a list and an answer. A library would add more code to explain
than it saves.
**Tradeoff:** Styling and accessibility details (labels, focus outlines, `aria-live` regions)
are done by hand.

## D36. One origin through a proxy, no CORS (Stage 6)

**Decision:** The UI calls `/api/...`. In development Vite forwards those requests to
FastAPI; in Docker, nginx does. The API has no CORS middleware.
**Why:** The browser never makes a cross-origin request, so there's no CORS policy to
configure or get wrong. It's also how the app would sit behind a load balancer in production.

## D37. Poll the job every second (Stage 6)

**Decision:** Each upload row calls `GET /jobs/{id}` every second until it finishes.
**Why:** Jobs take seconds, and there are only ever a few in flight. Polling uses the same
endpoint any API client would use.
**Alternatives:** Server-sent events or WebSockets would push updates instantly, but add a
long-lived connection type to the server for no visible gain here.

## D38. Hand-written TypeScript types (Stage 6)

**Decision:** `frontend/src/api.ts` declares the response types by hand, mirroring
`api/schemas.py`.
**Tradeoff:** They can drift from the Pydantic models. Generating them from FastAPI's OpenAPI
schema would remove that risk; with five small types, hand-written is easier to read.

## D39. CPU-only PyTorch in the Docker image (Stage 7)

**Decision:** The API image installs PyTorch from the CPU wheel index.
**Why:** Docker on a Mac can't use the Apple GPU, and the CPU wheel is far smaller than the
default CUDA build. The app already picks the device at startup (D12), so the same image
would use a GPU where one is passed through.
**Tradeoff, measured:** In Docker on the Mac, one answer took about 39 s on CPU (float32),
against about 2 s outside Docker on the Apple GPU (float16). Docker is for "runs anywhere";
for day-to-day use on a Mac, run the API directly.

## D40. nginx serves the UI and forwards /api (Stage 7)

**Decision:** The `web` image builds the React app and serves it with nginx, which forwards
`/api/*` to the `api` service (same as the Vite proxy in development, D36). The API port is
not published; only nginx is.
**Why:** One origin, no CORS, and one public entry point. nginx also enforces
`client_max_body_size 25m`, so oversized uploads are rejected before they reach Python. That
closes the gap from D18 (verified: a 26 MB upload got 413 from nginx).

## D41. Docker Compose details (Stage 7)

**Decision:**
- Two named volumes: `data` (index, uploads, job table) and `models` (Hugging Face cache, so
  the 3 GB download happens once).
- `web` waits for `api` to be healthy. The API health check allows 10 minutes for the first
  model download.
- The path settings are set in `environment:`, which Compose applies over `.env`, so values
  copied from `.env.example` can't move data off the volumes.
- The web port is `WEB_PORT` (default 8080).
- The API runs as a non-root user.
**Verified:** Built both images, ran the stack, uploaded, queried, restarted the API and saw
the same 10 chunks, and checked the nginx upload limit.
**Found while testing:** Port 8080 was taken by another local project, which is why the port
is now configurable. And the first compose draft let `.env` override the data paths, which is
why they're pinned now.

## D42. CI: lint, tests without models, frontend build, image build (Stage 7)

**Decision:** GitHub Actions runs three jobs on every push and pull request: backend (`ruff
format --check`, `ruff check`, `pytest`), frontend (`oxlint`, strict type-check and build),
and `docker compose build`.
**Why no models in CI:** Every test uses the fake embedder, store and generator (D13), so
nothing is downloaded. CI sets `HF_HUB_OFFLINE=1`, so if a test ever did try to load a real
model it would fail instead of quietly downloading 3 GB. It installs CPU-only PyTorch because
the generator module imports it.
**Also:** `tests/test_config.py` checks that `.env.example` lists every setting, so the
example config can't fall out of date.
