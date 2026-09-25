# Stage 6: minimal React frontend

## Goal

A simple web page for the whole flow: upload a document, see its ingestion status, ask a
question, and read the answer with the sources it was based on.

## What was built

`frontend/`, scaffolded with Vite's official `react-ts` template, then stripped of the demo:

| File | What it does |
|---|---|
| `src/api.ts` | Typed client: `getHealth`, `uploadDocument`, `getJob`, `askQuestion`. Types mirror `api/schemas.py`. Turns FastAPI errors (`{"detail": ...}`, including validation lists) into one readable message |
| `src/App.tsx` | Header with "N chunks indexed"; re-checks health after each successful upload and every 3 s while the API is unreachable |
| `src/UploadPanel.tsx` | File picker (`.pdf,.txt,.md`), one row per upload, each row polls `GET /jobs/{id}` every second until `succeeded` or `failed` |
| `src/AskPanel.tsx` | Question box, the answer, and a numbered list of sources; each expands to show the exact chunk text |
| `src/index.css` | About 250 lines of plain CSS, light and dark themes, stacks on phones |
| `vite.config.ts` | Dev proxy: `/api/*` goes to the API with the `/api` prefix removed |

## How it talks to the API

```
Browser --/api/documents--> Vite dev server --/documents--> FastAPI :8000
        <-- 202 {job_id} ---
        --/api/jobs/<id>--> (every 1 s until succeeded / failed)
        --/api/query------> answer + sources
```

The browser only ever talks to one origin, so the API needs no CORS configuration. In
Docker (Stage 7), nginx serves the built files and does the same `/api` forwarding.

## Verified in a real browser

With the real API (both models loaded) and the Vite dev server:

- The first load showed "Request failed (HTTP 502)" because the API was still loading Qwen,
  and the page never retried. **Fixed:** gateway errors now read "API not reachable yet...
  it may still be loading the models", and health is re-checked every 3 s until it answers.
- Uploading `02_risk_factors.txt`: the row went to "indexed, 10 chunks" and the header
  updated to "10 chunks indexed".
- "How much debt does Netflix have?": answered "$14.5 billion of senior notes outstanding as
  of December 31, 2025", with `02_risk_factors.txt` chunk 7 (similarity 0.64) first. Expanding
  it shows the INDEBTEDNESS paragraph.
- A fake `report.pdf`: the row shows "File is named .pdf but is not a PDF" in red.
- Phone width (375 px): no horizontal scrolling; forms stack vertically.
- `npm run build` (strict type-check) and `npm run lint` pass with no warnings.

## Design decisions

- **No UI library, no CSS framework** (D35). Four small components and one stylesheet. Nothing
  here needs more, and every line is readable in an interview.
- **Same origin through a proxy** instead of CORS (D36).
- **Polling every second** instead of WebSockets or server-sent events (D37).
- **Hand-written types that mirror the Pydantic models** (D38).

## Interview talking points

- **"How does the UI know when indexing is done?"** Upload returns a job ID; each upload row
  polls `GET /jobs/{id}` once a second and stops when the job succeeds or fails. The effect
  that schedules the next poll cleans up its timer, so a row that disappears stops polling.
- **"Why a proxy instead of CORS?"** With the proxy, the browser sees one origin, so there's
  no CORS policy to get wrong. The same idea runs in production through nginx.
- **"How do errors reach the user?"** The API always returns `{"detail": ...}`. The client
  turns that into one message, shown in the upload row or under the question. Network
  failures and "API still starting" get their own wording.
- **"Why no tests for the frontend?"** It's thin: fetch, show, poll. The logic that matters
  (validation, jobs, idempotency, grading) is in Python with 170 tests. The frontend has a
  strict type-check and lint in CI, and I checked each flow in a real browser. Component tests
  would be the next step if the UI grew.

## Known limitations and what's next

- Upload rows live in memory; a page refresh forgets them (the jobs are still in the API).
- One file per upload.
- No way to ask without retrieval from the UI. The API supports it (`use_rag: false`), but it
  wasn't part of this stage.

## Handover notes

- `npm run dev` expects the API on `127.0.0.1:8000`; override with `API_URL=... npm run dev`.
- If the page says "API not reachable yet", check the API terminal: the first start downloads
  and loads the models, which takes a while.
- Types in `src/api.ts` must be updated by hand when `api/schemas.py` changes.
