# Frontend

A small React + TypeScript UI (Vite, no component library): upload a document, watch its
ingestion job, ask a question, and read the answer with its cited sources.

```bash
npm install
npm run dev        # http://localhost:5173, forwards /api/* to the API on :8000
npm run build      # type-check (strict) and production build into dist/
npm run lint       # oxlint
```

Start the API first (`uvicorn --factory api.app:create_app --port 8000` from the repo root).
To point the dev server at a different API, set `API_URL`.

| File | What it does |
|---|---|
| `src/api.ts` | Typed client for the API; turns FastAPI error bodies into readable messages |
| `src/App.tsx` | Page layout and the "chunks indexed" status (retries while the API starts) |
| `src/UploadPanel.tsx` | Upload form and one row per upload, polling its job every second |
| `src/AskPanel.tsx` | Question form, answer, and expandable sources |
| `src/index.css` | All styling, light and dark |

See [docs/stages/stage-6-frontend.md](../docs/stages/stage-6-frontend.md) for the design notes.
