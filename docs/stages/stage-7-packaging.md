# Stage 7: Docker, CI and configuration

## Goal

Make the project easy to run and hard to break: one command to start everything, CI on every
push, a complete `.env.example`, and tests that never download models.

## What was built

| File | What it does |
|---|---|
| `Dockerfile` | API image: Python 3.11 slim, CPU-only PyTorch, app code, non-root user, health check |
| `.dockerignore` | Keeps tests, docs, data, the venv and local indexes out of the image |
| `frontend/Dockerfile` | Builds the UI with Node 22, serves `dist/` with nginx |
| `frontend/nginx.conf` | Serves the UI, forwards `/api/*` to the API, 25 MB upload cap, 300 s timeout |
| `docker-compose.yml` | `api` + `web`, volumes for data and models, `web` waits for a healthy `api` |
| `.github/workflows/ci.yml` | Three jobs: backend lint + tests, frontend lint + build, Docker build |
| `.env.example` | Every setting, with a note on which ones Compose overrides |
| `tests/test_config.py` | New test: `.env.example` must list every `Settings` field |

## How the containers fit together

```
browser --:8080--> web (nginx)
                     /        -> the built React files
                     /api/*   -> api:8000 (not published to the host)
                                   |
                                   +-- volume "data":   /data/chroma, /data/uploads, /data/jobs.sqlite3
                                   +-- volume "models": /models (Hugging Face cache)
```

## Verified

Locally, with Docker Desktop on the Mac:
- `docker compose build` built both images in about 2 minutes.
- The stack came up healthy (using the local model cache, to skip the 3 GB download).
- Through nginx on the published port: `/api/health` returned ok; uploading
  `02_risk_factors.txt` gave a job that succeeded in about 1 second with 10 chunks; the debt
  question answered "$14.5 billion of senior notes" in 39 s on CPU; a 26 MB upload was
  rejected by nginx with 413.
- After `docker compose restart api`, `/health` still showed 10 chunks, so the index survives
  restarts.
- The test suite passes with `HF_HUB_OFFLINE=1` (171 tests, under a second).

In GitHub Actions, the first run passed all three jobs: backend (ruff format, ruff lint,
171 tests in 1.5 s with Hugging Face offline), frontend (lint, strict build) and the
Docker image build.

## Two things testing caught

1. **Port 8080 was already in use** by another project on the same machine. The web port is
   now `WEB_PORT` (default 8080).
2. **`.env` could silently move the data.** `.env.example` has `CHROMA_DIR=chroma_db`, which
   is right for local runs. Copied into `.env` and loaded by Compose, it would have put the
   index inside the container: not on the volume, and not writable by the non-root user.
   Compose's `environment:` takes precedence over `env_file:`, so the three paths are pinned
   there.

## How to run it

```bash
docker compose up --build                 # first start downloads the models; be patient
open http://localhost:8080                # or WEB_PORT=8090 docker compose up --build
docker compose down                       # keeps the data and models volumes
docker compose down -v                    # also deletes them
```

## Interview talking points

- **"How do your tests avoid loading models?"** The pipeline receives its embedder, store and
  generator, so tests pass fakes. CI also sets Hugging Face to offline mode, so any accidental
  model load fails instead of downloading.
- **"Why is the Docker version slower?"** Docker on a Mac has no access to the Apple GPU, so
  the model runs on CPU in float32: about 39 s per answer against about 2 s natively. I
  measured it and wrote it down instead of hiding it. On a Linux host with an NVIDIA GPU, the
  same code would pick CUDA.
- **"What does nginx give you?"** One origin (no CORS), static file serving, and an upload size
  limit enforced before the request reaches Python.
- **"How do you keep config documented?"** `.env.example` lists every setting, and a test fails
  if a new setting isn't added to it.

## Known limitations and what's next

- No GPU support in Compose (it would need the NVIDIA container toolkit and a `deploy` block).
- The API image is large because of PyTorch (hundreds of MB, even as the CPU build).
- CI doesn't run the model-based evaluation; that needs the real models and a few minutes of
  compute, so it stays a manual `python -m eval.run_eval`.

## Handover notes

- First `docker compose up` can take several minutes before the UI loads: the API downloads
  about 3 GB of models, then loads them. `docker compose logs -f api` shows progress.
- To reuse a model cache you already have, mount it over `/models`, for example
  `~/.cache/huggingface:/models`, in a local override file.
- Data lives in the `data` volume. `docker compose down -v` deletes it.
