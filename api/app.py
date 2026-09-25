"""HTTP API: upload documents, track ingestion jobs, ask questions, check health.

Run with:  uvicorn --factory api.app:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, UploadFile

from api.schemas import (
    HealthResponse,
    JobAccepted,
    JobResponse,
    QueryRequest,
    QueryResponse,
    Source,
)
from api.uploads import (
    UploadRejected,
    check_content,
    remove_stale_temp_files,
    sanitize_filename,
    save_upload,
)
from rag.config import Settings
from rag.factory import build_pipeline
from rag.jobs import RESTART_ERROR, IngestTask, IngestWorker, JobStore
from rag.loader import SUPPORTED_SUFFIXES
from rag.logs import configure_logging
from rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_jobs(request: Request) -> JobStore:
    return request.app.state.jobs


def get_worker(request: Request) -> IngestWorker:
    return request.app.state.worker


PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
JobsDep = Annotated[JobStore, Depends(get_jobs)]
WorkerDep = Annotated[IngestWorker, Depends(get_worker)]


def create_app(settings: Settings | None = None, pipeline: RagPipeline | None = None) -> FastAPI:
    """Build the app.

    Without `pipeline`, the real models are loaded once at startup (and a failure stops the
    server from starting). Tests pass a pipeline built from fakes instead.
    """
    settings = settings or Settings.from_env()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.pipeline = pipeline or build_pipeline(settings)
        app.state.jobs = JobStore(settings.jobs_db)
        interrupted = app.state.jobs.fail_unfinished(RESTART_ERROR)
        removed = remove_stale_temp_files(settings.upload_dir)
        if interrupted or removed:
            logger.warning(
                "Previous run stopped mid-job: %d jobs marked failed, %d temp files removed",
                interrupted,
                removed,
            )
        app.state.worker = IngestWorker(app.state.pipeline, app.state.jobs)
        app.state.worker.start()
        logger.info("Ready: %d chunks indexed", app.state.pipeline.chunk_count())
        yield
        app.state.worker.stop()

    app = FastAPI(title="RAG Document Assistant", version="0.4.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    return app


# Endpoints are plain `def`, not `async def`: embedding and generation are slow, blocking
# calls, so FastAPI runs them in a worker thread instead of stalling the event loop.


@router.get("/health")
def health(pipeline: PipelineDep) -> HealthResponse:
    """Report that the service is up and how many chunks are indexed."""
    return HealthResponse(status="ok", chunks_indexed=pipeline.chunk_count())


@router.post("/documents", status_code=202)
def upload_document(
    file: UploadFile,
    response: Response,
    settings: SettingsDep,
    jobs: JobsDep,
    worker: WorkerDep,
) -> JobAccepted:
    """Upload a PDF, .txt or .md file. It is checked now and indexed in the background.

    Returns 202 with a job ID; poll `GET /jobs/{job_id}` for the result.
    """
    filename = sanitize_filename(file.filename or "")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        allowed = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise HTTPException(415, f"Unsupported file type. Allowed: {allowed}")

    try:
        tmp = save_upload(file.file, settings.upload_dir, suffix, settings.max_upload_bytes)
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    try:
        check_content(tmp, suffix)
    except UploadRejected as exc:
        tmp.unlink(missing_ok=True)
        raise HTTPException(exc.status_code, str(exc)) from exc

    job = jobs.create(filename)
    worker.submit(
        IngestTask(job_id=job.id, path=tmp, source=filename, keep_as=settings.upload_dir / filename)
    )
    response.headers["Location"] = f"/jobs/{job.id}"
    logger.info("Queued %s as job %s", filename, job.id)
    return JobAccepted(job_id=job.id, filename=filename, status=job.status)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, jobs: JobsDep) -> JobResponse:
    """Return an ingestion job's status, its result when done, or its error if it failed."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, f"No job with id {job_id}")
    return JobResponse.from_job(job)


@router.post("/query")
def query(body: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
    """Answer a question from the indexed documents, citing the chunks used."""
    answer = pipeline.ask(body.question, use_rag=body.use_rag, top_k=body.top_k)
    sources = [
        Source(source=hit.source, chunk=hit.chunk, score=hit.score, text=hit.text)
        for hit in answer.sources
    ]
    return QueryResponse(answer=answer.text, sources=sources)
