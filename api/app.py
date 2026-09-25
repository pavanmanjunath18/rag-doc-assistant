"""HTTP API: upload documents, ask questions, check health.

Run with:  uvicorn --factory api.app:create_app
"""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, UploadFile

from api.schemas import DocumentResponse, HealthResponse, QueryRequest, QueryResponse, Source
from api.uploads import UploadRejected, check_content, sanitize_filename, save_upload
from rag.config import Settings
from rag.factory import build_pipeline
from rag.loader import SUPPORTED_SUFFIXES, DocumentError
from rag.logs import configure_logging
from rag.pipeline import IngestStatus, RagPipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_pipeline(request: Request) -> RagPipeline:
    return request.app.state.pipeline


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
SettingsDep = Annotated[Settings, Depends(get_settings)]


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
        logger.info("Ready: %d chunks indexed", app.state.pipeline.chunk_count())
        yield

    app = FastAPI(title="RAG Document Assistant", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.include_router(router)
    return app


# Endpoints are plain `def`, not `async def`: embedding and generation are slow, blocking
# calls, so FastAPI runs them in a worker thread instead of stalling the event loop.


@router.get("/health")
def health(pipeline: PipelineDep) -> HealthResponse:
    """Report that the service is up and how many chunks are indexed."""
    return HealthResponse(status="ok", chunks_indexed=pipeline.chunk_count())


@router.post("/documents", status_code=201)
def upload_document(
    file: UploadFile, response: Response, pipeline: PipelineDep, settings: SettingsDep
) -> DocumentResponse:
    """Upload a PDF, .txt or .md file and index it.

    Returns 201 when something was indexed, 200 when identical content was already indexed.
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
        result = pipeline.ingest(tmp, source=filename)
        if result.status is IngestStatus.UNCHANGED:
            response.status_code = 200
        else:
            tmp.replace(settings.upload_dir / filename)
    except UploadRejected as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    except DocumentError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        tmp.unlink(missing_ok=True)

    logger.info("Upload %s: %s (%d chunks)", filename, result.status, result.chunks)
    return DocumentResponse(
        document_id=result.doc_id,
        filename=result.source,
        chunks=result.chunks,
        status=result.status,
    )


@router.post("/query")
def query(body: QueryRequest, pipeline: PipelineDep) -> QueryResponse:
    """Answer a question from the indexed documents, citing the chunks used."""
    answer = pipeline.ask(body.question, use_rag=body.use_rag, top_k=body.top_k)
    sources = [
        Source(source=hit.source, chunk=hit.chunk, score=hit.score, text=hit.text)
        for hit in answer.sources
    ]
    return QueryResponse(answer=answer.text, sources=sources)
