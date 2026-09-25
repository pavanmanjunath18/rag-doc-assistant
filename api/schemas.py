"""Request and response models for the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from rag.jobs import Job, JobStatus
from rag.pipeline import IngestStatus


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks_indexed: int


class JobAccepted(BaseModel):
    job_id: str
    filename: str = Field(description="Sanitized name the document will be cited under")
    status: JobStatus


class IngestOutcome(BaseModel):
    document_id: str = Field(description="SHA-256 of the file's bytes")
    filename: str = Field(description="Name the document is cited under")
    chunks: int = Field(description="Number of chunks stored for this document")
    status: IngestStatus = Field(
        description="indexed (new), replaced (new version of a same-named file) or "
        "unchanged (identical content was already indexed)"
    )


class JobResponse(BaseModel):
    job_id: str
    filename: str
    status: JobStatus = Field(description="queued, running, succeeded or failed")
    created_at: str
    started_at: str | None
    finished_at: str | None
    error: str | None = Field(description="Why the job failed (null unless status is failed)")
    result: IngestOutcome | None = Field(description="Set when status is succeeded")

    @classmethod
    def from_job(cls, job: Job) -> "JobResponse":
        result = None
        if job.document_id and job.cited_as and job.chunks is not None and job.ingest_status:
            result = IngestOutcome(
                document_id=job.document_id,
                filename=job.cited_as,
                chunks=job.chunks,
                status=job.ingest_status,
            )
        return cls(
            job_id=job.id,
            filename=job.filename,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
            result=result,
        )


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)
    top_k: int | None = Field(
        default=None, ge=1, le=10, description="Chunks to retrieve (server default if omitted)"
    )
    use_rag: bool = Field(default=True, description="False answers without retrieval (baseline)")

    @field_validator("question")
    @classmethod
    def question_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class Source(BaseModel):
    source: str
    chunk: int
    score: float = Field(description="Cosine similarity, higher is more relevant")
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(description="Retrieved chunks, most relevant first")
