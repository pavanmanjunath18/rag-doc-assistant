"""Background ingestion: a SQLite table of jobs and one worker thread that runs them.

One worker, one job at a time: there is a single embedding model in memory anyway, and running
jobs in order means two uploads of the same file name can't interleave (see D26).
"""

import logging
import queue
import sqlite3
import threading
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from rag.loader import DocumentError
from rag.pipeline import IngestResult, IngestStatus, RagPipeline

logger = logging.getLogger(__name__)

RESTART_ERROR = "The server restarted before this job finished. Upload the file again."

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id            TEXT PRIMARY KEY,
    filename      TEXT NOT NULL,
    status        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    finished_at   TEXT,
    error         TEXT,
    document_id   TEXT,
    cited_as      TEXT,
    chunks        INTEGER,
    ingest_status TEXT
)
"""


_UPDATABLE_COLUMNS = frozenset(
    {
        "status",
        "started_at",
        "finished_at",
        "error",
        "document_id",
        "cited_as",
        "chunks",
        "ingest_status",
    }
)


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Job:
    id: str
    filename: str
    status: JobStatus
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    error: str | None = None
    document_id: str | None = None
    cited_as: str | None = None  # may differ from filename if the content was already indexed
    chunks: int | None = None
    ingest_status: IngestStatus | None = None


class JobStore:
    """Job records in a SQLite file. Each call opens its own connection, so it's thread-safe."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(_SCHEMA)

    def create(self, filename: str) -> Job:
        """Record a new queued job and return it."""
        job = Job(
            id=uuid.uuid4().hex, filename=filename, status=JobStatus.QUEUED, created_at=_now()
        )
        with self._connect() as db:
            db.execute(
                "INSERT INTO jobs (id, filename, status, created_at) VALUES (?, ?, ?, ?)",
                (job.id, job.filename, job.status, job.created_at),
            )
        return job

    def get(self, job_id: str) -> Job | None:
        """Return the job, or None if there is no job with this ID."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_job(row) if row else None

    def mark_running(self, job_id: str) -> None:
        self._update(job_id, status=JobStatus.RUNNING, started_at=_now())

    def mark_succeeded(self, job_id: str, result: IngestResult) -> None:
        self._update(
            job_id,
            status=JobStatus.SUCCEEDED,
            finished_at=_now(),
            document_id=result.doc_id,
            cited_as=result.source,
            chunks=result.chunks,
            ingest_status=result.status,
        )

    def mark_failed(self, job_id: str, error: str) -> None:
        self._update(job_id, status=JobStatus.FAILED, finished_at=_now(), error=error)

    def fail_unfinished(self, reason: str) -> int:
        """Mark every queued or running job as failed. Used at startup after a restart."""
        with self._connect() as db:
            cursor = db.execute(
                "UPDATE jobs SET status = ?, finished_at = ?, error = ? WHERE status IN (?, ?)",
                (JobStatus.FAILED, _now(), reason, JobStatus.QUEUED, JobStatus.RUNNING),
            )
        return cursor.rowcount

    def _update(self, job_id: str, **fields: object) -> None:
        # Column names come only from the fixed set below, never from user input;
        # values always go through "?" placeholders.
        unknown = set(fields) - _UPDATABLE_COLUMNS
        if unknown:
            raise ValueError(f"Unknown job columns: {sorted(unknown)}")
        columns = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as db:
            db.execute(f"UPDATE jobs SET {columns} WHERE id = ?", (*fields.values(), job_id))

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self._path)
        db.row_factory = sqlite3.Row
        try:
            with db:  # commits on success, rolls back on error
                yield db
        finally:
            db.close()


@dataclass(frozen=True)
class IngestTask:
    job_id: str
    path: Path  # the uploaded temp file; deleted when the job ends
    source: str  # name to cite the document under
    keep_as: Path  # where to keep the file if it was indexed


class IngestWorker:
    """Runs ingestion tasks one at a time on a background thread."""

    def __init__(self, pipeline: RagPipeline, jobs: JobStore) -> None:
        self._pipeline = pipeline
        self._jobs = jobs
        self._queue: queue.Queue[IngestTask | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="ingest-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        """Finish the jobs already queued, then stop the thread."""
        self._queue.put(None)
        self._thread.join()

    def submit(self, task: IngestTask) -> None:
        self._queue.put(task)

    def wait_until_idle(self) -> None:
        """Block until every submitted task has finished (used by tests)."""
        self._queue.join()

    def _run(self) -> None:
        while (task := self._queue.get()) is not None:
            try:
                self._process(task)
            finally:
                self._queue.task_done()
        self._queue.task_done()

    def _process(self, task: IngestTask) -> None:
        self._jobs.mark_running(task.job_id)
        try:
            result = self._pipeline.ingest(task.path, source=task.source)
            if result.status is not IngestStatus.UNCHANGED:
                task.path.replace(task.keep_as)
            self._jobs.mark_succeeded(task.job_id, result)
        except DocumentError as exc:
            self._jobs.mark_failed(task.job_id, str(exc))
        except Exception as exc:
            logger.exception("Job %s failed", task.job_id)
            self._jobs.mark_failed(task.job_id, f"Unexpected error: {type(exc).__name__}: {exc}")
        finally:
            task.path.unlink(missing_ok=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        filename=row["filename"],
        status=JobStatus(row["status"]),
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        error=row["error"],
        document_id=row["document_id"],
        cited_as=row["cited_as"],
        chunks=row["chunks"],
        ingest_status=IngestStatus(row["ingest_status"]) if row["ingest_status"] else None,
    )
