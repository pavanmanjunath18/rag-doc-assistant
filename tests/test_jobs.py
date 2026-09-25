"""Tests for the job table and the background worker's job states."""

import threading
from pathlib import Path

import pytest

from rag.jobs import IngestTask, IngestWorker, JobStatus, JobStore
from rag.loader import DocumentError
from rag.pipeline import IngestResult, IngestStatus, RagPipeline


@pytest.fixture
def jobs(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.sqlite3")


def result(status: IngestStatus = IngestStatus.INDEXED) -> IngestResult:
    return IngestResult(doc_id="abc123", source="report.txt", chunks=4, status=status)


# --- JobStore ---------------------------------------------------------------------------


def test_new_job_is_queued(jobs: JobStore) -> None:
    job = jobs.create("report.txt")
    stored = jobs.get(job.id)
    assert stored == job
    assert stored.status is JobStatus.QUEUED
    assert stored.started_at is None and stored.finished_at is None


def test_job_moves_through_running_to_succeeded(jobs: JobStore) -> None:
    job = jobs.create("report.txt")
    jobs.mark_running(job.id)
    assert jobs.get(job.id).status is JobStatus.RUNNING
    assert jobs.get(job.id).started_at is not None

    jobs.mark_succeeded(job.id, result())
    done = jobs.get(job.id)
    assert done.status is JobStatus.SUCCEEDED
    assert (done.document_id, done.cited_as, done.chunks) == ("abc123", "report.txt", 4)
    assert done.ingest_status is IngestStatus.INDEXED
    assert done.finished_at is not None and done.error is None


def test_failed_job_keeps_the_error(jobs: JobStore) -> None:
    job = jobs.create("report.txt")
    jobs.mark_failed(job.id, "No text extracted from report.txt")
    failed = jobs.get(job.id)
    assert failed.status is JobStatus.FAILED
    assert failed.error == "No text extracted from report.txt"


def test_unknown_job_is_none(jobs: JobStore) -> None:
    assert jobs.get("nope") is None


def test_jobs_survive_reopening_the_database(tmp_path: Path) -> None:
    job = JobStore(tmp_path / "jobs.sqlite3").create("report.txt")
    assert JobStore(tmp_path / "jobs.sqlite3").get(job.id) is not None


def test_fail_unfinished_only_touches_queued_and_running(jobs: JobStore) -> None:
    queued = jobs.create("a.txt")
    running = jobs.create("b.txt")
    jobs.mark_running(running.id)
    done = jobs.create("c.txt")
    jobs.mark_succeeded(done.id, result())

    assert jobs.fail_unfinished("restarted") == 2
    assert jobs.get(queued.id).status is JobStatus.FAILED
    assert jobs.get(running.id).error == "restarted"
    assert jobs.get(done.id).status is JobStatus.SUCCEEDED


def test_update_rejects_unknown_columns(jobs: JobStore) -> None:
    job = jobs.create("a.txt")
    with pytest.raises(ValueError, match="Unknown job columns"):
        jobs._update(job.id, **{"status = 'x'; --": "y"})


# --- IngestWorker -----------------------------------------------------------------------


class ScriptedPipeline:
    """Stands in for RagPipeline.ingest: can block, raise, or return a chosen result."""

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.outcome: IngestResult | Exception = result()
        self.order: list[str] = []

    def ingest(self, path: Path, source: str | None = None) -> IngestResult:
        self.order.append(source or path.name)
        self.started.set()
        self.release.wait(timeout=5)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def scripted() -> ScriptedPipeline:
    return ScriptedPipeline()


@pytest.fixture
def worker(scripted: ScriptedPipeline, jobs: JobStore) -> IngestWorker:
    w = IngestWorker(scripted, jobs)  # type: ignore[arg-type]  # duck-typed stand-in
    w.start()
    yield w
    w.stop()


def submit(worker: IngestWorker, jobs: JobStore, tmp_path: Path, name: str) -> tuple[str, Path]:
    job = jobs.create(name)
    path = tmp_path / f".upload-{name}"
    path.write_text("content", encoding="utf-8")
    worker.submit(IngestTask(job.id, path, name, keep_as=tmp_path / name))
    return job.id, path


def test_job_is_running_while_ingesting_then_succeeds(
    worker: IngestWorker, scripted: ScriptedPipeline, jobs: JobStore, tmp_path: Path
) -> None:
    scripted.release.clear()
    job_id, tmp = submit(worker, jobs, tmp_path, "report.txt")

    assert scripted.started.wait(timeout=5)
    assert jobs.get(job_id).status is JobStatus.RUNNING

    scripted.release.set()
    worker.wait_until_idle()
    assert jobs.get(job_id).status is JobStatus.SUCCEEDED
    assert not tmp.exists()
    assert (tmp_path / "report.txt").read_text() == "content"


def test_unchanged_result_does_not_keep_the_file(
    worker: IngestWorker, scripted: ScriptedPipeline, jobs: JobStore, tmp_path: Path
) -> None:
    scripted.outcome = result(IngestStatus.UNCHANGED)
    job_id, tmp = submit(worker, jobs, tmp_path, "report.txt")
    worker.wait_until_idle()
    assert jobs.get(job_id).status is JobStatus.SUCCEEDED
    assert not tmp.exists()
    assert not (tmp_path / "report.txt").exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (DocumentError("No text extracted from report.txt"), "No text extracted from report.txt"),
        (RuntimeError("boom"), "Unexpected error: RuntimeError: boom"),
    ],
)
def test_errors_fail_the_job_and_clean_up(
    worker: IngestWorker,
    scripted: ScriptedPipeline,
    jobs: JobStore,
    tmp_path: Path,
    error: Exception,
    message: str,
) -> None:
    scripted.outcome = error
    job_id, tmp = submit(worker, jobs, tmp_path, "report.txt")
    worker.wait_until_idle()
    job = jobs.get(job_id)
    assert job.status is JobStatus.FAILED
    assert job.error == message
    assert not tmp.exists()


def test_jobs_run_one_at_a_time_in_order(
    worker: IngestWorker, scripted: ScriptedPipeline, jobs: JobStore, tmp_path: Path
) -> None:
    scripted.release.clear()
    first, _ = submit(worker, jobs, tmp_path, "a.txt")
    second, _ = submit(worker, jobs, tmp_path, "b.txt")
    assert scripted.started.wait(timeout=5)

    assert jobs.get(first).status is JobStatus.RUNNING
    assert jobs.get(second).status is JobStatus.QUEUED

    scripted.release.set()
    worker.wait_until_idle()
    assert scripted.order == ["a.txt", "b.txt"]
    assert jobs.get(second).status is JobStatus.SUCCEEDED


def test_stop_finishes_queued_jobs_first(
    scripted: ScriptedPipeline, jobs: JobStore, tmp_path: Path
) -> None:
    worker = IngestWorker(scripted, jobs)  # type: ignore[arg-type]
    worker.start()
    job_id, _ = submit(worker, jobs, tmp_path, "report.txt")
    worker.stop()
    assert jobs.get(job_id).status is JobStatus.SUCCEEDED


def test_ingest_signature_matches_real_pipeline() -> None:
    # Guards the stand-in above from drifting away from the real method.
    assert (
        ScriptedPipeline.ingest.__code__.co_varnames[:3]
        == (RagPipeline.ingest.__code__.co_varnames[:3])
    )
