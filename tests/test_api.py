"""API tests: real FastAPI app, pipeline and job worker, with fake models and an in-memory store."""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.config import Settings
from rag.jobs import RESTART_ERROR, JobStore
from rag.pipeline import EMPTY_INDEX_ANSWER, RagPipeline
from tests.fakes import FakeEmbedder, FakeGenerator

DEBT = b"INDEBTEDNESS\nNetflix had $14.5 billion of senior notes outstanding."
REVENUE = b"REVENUES\nTotal revenues were $45.2 billion, up 16 percent."


def upload(client: TestClient, name: str, content: bytes) -> Any:
    return client.post("/documents", files={"file": (name, content)})


def ingest(client: TestClient, name: str, content: bytes) -> dict[str, Any]:
    """Upload, wait for the background job to finish, and return the finished job."""
    response = upload(client, name, content)
    assert response.status_code == 202, response.text
    client.app.state.worker.wait_until_idle()
    return client.get(f"/jobs/{response.json()['job_id']}").json()


def leftover_files(settings: Settings) -> list[str]:
    if not settings.upload_dir.exists():
        return []
    return sorted(p.name for p in settings.upload_dir.iterdir())


# --- health -----------------------------------------------------------------------------


def test_health_reports_indexed_chunks(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 0}
    ingest(client, "debt.txt", DEBT)
    assert client.get("/health").json()["chunks_indexed"] == 1


# --- upload and jobs --------------------------------------------------------------------


def test_upload_returns_202_with_a_job_to_poll(client: TestClient) -> None:
    response = upload(client, "debt.txt", DEBT)
    assert response.status_code == 202
    body = response.json()
    assert body["filename"] == "debt.txt"
    assert body["status"] in {"queued", "running", "succeeded"}  # the worker may be quick
    assert response.headers["Location"] == f"/jobs/{body['job_id']}"


def test_successful_job_reports_the_result(client: TestClient, settings: Settings) -> None:
    job = ingest(client, "debt.txt", DEBT)

    assert job["status"] == "succeeded"
    assert job["error"] is None
    assert job["started_at"] and job["finished_at"]
    assert job["result"] == {
        "document_id": hashlib.sha256(DEBT).hexdigest(),
        "filename": "debt.txt",
        "chunks": 1,
        "status": "indexed",
    }
    assert (settings.upload_dir / "debt.txt").read_bytes() == DEBT
    assert leftover_files(settings) == ["debt.txt"]


def test_uploading_the_same_file_again_is_a_no_op(
    client: TestClient, settings: Settings, embedder: FakeEmbedder
) -> None:
    first = ingest(client, "debt.txt", DEBT)
    embedded = embedder.texts_embedded

    second = ingest(client, "debt.txt", DEBT)

    assert second["result"] == {**first["result"], "status": "unchanged"}
    assert embedder.texts_embedded == embedded
    assert client.get("/health").json()["chunks_indexed"] == 1
    assert leftover_files(settings) == ["debt.txt"]


def test_uploading_a_changed_file_replaces_it(client: TestClient, settings: Settings) -> None:
    ingest(client, "report.txt", DEBT)
    job = ingest(client, "report.txt", REVENUE)

    assert job["result"]["status"] == "replaced"
    assert client.get("/health").json()["chunks_indexed"] == 1
    sources = client.post("/query", json={"question": "senior notes revenues"}).json()["sources"]
    assert [s["text"] for s in sources] == [REVENUE.decode()]
    assert (settings.upload_dir / "report.txt").read_bytes() == REVENUE


@pytest.mark.parametrize(
    ("name", "content", "error"),
    [
        ("blank.txt", b"   \n\n  ", "No text extracted from"),
        ("broken.pdf", b"%PDF-1.4 then garbage", "Could not read PDF broken.pdf"),
    ],
)
def test_unreadable_document_fails_its_job_with_the_reason(
    client: TestClient, settings: Settings, name: str, content: bytes, error: str
) -> None:
    job = ingest(client, name, content)
    assert job["status"] == "failed"
    assert error in job["error"]
    assert name in job["error"] and ".upload-" not in job["error"]  # user's name, not temp name
    assert job["result"] is None
    assert leftover_files(settings) == []


def test_unexpected_error_fails_the_job_and_the_worker_keeps_going(
    client: TestClient, pipeline: RagPipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_ingest = pipeline.ingest
    calls = {"n": 0}

    def flaky_ingest(path: Path, source: str | None = None) -> Any:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("disk full")
        return real_ingest(path, source=source)

    monkeypatch.setattr(pipeline, "ingest", flaky_ingest)

    failed = ingest(client, "debt.txt", DEBT)
    succeeded = ingest(client, "revenue.txt", REVENUE)

    assert failed["status"] == "failed"
    assert failed["error"] == "Unexpected error: RuntimeError: disk full"
    assert succeeded["status"] == "succeeded"


def test_unknown_job_is_404(client: TestClient) -> None:
    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404
    assert "No job with id" in response.json()["detail"]


def test_filename_is_sanitized(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    job = ingest(client, "../../evil name.txt", DEBT)
    assert job["filename"] == "evil_name.txt"
    assert leftover_files(settings) == ["evil_name.txt"]
    # nothing was written outside the upload folder (the jobs database is ours)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["jobs.sqlite3", "uploads"]


def test_restart_fails_unfinished_jobs_and_removes_temp_files(
    settings: Settings, pipeline: RagPipeline
) -> None:
    from api.app import create_app

    jobs = JobStore(settings.jobs_db)
    stuck = jobs.create("report.txt")
    settings.upload_dir.mkdir(parents=True)
    (settings.upload_dir / ".upload-abc.txt").write_bytes(DEBT)

    with TestClient(create_app(settings, pipeline)) as client:
        job = client.get(f"/jobs/{stuck.id}").json()

    assert job["status"] == "failed"
    assert job["error"] == RESTART_ERROR
    assert leftover_files(settings) == []


# --- upload: rejected before a job is created -------------------------------------------


@pytest.mark.parametrize(
    ("name", "content", "status", "message"),
    [
        ("sheet.xlsx", b"data", 415, "Unsupported file type"),
        ("no_extension", b"data", 415, "Unsupported file type"),
        ("fake.pdf", b"just text, not a pdf", 415, "not a PDF"),
        ("binary.txt", b"abc\x00\x01\x02", 415, "binary data"),
        ("empty.txt", b"", 400, "File is empty"),
        ("big.txt", b"a" * (1024 * 1024 + 1), 413, "1 MB limit"),
    ],
)
def test_bad_uploads_are_rejected_immediately(
    client: TestClient, settings: Settings, name: str, content: bytes, status: int, message: str
) -> None:
    response = upload(client, name, content)
    assert response.status_code == status
    assert message in response.json()["detail"]
    assert leftover_files(settings) == []
    assert client.get("/health").json()["chunks_indexed"] == 0


def test_upload_without_file_field_is_rejected(client: TestClient) -> None:
    assert client.post("/documents").status_code == 422


# --- query ------------------------------------------------------------------------------


def test_query_returns_answer_with_ranked_sources(client: TestClient) -> None:
    ingest(client, "debt.txt", DEBT)
    ingest(client, "revenue.txt", REVENUE)

    response = client.post("/query", json={"question": "How much in senior notes?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FakeGenerator.answer
    assert [s["source"] for s in body["sources"]] == ["debt.txt", "revenue.txt"]
    assert body["sources"][0]["text"] == DEBT.decode()
    assert body["sources"][0]["score"] > body["sources"][1]["score"]


def test_query_top_k_limits_sources(client: TestClient) -> None:
    ingest(client, "debt.txt", DEBT)
    ingest(client, "revenue.txt", REVENUE)
    response = client.post("/query", json={"question": "senior notes", "top_k": 1})
    assert len(response.json()["sources"]) == 1


def test_query_baseline_has_no_sources(client: TestClient, generator: FakeGenerator) -> None:
    ingest(client, "debt.txt", DEBT)
    response = client.post("/query", json={"question": "  How much debt?  ", "use_rag": False})
    assert response.json()["sources"] == []
    assert generator.calls == [[{"role": "user", "content": "How much debt?"}]]


def test_query_with_nothing_indexed(client: TestClient, generator: FakeGenerator) -> None:
    response = client.post("/query", json={"question": "Anything?"})
    assert response.status_code == 200
    assert response.json() == {"answer": EMPTY_INDEX_ANSWER, "sources": []}
    assert generator.calls == []


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 1001},
        {"question": "ok", "top_k": 0},
        {"question": "ok", "top_k": 11},
        {"question": "ok", "unexpected": True},
    ],
)
def test_invalid_query_is_rejected(
    client: TestClient, generator: FakeGenerator, body: dict[str, Any]
) -> None:
    response = client.post("/query", json=body)
    assert response.status_code == 422
    assert generator.calls == []
