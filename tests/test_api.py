"""API tests: real FastAPI app and pipeline, with fake models and an in-memory store."""

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag.config import Settings
from rag.pipeline import EMPTY_INDEX_ANSWER
from tests.fakes import FakeEmbedder, FakeGenerator

DEBT = b"INDEBTEDNESS\nNetflix had $14.5 billion of senior notes outstanding."
REVENUE = b"REVENUES\nTotal revenues were $45.2 billion, up 16 percent."


def upload(client: TestClient, name: str, content: bytes) -> Any:
    return client.post("/documents", files={"file": (name, content)})


def leftover_files(settings: Settings) -> list[str]:
    if not settings.upload_dir.exists():
        return []
    return sorted(p.name for p in settings.upload_dir.iterdir())


# --- health -----------------------------------------------------------------------------


def test_health_reports_indexed_chunks(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "chunks_indexed": 0}
    upload(client, "debt.txt", DEBT)
    assert client.get("/health").json()["chunks_indexed"] == 1


# --- upload: happy path -----------------------------------------------------------------


def test_upload_indexes_and_saves_the_file(client: TestClient, settings: Settings) -> None:
    response = upload(client, "debt.txt", DEBT)
    assert response.status_code == 201
    assert response.json() == {
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
    first = upload(client, "debt.txt", DEBT).json()
    embedded = embedder.texts_embedded

    response = upload(client, "debt.txt", DEBT)

    assert response.status_code == 200
    assert response.json() == {**first, "status": "unchanged"}
    assert embedder.texts_embedded == embedded
    assert client.get("/health").json()["chunks_indexed"] == 1
    assert leftover_files(settings) == ["debt.txt"]


def test_uploading_a_changed_file_replaces_it(client: TestClient, settings: Settings) -> None:
    upload(client, "report.txt", DEBT)
    response = upload(client, "report.txt", REVENUE)

    assert response.status_code == 201
    assert response.json()["status"] == "replaced"
    assert client.get("/health").json()["chunks_indexed"] == 1
    sources = client.post("/query", json={"question": "senior notes revenues"}).json()["sources"]
    assert [s["text"] for s in sources] == [REVENUE.decode()]
    assert (settings.upload_dir / "report.txt").read_bytes() == REVENUE


def test_minimal_pdf_is_accepted_past_validation(client: TestClient) -> None:
    # A real PDF header with no extractable text: passes the type checks, then fails
    # extraction with a clear 422 rather than a server error.
    response = upload(client, "scan.pdf", b"%PDF-1.4\n%%EOF\n")
    assert response.status_code == 422


def test_filename_is_sanitized(client: TestClient, settings: Settings, tmp_path: Path) -> None:
    response = upload(client, "../../evil name.txt", DEBT)
    assert response.status_code == 201
    assert response.json()["filename"] == "evil_name.txt"
    assert leftover_files(settings) == ["evil_name.txt"]
    # nothing was written outside the upload folder
    assert list(tmp_path.iterdir()) == [settings.upload_dir]


# --- upload: rejected -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "content", "status", "message"),
    [
        ("sheet.xlsx", b"data", 415, "Unsupported file type"),
        ("no_extension", b"data", 415, "Unsupported file type"),
        ("fake.pdf", b"just text, not a pdf", 415, "not a PDF"),
        ("binary.txt", b"abc\x00\x01\x02", 415, "binary data"),
        ("empty.txt", b"", 400, "File is empty"),
        ("blank.txt", b"   \n\n  ", 422, "No text extracted"),
        ("big.txt", b"a" * (1024 * 1024 + 1), 413, "1 MB limit"),
    ],
)
def test_bad_uploads_are_rejected_with_clear_errors(
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
    upload(client, "debt.txt", DEBT)
    upload(client, "revenue.txt", REVENUE)

    response = client.post("/query", json={"question": "How much in senior notes?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == FakeGenerator.answer
    assert [s["source"] for s in body["sources"]] == ["debt.txt", "revenue.txt"]
    assert body["sources"][0]["text"] == DEBT.decode()
    assert body["sources"][0]["score"] > body["sources"][1]["score"]


def test_query_top_k_limits_sources(client: TestClient) -> None:
    upload(client, "debt.txt", DEBT)
    upload(client, "revenue.txt", REVENUE)
    response = client.post("/query", json={"question": "senior notes", "top_k": 1})
    assert len(response.json()["sources"]) == 1


def test_query_baseline_has_no_sources(client: TestClient, generator: FakeGenerator) -> None:
    upload(client, "debt.txt", DEBT)
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
