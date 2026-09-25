"""Tests for upload helpers: filename sanitizing, size-limited saving, content checks."""

import io
from pathlib import Path

import pytest

from api.uploads import (
    MAX_FILENAME_LENGTH,
    UploadRejected,
    check_content,
    sanitize_filename,
    save_upload,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.txt", "evil.txt"),
        ("/absolute/path/notes.md", "notes.md"),
        ("Q3 results (final).pdf", "Q3_results_final_.pdf"),
        (".hidden.txt", "hidden.txt"),
        ("..", "upload"),
        ("", "upload"),
        ("résumé.txt", "r_sum_.txt"),
        ("ｆｕｌｌｗｉｄｔｈ.txt", "fullwidth.txt"),
        ("trailing.txt.", "trailing.txt"),
    ],
)
def test_sanitize_filename(raw: str, expected: str) -> None:
    assert sanitize_filename(raw) == expected


def test_long_names_are_capped_but_keep_the_extension() -> None:
    name = sanitize_filename("a" * 500 + ".pdf")
    assert len(name) == MAX_FILENAME_LENGTH
    assert name.endswith(".pdf")


def test_save_upload_writes_a_temp_file(tmp_path: Path) -> None:
    saved = save_upload(io.BytesIO(b"hello"), tmp_path, ".txt", max_bytes=10)
    assert saved.parent == tmp_path
    assert saved.suffix == ".txt"
    assert saved.read_bytes() == b"hello"


def test_save_upload_stops_at_the_limit_and_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as exc:
        save_upload(io.BytesIO(b"x" * 11), tmp_path, ".txt", max_bytes=10)
    assert exc.value.status_code == 413
    assert list(tmp_path.iterdir()) == []


def test_save_upload_rejects_empty_and_cleans_up(tmp_path: Path) -> None:
    with pytest.raises(UploadRejected) as exc:
        save_upload(io.BytesIO(b""), tmp_path, ".txt", max_bytes=10)
    assert exc.value.status_code == 400
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("suffix", "content", "ok"),
    [
        (".pdf", b"%PDF-1.7 rest", True),
        (".pdf", b"<html>not a pdf</html>", False),
        (".txt", "plain text, with unicode: ünïcödé".encode(), True),
        (".md", b"# heading\x00binary", False),
    ],
)
def test_check_content(tmp_path: Path, suffix: str, content: bytes, ok: bool) -> None:
    path = tmp_path / f"file{suffix}"
    path.write_bytes(content)
    if ok:
        check_content(path, suffix)
    else:
        with pytest.raises(UploadRejected) as exc:
            check_content(path, suffix)
        assert exc.value.status_code == 415
