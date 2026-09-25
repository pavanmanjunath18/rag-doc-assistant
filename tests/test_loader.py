"""Tests for document loading and chunking (no models needed)."""

from pathlib import Path

import pytest

from rag.loader import DocumentError, chunk_text, load_text


def numbered_words(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def test_chunks_respect_size() -> None:
    chunks = chunk_text("word " * 1000, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)


def test_consecutive_chunks_share_whole_words() -> None:
    chunks = chunk_text(numbered_words(500), size=200, overlap=50)
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        # compare whole tokens, so "w1" can't falsely match inside "w10"
        assert prev.split()[-1] in nxt.split()


def test_every_chunk_is_made_of_whole_words() -> None:
    text = numbered_words(500)
    original = set(text.split())
    for chunk in chunk_text(text, size=200, overlap=50):
        assert set(chunk.split()) <= original


def test_figures_are_never_split() -> None:
    text = " ".join(["$45,183,036"] * 200)
    for chunk in chunk_text(text, size=100, overlap=30):
        assert set(chunk.split()) == {"$45,183,036"}


def test_all_words_are_covered() -> None:
    text = numbered_words(500)
    covered = {word for chunk in chunk_text(text, size=200, overlap=50) for word in chunk.split()}
    assert covered == set(text.split())


def test_line_breaks_are_kept() -> None:
    text = "INDEBTEDNESS\nAs of December 31, 2025, Netflix had $14.5 billion of senior notes."
    assert chunk_text(text, size=800, overlap=150) == [text]


def test_short_text_is_single_chunk() -> None:
    assert chunk_text("hello world", size=200, overlap=50) == ["hello world"]


def test_empty_text_gives_no_chunks() -> None:
    assert chunk_text("   \n  ", size=200, overlap=50) == []


def test_text_without_whitespace_still_terminates() -> None:
    text = "x" * 450
    chunks = chunk_text(text, size=200, overlap=50)
    assert all(len(c) <= 200 for c in chunks)
    assert chunks[-1].endswith("x")


@pytest.mark.parametrize(("size", "overlap"), [(100, 100), (100, 150), (0, 0), (100, -1)])
def test_invalid_size_or_overlap_raises(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_text("abc", size=size, overlap=overlap)


def test_load_text_reads_txt(tmp_path: Path) -> None:
    doc = tmp_path / "notes.txt"
    doc.write_text("HEADING\nbody text", encoding="utf-8")
    assert load_text(doc) == "HEADING\nbody text"


def test_load_text_replaces_invalid_bytes_and_warns(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    doc = tmp_path / "latin1.txt"
    doc.write_bytes(b"caf\xe9 revenue")
    text = load_text(doc)
    assert text == "caf\N{REPLACEMENT CHARACTER} revenue"
    assert "not valid UTF-8" in caplog.text


def test_load_text_rejects_empty_file(tmp_path: Path) -> None:
    doc = tmp_path / "blank.txt"
    doc.write_text("  \n ", encoding="utf-8")
    with pytest.raises(DocumentError, match="No text extracted"):
        load_text(doc)


def test_load_text_rejects_unsupported_type(tmp_path: Path) -> None:
    doc = tmp_path / "sheet.xlsx"
    doc.write_bytes(b"not really a spreadsheet")
    with pytest.raises(DocumentError, match="Unsupported file type"):
        load_text(doc)


def test_load_text_wraps_corrupt_pdf_errors(tmp_path: Path) -> None:
    doc = tmp_path / "broken.pdf"
    doc.write_bytes(b"%PDF-1.4 but then garbage with no structure")
    with pytest.raises(DocumentError, match="Could not read PDF broken.pdf"):
        load_text(doc)
