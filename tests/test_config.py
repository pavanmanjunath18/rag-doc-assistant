"""Tests for reading settings from environment variables."""

from pathlib import Path

import pytest

from rag.config import Settings


def test_defaults_match_the_proof_notebook(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K", "MAX_NEW_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("rag.config.load_dotenv", lambda: None)
    s = Settings.from_env()
    assert (s.chunk_size, s.chunk_overlap, s.top_k, s.max_new_tokens) == (800, 150, 3, 200)


def test_environment_overrides_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "5")
    monkeypatch.setenv("UPLOAD_DIR", "/tmp/somewhere")
    monkeypatch.setenv("MAX_UPLOAD_MB", "2")
    s = Settings.from_env()
    assert s.top_k == 5
    assert s.upload_dir == Path("/tmp/somewhere")
    assert s.max_upload_bytes == 2 * 1024 * 1024


def test_non_integer_value_gives_a_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOP_K", "three")
    with pytest.raises(ValueError, match="TOP_K must be an integer, got 'three'"):
        Settings.from_env()


def test_env_example_lists_every_setting() -> None:
    # .env.example is the documentation for configuration, so it must stay complete.
    example = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    names = {line.split("=", 1)[0] for line in example.splitlines() if "=" in line}
    fields = {name.upper() for name in Settings.__dataclass_fields__}
    assert names == fields
