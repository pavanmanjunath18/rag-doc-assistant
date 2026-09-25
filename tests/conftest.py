"""Shared fixtures: a pipeline built from fakes, and an API client using it."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from rag.config import Settings
from rag.pipeline import RagPipeline
from tests.fakes import FakeEmbedder, FakeGenerator, InMemoryStore


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        chroma_dir=tmp_path / "chroma", upload_dir=tmp_path / "uploads", max_upload_mb=1
    )


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def pipeline(settings: Settings, store: InMemoryStore, generator: FakeGenerator) -> RagPipeline:
    return RagPipeline(
        FakeEmbedder(),
        store,
        generator,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        top_k=settings.top_k,
    )


@pytest.fixture
def client(settings: Settings, pipeline: RagPipeline) -> Iterator[TestClient]:
    with TestClient(create_app(settings, pipeline)) as test_client:
        yield test_client
