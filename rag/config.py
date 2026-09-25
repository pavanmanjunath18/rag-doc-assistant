"""Central configuration, read from environment variables (or a local .env file).

Defaults match the proof notebook (RAG_project.ipynb) so results are comparable.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    """All tunable settings in one immutable object, passed to whatever needs them."""

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    gen_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    chroma_dir: Path = Path("chroma_db")
    collection: str = "docs"
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k: int = 3
    max_new_tokens: int = 200
    upload_dir: Path = Path("uploads")
    jobs_db: Path = Path("jobs.sqlite3")
    max_upload_mb: int = 20
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @classmethod
    def from_env(cls) -> Self:
        """Build settings from environment variables, falling back to the defaults above."""
        load_dotenv()
        d = cls()
        return cls(
            embed_model=os.getenv("EMBED_MODEL", d.embed_model),
            gen_model=os.getenv("GEN_MODEL", d.gen_model),
            chroma_dir=Path(os.getenv("CHROMA_DIR", str(d.chroma_dir))),
            collection=os.getenv("COLLECTION", d.collection),
            chunk_size=_int_env("CHUNK_SIZE", d.chunk_size),
            chunk_overlap=_int_env("CHUNK_OVERLAP", d.chunk_overlap),
            top_k=_int_env("TOP_K", d.top_k),
            max_new_tokens=_int_env("MAX_NEW_TOKENS", d.max_new_tokens),
            upload_dir=Path(os.getenv("UPLOAD_DIR", str(d.upload_dir))),
            jobs_db=Path(os.getenv("JOBS_DB", str(d.jobs_db))),
            max_upload_mb=_int_env("MAX_UPLOAD_MB", d.max_upload_mb),
            log_level=os.getenv("LOG_LEVEL", d.log_level),
        )


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from None
