"""Central configuration, read from environment variables (or a local .env file).

Defaults match the proof notebook (RAG_project.ipynb) so results are comparable.
"""

import os

from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GEN_MODEL: str = os.getenv("GEN_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
CHROMA_DIR: str = os.getenv("CHROMA_DIR", "chroma_db")
COLLECTION: str = os.getenv("COLLECTION", "docs")
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))
TOP_K: int = int(os.getenv("TOP_K", "3"))
MAX_NEW_TOKENS: int = int(os.getenv("MAX_NEW_TOKENS", "200"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
