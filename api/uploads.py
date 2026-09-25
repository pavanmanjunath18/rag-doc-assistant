"""Upload validation helpers, kept free of FastAPI so they can be unit tested on their own."""

import re
import tempfile
import unicodedata
from pathlib import Path
from typing import BinaryIO

MAX_FILENAME_LENGTH = 100
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_READ_SIZE = 1024 * 1024
_SNIFF_SIZE = 8192
_PDF_MAGIC = b"%PDF-"
_TEMP_PREFIX = ".upload-"


class UploadRejected(Exception):
    """The upload can't be accepted. Carries the HTTP status code to return."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def sanitize_filename(raw: str) -> str:
    """Reduce a client-supplied filename to a safe base name.

    Drops any directory part (so "../../etc/passwd" can't escape the upload folder), keeps
    only letters, digits, ".", "_" and "-", strips leading/trailing dots, and caps the length
    while keeping the extension.
    """
    name = unicodedata.normalize("NFKC", raw).replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_CHARS.sub("_", name).strip("._")
    stem, dot, suffix = name.rpartition(".")
    if not dot:
        stem, suffix = name, ""
    stem = stem[: MAX_FILENAME_LENGTH - len(suffix) - 1] or "upload"
    return f"{stem}.{suffix}" if suffix else stem


def save_upload(src: BinaryIO, directory: Path, suffix: str, max_bytes: int) -> Path:
    """Copy an upload into a temporary file in `directory`, enforcing a size limit.

    Reads in 1 MB blocks and stops as soon as the limit is passed, so an oversized file is
    never fully written. Returns the temporary file's path; the caller moves or deletes it.
    """
    directory.mkdir(parents=True, exist_ok=True)
    total = 0
    with tempfile.NamedTemporaryFile(
        dir=directory, prefix=_TEMP_PREFIX, suffix=suffix, delete=False
    ) as out:
        tmp = Path(out.name)
        try:
            while block := src.read(_READ_SIZE):
                total += len(block)
                if total > max_bytes:
                    limit_mb = max_bytes // (1024 * 1024)
                    raise UploadRejected(413, f"File is larger than the {limit_mb} MB limit")
                out.write(block)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    if total == 0:
        tmp.unlink(missing_ok=True)
        raise UploadRejected(400, "File is empty")
    return tmp


def check_content(path: Path, suffix: str) -> None:
    """Check that the file's bytes match its extension (the name alone proves nothing)."""
    with path.open("rb") as f:
        head = f.read(_SNIFF_SIZE)
    if suffix == ".pdf" and not head.startswith(_PDF_MAGIC):
        raise UploadRejected(415, "File is named .pdf but is not a PDF")
    if suffix in {".txt", ".md"} and b"\x00" in head:
        raise UploadRejected(415, "File is named as text but contains binary data")


def remove_stale_temp_files(directory: Path) -> int:
    """Delete temp uploads left behind by a server that stopped mid-job. Returns the count."""
    if not directory.exists():
        return 0
    stale = list(directory.glob(f"{_TEMP_PREFIX}*"))
    for path in stale:
        path.unlink(missing_ok=True)
    return len(stale)
