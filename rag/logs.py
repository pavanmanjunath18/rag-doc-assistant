"""Logging setup shared by the CLI and the API."""

import logging


def configure_logging(level: str) -> None:
    """Show this project's logs at `level`, and only warnings from third-party libraries."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    for name in ("rag", "api"):
        logging.getLogger(name).setLevel(level)
