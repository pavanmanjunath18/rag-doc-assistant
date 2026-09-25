"""Command-line entry point.

Usage:
  python cli.py ingest data/corpus/            # index every PDF/txt/md file in a folder
  python cli.py ask "What are Netflix's main risk factors?"
  python cli.py ask "..." --no-rag             # baseline, model only
"""

import argparse
from pathlib import Path

from rag.config import Settings
from rag.factory import build_pipeline
from rag.loader import SUPPORTED_SUFFIXES
from rag.logs import configure_logging


def main() -> None:
    """Parse arguments and run the ingest or ask command."""
    parser = argparse.ArgumentParser(description="RAG document assistant")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="index a file, or every supported file in a folder")
    p_ingest.add_argument("path", type=Path)

    p_ask = sub.add_parser("ask", help="ask a question")
    p_ask.add_argument("question")
    p_ask.add_argument("--no-rag", action="store_true", help="skip retrieval (baseline)")

    args = parser.parse_args()

    settings = Settings.from_env()
    configure_logging(settings.log_level)
    pipeline = build_pipeline(settings)

    if args.cmd == "ingest":
        path: Path = args.path
        if path.is_dir():
            files = sorted(p for p in path.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES)
        else:
            files = [path]
        for f in files:
            ingested = pipeline.ingest(f)
            print(f"{ingested.status}: {ingested.source} ({ingested.chunks} chunks)")
    else:
        result = pipeline.ask(args.question, use_rag=not args.no_rag)
        print("\n" + result.text)
        if result.sources:
            print("\nSources:")
            for rank, hit in enumerate(result.sources, start=1):
                print(f"  [{rank}] {hit.source} (chunk {hit.chunk}, score {hit.score:.2f})")


if __name__ == "__main__":
    main()
