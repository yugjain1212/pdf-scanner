"""Command-line interface for PDF page similarity search."""

import argparse
import sys
import textwrap
from collections.abc import Sequence

from src.pdf_reader import PDFReaderError, extract_pages
from src.search_engine import PDFSearchEngine, PDFSearchError, SearchResult

MAX_EXCERPT_CHARACTERS = 1200


def create_excerpt(text: str, max_chars: int = MAX_EXCERPT_CHARACTERS) -> str:
    """Limit displayed page text without changing the text used for scoring."""
    if max_chars < 1:
        raise ValueError("max_chars must be at least 1.")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _create_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find the PDF page most similar to a query using TF-IDF "
            "and cosine similarity."
        )
    )
    parser.add_argument("pdf_path", help="path to a selectable-text PDF")
    parser.add_argument("query", help="words or phrases to search for")
    parser.add_argument(
        "--top-k",
        type=int,
        default=1,
        metavar="N",
        help="also list the N most relevant pages (default: 1)",
    )
    return parser


def _print_results(
    pdf_path: str,
    query: str,
    results: Sequence[SearchResult],
    top_k: int,
) -> None:
    best_result = results[0]

    print("========================================")
    print("       PDF PAGE SIMILARITY SEARCH")
    print("========================================")
    print(f"\nPDF:\n{pdf_path}")
    print(f"\nQuery:\n{query}")

    if top_k > 1:
        print(f"\nTop {len(results)} Relevant Pages")
        print("=" * 27)
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank}. Page {result['page_number']}"
                f"    Score: {result['score']:.4f}"
            )

    print(f"\nMost Relevant Page:\n{best_result['page_number']}")
    print(f"\nSimilarity Score:\n{best_result['score']:.4f}")

    is_excerpt = len(best_result["text"]) > MAX_EXCERPT_CHARACTERS
    label = "Relevant Text (excerpt):" if is_excerpt else "Relevant Text:"
    excerpt = create_excerpt(best_result["text"])
    print(f"\n{label}")
    print("----------------------------------------")
    print(textwrap.fill(excerpt, width=80))
    print("----------------------------------------")


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run the search, and return a process exit code."""
    parser = _create_argument_parser()
    arguments = parser.parse_args(argv)

    if arguments.top_k < 1:
        print("Error: --top-k must be at least 1.", file=sys.stderr)
        return 2
    if not arguments.query.strip():
        print("Error: Query cannot be empty.", file=sys.stderr)
        return 1

    try:
        pages = extract_pages(arguments.pdf_path)
        search_engine = PDFSearchEngine(pages)
        results = search_engine.search_top_k(arguments.query, arguments.top_k)
    except (FileNotFoundError, PDFReaderError, PDFSearchError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    _print_results(arguments.pdf_path, arguments.query, results, arguments.top_k)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
