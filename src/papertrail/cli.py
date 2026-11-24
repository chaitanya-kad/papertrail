"""Command-line search and evaluation commands."""

import argparse
from pathlib import Path

from .documents import load_documents, split_into_passages
from .evaluation import compare_modes, load_queries
from .retrieval import SearchEngine


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_passages(directory: Path):
    return split_into_passages(load_documents(directory))


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a local research library with PaperTrail.")
    parser.add_argument("--query", help="Question or words to search for")
    parser.add_argument("--mode", choices=["keyword", "semantic", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--documents", type=Path, default=PROJECT_ROOT / "data" / "documents",
        help="Folder containing .md, .txt, and (with the pdf extra) .pdf files",
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Compare all retrieval modes on the sample benchmark"
    )
    args = parser.parse_args()

    if not args.evaluate and not args.query:
        parser.error("provide --query or use --evaluate")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    passages = _load_passages(args.documents)
    if not passages:
        parser.error(f"No searchable documents found in {args.documents}")

    if args.evaluate:
        queries_path = PROJECT_ROOT / "data" / "evaluation" / "queries.json"
        queries = load_queries(queries_path)
        try:
            scores = compare_modes(passages, queries, top_k=args.top_k)
        except RuntimeError as exc:
            parser.error(str(exc))
        print(f"Source-level benchmark ({len(queries)} queries, k={args.top_k})")
        print(f"{'Mode':<12} {'P@k':>8} {'R@k':>8} {'MRR':>8} {'Latency':>12}")
        for mode, values in scores.items():
            print(
                f"{mode:<12} {values[f'precision@{args.top_k}']:>8.3f} "
                f"{values[f'recall@{args.top_k}']:>8.3f} {values['mrr']:>8.3f} "
                f"{values['mean_latency_ms']:>9.1f} ms"
            )
        return

    engine = SearchEngine(passages)
    try:
        results = engine.search(args.query, mode=args.mode, top_k=args.top_k)
    except RuntimeError as exc:
        parser.error(str(exc))
    if not results:
        print("No matching passages found. Try a different query or search mode.")
        return

    for rank, result in enumerate(results, start=1):
        location = result.passage.source
        if result.passage.page is not None:
            location += f", page {result.passage.page}"
        print(f"{rank}. {location} · passage {result.passage.number} · score {result.score:.4f}")
        print(f"   {result.passage.text}\n")


if __name__ == "__main__":
    main()
