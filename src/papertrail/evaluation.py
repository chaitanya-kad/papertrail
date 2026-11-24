"""Small, source-level retrieval benchmark for comparing search modes."""

import json
from pathlib import Path
import time
from typing import Any

from .documents import Passage
from .retrieval import SearchEngine


def load_queries(path: Path) -> list[dict[str, Any]]:
    """Read evaluation queries with a query string and relevant source filenames."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    queries = payload.get("queries", [])
    for item in queries:
        if not item.get("query") or not item.get("relevant_sources"):
            raise ValueError("Each evaluation query needs query and relevant_sources fields")
    return queries


def evaluate_mode(
    engine: SearchEngine,
    queries: list[dict[str, Any]],
    *,
    mode: str,
    top_k: int = 5,
) -> dict[str, float]:
    """Measure source-level Precision@k, Recall@k, MRR, and mean latency."""
    precision_total = 0.0
    recall_total = 0.0
    reciprocal_rank_total = 0.0
    latency_total = 0.0
    source_count = len({passage.source for passage in engine.passages})
    precision_denominator = max(1, min(top_k, source_count))

    def is_relevant(source: str, relevant: set[str]) -> bool:
        return source in relevant or any(source.endswith("/" + name) for name in relevant)

    for case in queries:
        relevant = {str(source) for source in case["relevant_sources"]}
        started = time.perf_counter()
        # Retrieve the full corpus before collapsing passages into source files;
        # otherwise several high-ranked chunks from one file can crowd out another.
        results = engine.search(case["query"], mode=mode, top_k=len(engine.passages))
        latency_total += time.perf_counter() - started

        # Multiple passages can come from one file; count each source once.
        ranked_sources: list[str] = []
        for result in results:
            if result.passage.source not in ranked_sources:
                ranked_sources.append(result.passage.source)
        top_sources = ranked_sources[:top_k]
        hits = sum(is_relevant(source, relevant) for source in set(top_sources))
        precision_total += hits / precision_denominator
        recall_total += hits / len(relevant)

        for rank, source in enumerate(top_sources, start=1):
            if is_relevant(source, relevant):
                reciprocal_rank_total += 1 / rank
                break

    count = max(len(queries), 1)
    return {
        f"precision@{top_k}": precision_total / count,
        f"recall@{top_k}": recall_total / count,
        "mrr": reciprocal_rank_total / count,
        "mean_latency_ms": latency_total * 1000 / count,
    }


def compare_modes(
    passages: list[Passage], queries: list[dict[str, Any]], *, top_k: int = 5
) -> dict[str, dict[str, float]]:
    """Evaluate all retrieval modes against the same query set."""
    engine = SearchEngine(passages)
    # Exclude model loading and document encoding from the timed query latency.
    from .semantic import SemanticIndex

    engine.semantic = SemanticIndex(passages)
    scores = {"keyword": evaluate_mode(engine, queries, mode="keyword", top_k=top_k)}
    scores["semantic"] = evaluate_mode(engine, queries, mode="semantic", top_k=top_k)
    scores["hybrid"] = evaluate_mode(engine, queries, mode="hybrid", top_k=top_k)
    return scores
