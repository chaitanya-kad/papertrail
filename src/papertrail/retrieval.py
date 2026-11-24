"""Combine lexical, semantic, and reciprocal-rank-fusion retrieval."""

from dataclasses import dataclass

from .bm25 import BM25Index
from .documents import Passage
from .semantic import SemanticIndex


@dataclass(frozen=True)
class SearchResult:
    passage: Passage
    score: float
    method: str


class SearchEngine:
    """Search passages with BM25, embeddings, or reciprocal-rank fusion."""

    def __init__(self, passages: list[Passage], semantic: SemanticIndex | None = None):
        self.passages = passages
        self.bm25 = BM25Index(passages)
        self.semantic = semantic

    def search(self, query: str, *, mode: str = "keyword", top_k: int = 5) -> list[SearchResult]:
        if mode not in {"keyword", "semantic", "hybrid"}:
            raise ValueError("mode must be keyword, semantic, or hybrid")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.passages or not query.strip():
            return []

        if mode == "keyword":
            return [SearchResult(p, score, mode) for p, score in self.bm25.search(query, top_k=top_k)]

        if self.semantic is None:
            self.semantic = SemanticIndex(self.passages)
        if mode == "semantic":
            return [SearchResult(p, score, mode) for p, score in self.semantic.search(query, top_k=top_k)]

        candidate_count = min(len(self.passages), max(top_k * 4, 20))
        lexical = self.bm25.search(query, top_k=candidate_count)
        dense = self.semantic.search(query, top_k=candidate_count)
        # Reciprocal Rank Fusion combines rankings without comparing unlike score scales.
        fused: dict[str, tuple[Passage, float]] = {}
        rank_constant = 60
        for ranking in (lexical, dense):
            for rank, (passage, _score) in enumerate(ranking, start=1):
                current = fused.get(passage.passage_id, (passage, 0.0))
                fused[passage.passage_id] = (passage, current[1] + 1 / (rank_constant + rank))

        results = sorted(fused.values(), key=lambda item: (-item[1], item[0].passage_id))[:top_k]
        return [SearchResult(passage, score, mode) for passage, score in results]
