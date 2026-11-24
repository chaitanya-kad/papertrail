"""Optional dense retrieval using a sentence-transformers embedding model."""

from .documents import Passage


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class SemanticIndex:
    """Encode passages and rank them by cosine similarity to a query."""

    def __init__(self, passages: list[Passage], *, model_name: str = DEFAULT_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "Semantic search needs sentence-transformers. Install it with "
                "`python -m pip install -e '.[semantic]'`."
            ) from exc

        self.passages = passages
        self.model = SentenceTransformer(model_name)
        self.embeddings = self.model.encode(
            [passage.text for passage in passages],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(passages) > 40,
        )

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Passage, float]]:
        """Return passages ordered by cosine similarity (higher is better)."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.passages:
            return []
        query_vector = self.model.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        scores = self.embeddings @ query_vector
        ranked = sorted(range(len(self.passages)), key=lambda i: (-float(scores[i]), i))[:top_k]
        return [(self.passages[index], float(scores[index])) for index in ranked]
