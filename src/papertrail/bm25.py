"""A compact Okapi BM25 index that keeps the ranking mechanics visible."""

from collections import Counter, defaultdict
import math

from .documents import Passage, tokenize


class BM25Index:
    """Build an in-memory inverted index and rank passages with BM25."""

    def __init__(self, passages: list[Passage], *, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.passages = passages
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.lengths: list[int] = []
        self.postings: dict[str, list[int]] = defaultdict(list)

        for passage_id, passage in enumerate(passages):
            frequencies = Counter(tokenize(passage.text))
            self.term_frequencies.append(frequencies)
            self.lengths.append(sum(frequencies.values()))
            for term in frequencies:
                self.document_frequencies[term] += 1
                self.postings[term].append(passage_id)

        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0

    def search(self, query: str, *, top_k: int = 5) -> list[tuple[Passage, float]]:
        """Return the top passages as (passage, score) pairs."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self.passages:
            return []

        scores: Counter[int] = Counter()
        for term in tokenize(query):
            document_frequency = self.document_frequencies.get(term, 0)
            if not document_frequency:
                continue
            inverse_frequency = math.log(
                1 + (len(self.passages) - document_frequency + 0.5) / (document_frequency + 0.5)
            )

            for passage_index in self.postings[term]:
                frequency = self.term_frequencies[passage_index][term]
                length = self.lengths[passage_index]
                length_norm = 1 - self.b + self.b * length / self.average_length
                scores[passage_index] += inverse_frequency * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_norm)
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [(self.passages[index], score) for index, score in ranked]
