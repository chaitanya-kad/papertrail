# PaperTrail architecture

## Data flow

1. **Load:** `documents.py` reads Markdown and text files, and extracts PDF text page by page when the `pdf` extra is installed.
2. **Chunk:** each text unit is split into word windows. Each passage retains its relative filename and, for PDFs, its page number.
3. **Index:** `bm25.py` builds term frequencies, document frequencies, and postings lists. `semantic.py` optionally creates normalized sentence embeddings.
4. **Retrieve:** `retrieval.py` returns BM25 scores, cosine similarity scores, or Reciprocal Rank Fusion (RRF).
5. **Evaluate:** `evaluation.py` compares methods against the same labeled query set using source-level Precision@k, Recall@k, MRR, and mean latency.
6. **Present:** the CLI and Streamlit app display passages with source references so a reader can inspect the original material.

## Why Reciprocal Rank Fusion?

BM25 scores and cosine similarities use different scales, so directly averaging them would let one score type dominate. RRF uses each method's rank instead:

`RRF(document) = sum(1 / (c + rank(document)))`

PaperTrail uses `c = 60`. A passage ranked highly by both systems receives a higher combined score.

## Boundaries

- The initial index is built in memory from local files. It is suitable for a personal collection and learning experiments, not millions of documents.
- Semantic mode downloads the selected sentence-transformers model the first time it is used, then uses the local cached copy.
- Search results are retrieved evidence, not generated answers. This keeps citations inspectable and avoids requiring a hosted LLM or API key.
- Scanned PDFs without an embedded text layer need OCR, which is outside this version.
- Evaluation labels cover the included sample documents only; meaningful project results require a larger, independently labeled corpus.
