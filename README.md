# PaperTrail

**Search research documents and inspect the evidence behind every result.** PaperTrail is a local-first document search app that compares keyword retrieval, semantic retrieval, and a hybrid ranking.

It is designed as a learning project: the BM25 index and ranking flow are implemented in this repository, while a small sentence-transformers model provides dense embeddings for semantic search.

## What it does

- Reads Markdown and plain-text files, plus PDFs when the PDF extra is installed.
- Splits content into overlapping passages while preserving source filenames and PDF page numbers.
- Ranks passages with a hand-built BM25 inverted index.
- Optionally finds semantically similar passages with `all-MiniLM-L6-v2`.
- Combines both rankings with Reciprocal Rank Fusion (RRF).
- Compares retrieval methods using a small hand-labeled benchmark.
- Provides a Streamlit interface and a command-line interface.
- Keeps documents and retrieval local; it does not call a hosted LLM or need an API key.

## Quick start

Requires Python 3.10 or newer.

```bash
git clone https://github.com/chaitanya-kad/papertrail.git
cd papertrail
python -m pip install -e ".[full]"
streamlit run src/papertrail/app.py
```

The first semantic or hybrid search downloads the embedding model once. The model is then cached by `sentence-transformers` on your machine.

For keyword search only, install the dependency-free core:

```bash
python -m pip install -e .
papertrail --mode keyword --query "how is retrieval quality measured?"
```

To add only the browser app and PDF support, without semantic retrieval:

```bash
python -m pip install -e ".[app,pdf]"
streamlit run src/papertrail/app.py
```

## Add your documents

Place `.md` and `.txt` files in `data/documents/`. The app's sidebar can also accept `.md`, `.txt`, and `.pdf` uploads; those files are stored under `data/uploads/`. PDF support requires the `pdf` or `full` extra. PaperTrail shows the filename and page number for each matching passage.

The sample documents in `data/documents/` explain chunking, retrieval evaluation, and keyword versus semantic search. They let you try the project before adding your own files.

## Search from the command line

```bash
papertrail --mode keyword --query "rare terms and exact phrases"
papertrail --mode semantic --query "find passages with similar meaning"
papertrail --mode hybrid --query "how can I evaluate a search engine?" --top-k 5
```

Semantic and hybrid modes require the `semantic` extra:

```bash
python -m pip install -e ".[semantic]"
```

## Compare search methods

The included benchmark is labeled at the source-file level. It reports Precision@k, Recall@k, Mean Reciprocal Rank (MRR), and average query latency for all three retrieval methods:

```bash
papertrail --evaluate --top-k 3
```

The labels and queries are in [`data/evaluation/queries.json`](data/evaluation/queries.json). Treat the scores as a small demonstration, not as evidence that one retriever is universally better. For a credible experiment, grow the query set, label it independently, and reserve held-out queries for a final comparison.

## How retrieval works

### BM25 keyword search

PaperTrail tokenizes each passage and stores term frequencies, document frequencies, and a postings list from terms to passages. BM25 gives more weight to rare query terms, accounts for repeated terms, and adjusts for passage length. This is a useful baseline when the query and document share vocabulary.

### Semantic search

The sentence-transformers model maps each passage and query to a normalized vector. Cosine similarity ranks passages whose meaning is close, even when they use different words. The model is pretrained; PaperTrail does not train a new embedding model.

### Hybrid search

BM25 scores and cosine similarities are on different scales, so PaperTrail combines their rankings with Reciprocal Rank Fusion:

`RRF(passage) = Σ 1 / (60 + rank)`

A passage ranked highly by both systems receives a stronger fused rank. See [`docs/architecture.md`](docs/architecture.md) for the data flow and design trade-offs.

## Project structure

```text
src/papertrail/
  documents.py    file loading, PDF page extraction, passage chunking
  bm25.py         inverted index and BM25 ranking
  semantic.py     optional sentence-transformers embeddings
  retrieval.py    keyword, semantic, and hybrid search interface
  evaluation.py   Precision@k, Recall@k, MRR, and latency
  cli.py          terminal search and benchmark commands
  app.py          Streamlit application
data/
  documents/      small sample research notes
  evaluation/     hand-labeled query set
docs/
  architecture.md design notes and limitations
```

## Current limits

- The index is rebuilt in memory when the library changes; this version targets personal collections, not millions of documents.
- Scanned PDFs without selectable text need OCR, which is not included.
- The included benchmark is intentionally small and uses sample notes.
- Search returns source-linked passages rather than generated answers. This keeps the retrieved evidence visible and avoids sending private files to an external model.

## Learning path

1. Read `documents.py` and follow one file through loading and chunking.
2. Trace a query through `bm25.py` and calculate one BM25 score by hand.
3. Compare the keyword, semantic, and hybrid result lists in the app.
4. Add search questions and relevance labels, then study how each metric changes.
5. Change one ranking or chunking choice at a time and record the effect.

The goal is not just a working search box; it is a retrieval pipeline you can explain from input files through evaluation.
