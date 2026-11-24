# Keyword and semantic search

Keyword search matches terms that appear in both the question and a document. BM25 is a ranking method that gives more weight to rare terms, accounts for repeated occurrences, and adjusts for document length. It works especially well when a query uses the same vocabulary as the relevant passage.

Semantic search represents text as a numeric embedding. Passages with similar meanings can be retrieved even when the wording differs. For example, a query asking how a system finds relevant text may match a passage about information retrieval, even if the exact words are different. Embeddings add useful recall but require a model and more storage than a simple inverted index.

Hybrid search combines lexical and semantic evidence. A fair comparison should use the same documents and evaluation queries for each approach, then report retrieval metrics and query latency.
