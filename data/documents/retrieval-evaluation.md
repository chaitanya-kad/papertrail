# Evaluating document retrieval

Retrieval evaluation measures whether a search system returns useful documents near the top of its results. A small evaluation set contains search questions and the passages that should be considered relevant for each question. The same set can be used to compare keyword search, semantic search, and hybrid search.

Precision at k asks what fraction of the first k results are relevant. Recall at k asks what fraction of all known relevant passages appear in the first k results. Mean reciprocal rank rewards a system when its first relevant result appears near the top. These metrics describe different aspects of search quality, so a project should report which one it is optimizing.

The evaluation set should be written before tuning ranking parameters. Otherwise, repeatedly adjusting the system against the same questions can make the reported score look better without improving search on new questions. Keep a small held-out set for a final check.
