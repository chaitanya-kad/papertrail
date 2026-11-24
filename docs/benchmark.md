# Understanding the benchmark

PaperTrail's included benchmark has five example questions. Each question names the source files that contain relevant material. The evaluator searches the passages, collapses multiple hits from the same file, and compares the first `k` unique sources with the labels.

| Metric | What it tells you |
| --- | --- |
| Precision@k | Among the first `k` source results (or all available sources when fewer than `k` exist), what fraction are labeled relevant? |
| Recall@k | Of all labeled relevant sources, what fraction appeared in the first `k`? |
| MRR | How high is the first relevant source on average? |
| Mean latency | Average retrieval time for one query in this local run. |

The evaluation data is intentionally tiny so it is easy to inspect and edit. It is not a statistically reliable model comparison. To make a stronger experiment, use documents representative of the intended collection, create diverse questions, have relevance labels reviewed, and hold some questions out from tuning.
