# Evaluation

The offline harness uses twelve hand-authored cases: long-distance recall, changing decisions, conflicting facts, cross-session memory, related code entities, unresolved tasks, repeated irrelevant content, assistant correction, large tool output, privacy-sensitive content, expiry, and stable preferences.

Baselines are recent-window only, vector top-k only, relevance/recency/frequency, and the full system. Ablations run the full system without affinity, frequency, or recency, plus vector-only and recent-only.

Reported retrieval metrics are Recall@K, Precision@K, MRR, nDCG@K, stale-memory rate, superseded-memory rate, and provenance accuracy. Context helpers measure relevant-token ratio, contradiction and duplicate rates, budget adherence, coverage, and trust-weighted precision. Exact-search latency reports median and p95 wall time at multiple corpus sizes.

The deterministic lexical test provider makes CI reproducible but is not semantic. Production semantic quality must also be evaluated with the configured local or external embedding model. No downstream model calls are made by the default benchmark, so answer correctness, task completion, hallucination rate, and provider cost require a separate live evaluation and are not claimed here.

