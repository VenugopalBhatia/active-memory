# Scoring

Every signal is independently normalized to `[0, 1]`.

## Relevance

```text
relevance = clip((cosine_similarity + 1) / 2, 0, 1)
```

All vectors are normalized at the provider boundary. Exact search retains raw cosine for inspection.

## Recency

```text
recency = exp(-ln(2) * age_seconds / half_life_seconds[memory_type])
```

Defaults range from seven days for task state to ten years for preferences. Recency does not change validity; superseded or expired records are filtered separately.

## Frequency

```text
frequency = log1p(inclusion_count) / log1p(max_inclusion_count)
```

`inclusion_count` changes only after final context packing. Candidate retrieval, inspection, and scoring are read-only.

## Stage one

```text
stage_one = 0.65 * relevance + 0.25 * recency + 0.10 * frequency
```

The highest stage-one records become seeds.

## Affinity and final score

```text
affinity(candidate, seeds) = max(edge.weight * seed.stage_one_score)
final = 0.55 * relevance + 0.20 * recency + 0.10 * frequency + 0.15 * affinity
```

Affinity is zero without an explicit edge. Embedding similarity never creates an edge. Defaults are policy choices, not learned weights.

## Weaknesses

Type-specific decay is coarse, frequency can reinforce repeatedly useful but outdated content, deterministic relationships miss implicit dependencies, and a single linear score cannot represent every query intent. Ablations must be consulted before claiming a signal improves a dataset.

