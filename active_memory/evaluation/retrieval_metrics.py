"""Standard retrieval metrics with stale-record accounting."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    stale_rate: float
    superseded_rate: float
    provenance_accuracy: float


def measure(ranked_ids: Sequence[str], relevant_ids: set[str], *, k: int, stale_ids: set[str] | None = None, superseded_ids: set[str] | None = None, provenance_correct: set[str] | None = None) -> RetrievalMetrics:
    top = list(ranked_ids[:k])
    hits = [memory_id for memory_id in top if memory_id in relevant_ids]
    recall = len(set(hits)) / len(relevant_ids) if relevant_ids else 1.0
    precision = len(hits) / len(top) if top else (1.0 if not relevant_ids else 0.0)
    reciprocal_rank = next((1.0 / (index + 1) for index, memory_id in enumerate(ranked_ids) if memory_id in relevant_ids), 0.0)
    dcg = sum((1.0 if memory_id in relevant_ids else 0.0) / math.log2(index + 2) for index, memory_id in enumerate(top))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(k, len(relevant_ids))))
    stale = len(set(top) & (stale_ids or set())) / len(top) if top else 0.0
    superseded = len(set(top) & (superseded_ids or set())) / len(top) if top else 0.0
    provenance = len(set(hits) & (provenance_correct or relevant_ids)) / len(hits) if hits else (1.0 if not relevant_ids else 0.0)
    return RetrievalMetrics(recall, precision, reciprocal_rank, dcg / ideal if ideal else 1.0, stale, superseded, provenance)

