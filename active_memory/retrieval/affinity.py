"""Relationship-based affinity computed after semantic seed selection."""

from __future__ import annotations

from collections.abc import Sequence

from active_memory.models import MemoryEdge, RetrievalResult


def affinity_scores(results: Sequence[RetrievalResult], seeds: Sequence[RetrievalResult], edges: Sequence[MemoryEdge]) -> tuple[dict[str, float], dict[str, str]]:
    seed_scores = {seed.memory.id: seed.stage_one_score for seed in seeds}
    result_ids = {result.memory.id for result in results}
    scores = {memory_id: 0.0 for memory_id in result_ids}
    reasons: dict[str, str] = {}
    for edge in edges:
        for candidate_id, seed_id in ((edge.source_id, edge.target_id), (edge.target_id, edge.source_id)):
            if candidate_id not in result_ids or seed_id not in seed_scores:
                continue
            value = min(1.0, max(0.0, edge.weight * seed_scores[seed_id]))
            if value > scores[candidate_id]:
                scores[candidate_id] = value
                reasons[candidate_id] = f"linked to seed {seed_id} through {edge.edge_type}"
    return scores, reasons

