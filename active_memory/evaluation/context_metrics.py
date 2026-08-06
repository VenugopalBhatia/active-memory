"""Metrics for assembled context content and budget behavior."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ContextMetrics:
    relevant_token_ratio: float
    contradiction_rate: float
    duplicate_rate: float
    budget_adherence: float
    coverage: float
    trust_weighted_precision: float


def measure_context(*, selected_ids: list[str], relevant_ids: set[str], token_costs: dict[str, int], contradiction_ids: set[str], duplicate_ids: set[str], input_tokens: int, input_budget: int, trust_weights: dict[str, float]) -> ContextMetrics:
    total_tokens = sum(token_costs.get(memory_id, 0) for memory_id in selected_ids)
    relevant_tokens = sum(token_costs.get(memory_id, 0) for memory_id in selected_ids if memory_id in relevant_ids)
    count = len(selected_ids)
    weighted_total = sum(trust_weights.get(memory_id, 0.0) for memory_id in selected_ids)
    weighted_relevant = sum(trust_weights.get(memory_id, 0.0) for memory_id in selected_ids if memory_id in relevant_ids)
    return ContextMetrics(
        relevant_tokens / total_tokens if total_tokens else (1.0 if not relevant_ids else 0.0),
        len(set(selected_ids) & contradiction_ids) / count if count else 0.0,
        len(set(selected_ids) & duplicate_ids) / count if count else 0.0,
        1.0 if input_tokens <= input_budget else 0.0,
        len(set(selected_ids) & relevant_ids) / len(relevant_ids) if relevant_ids else 1.0,
        weighted_relevant / weighted_total if weighted_total else (1.0 if not relevant_ids else 0.0),
    )

