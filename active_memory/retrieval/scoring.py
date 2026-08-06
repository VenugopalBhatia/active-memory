"""Independent, bounded scoring signals for first-pass retrieval."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from active_memory.models import Memory, RetrievalResult
from active_memory.retrieval.candidates import Candidate


def _validate_weight(value: float, name: str) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within [0, 1]")


@dataclass(frozen=True, slots=True)
class StageOneWeights:
    relevance: float = 0.65
    recency: float = 0.25
    frequency: float = 0.10

    def __post_init__(self) -> None:
        for name in ("relevance", "recency", "frequency"):
            _validate_weight(getattr(self, name), name)
        if not math.isclose(self.relevance + self.recency + self.frequency, 1.0, abs_tol=1e-9):
            raise ValueError("stage-one weights must sum to 1.0")


DEFAULT_HALF_LIVES: Mapping[str, timedelta] = {
    "task": timedelta(days=7),
    "error": timedelta(days=14),
    "resolution": timedelta(days=30),
    "tool_observation": timedelta(days=30),
    "code_change": timedelta(days=60),
    "episode": timedelta(days=90),
    "fact": timedelta(days=180),
    "decision": timedelta(days=365),
    "summary": timedelta(days=365),
    "preference": timedelta(days=3650),
}


@dataclass(frozen=True, slots=True)
class ScoringPolicy:
    stage_one: StageOneWeights = field(default_factory=StageOneWeights)
    half_lives: Mapping[str, timedelta] = field(default_factory=lambda: dict(DEFAULT_HALF_LIVES))

    def __post_init__(self) -> None:
        if any(value.total_seconds() <= 0 for value in self.half_lives.values()):
            raise ValueError("recency half-lives must be positive")


def relevance_score(cosine_similarity: float) -> float:
    return min(1.0, max(0.0, (cosine_similarity + 1.0) / 2.0))


def recency_score(memory: Memory, now: datetime, half_lives: Mapping[str, timedelta] = DEFAULT_HALF_LIVES) -> float:
    half_life = half_lives.get(memory.memory_type, timedelta(days=90)).total_seconds()
    age_seconds = max(0.0, (now - (memory.valid_from or memory.created_at)).total_seconds())
    return min(1.0, max(0.0, math.exp(-math.log(2.0) * age_seconds / half_life)))


def frequency_score(inclusion_count: int, maximum_inclusion_count: int) -> float:
    if inclusion_count <= 0 or maximum_inclusion_count <= 0:
        return 0.0
    return min(1.0, max(0.0, math.log1p(inclusion_count) / math.log1p(maximum_inclusion_count)))


def score_stage_one(candidates: Sequence[Candidate], now: datetime, policy: ScoringPolicy | None = None) -> list[RetrievalResult]:
    policy = policy or ScoringPolicy()
    maximum = max((candidate.memory.inclusion_count for candidate in candidates), default=0)
    results: list[RetrievalResult] = []
    for candidate in candidates:
        relevance = relevance_score(candidate.cosine_similarity)
        recency = recency_score(candidate.memory, now, policy.half_lives)
        frequency = frequency_score(candidate.memory.inclusion_count, maximum)
        weights = policy.stage_one
        score = min(1.0, max(0.0, weights.relevance * relevance + weights.recency * recency + weights.frequency * frequency))
        age_days = max(0.0, (now - candidate.memory.created_at).total_seconds()) / 86400.0
        reasons = [f"semantic similarity: {candidate.cosine_similarity:.3f}", f"observed {age_days:.1f} days ago"]
        if candidate.memory.inclusion_count:
            reasons.append(f"included in {candidate.memory.inclusion_count} previous contexts")
        reasons.append(f"trust: {candidate.memory.trust_level}")
        results.append(RetrievalResult(candidate.memory, relevance, recency, frequency, 0.0, score, score, reasons))
    return sorted(results, key=lambda result: (-result.stage_one_score, -result.relevance, result.memory.created_at, result.memory.id))

