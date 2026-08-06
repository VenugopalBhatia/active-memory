from __future__ import annotations

from datetime import timedelta

import pytest

from active_memory.models import Memory, utc_now
from active_memory.retrieval.candidates import Candidate
from active_memory.retrieval.scoring import (
    ScoringPolicy,
    StageOneWeights,
    frequency_score,
    recency_score,
    relevance_score,
    score_stage_one,
)


def memory(memory_id: str, *, age_days: int = 0, inclusions: int = 0, memory_type: str = "fact") -> Memory:
    created = utc_now() - timedelta(days=age_days)
    return Memory(memory_id, "project", "s1", "msg", memory_type, memory_id, [1.0, 0.0], created, created, 1, "user_confirmed", inclusion_count=inclusions)


def test_relevance_is_bounded_and_monotonic() -> None:
    assert relevance_score(-2.0) == 0.0
    assert relevance_score(-1.0) == 0.0
    assert relevance_score(0.0) == 0.5
    assert relevance_score(1.0) == 1.0
    assert relevance_score(2.0) == 1.0


def test_older_memory_has_lower_recency_and_stable_types_decay_slower() -> None:
    now = utc_now()
    fresh = memory("fresh", age_days=1)
    old_fact = memory("old", age_days=100)
    old_preference = memory("preference", age_days=100, memory_type="preference")
    assert recency_score(fresh, now) > recency_score(old_fact, now)
    assert recency_score(old_preference, now) > recency_score(old_fact, now)


def test_frequency_uses_only_inclusion_count_and_is_bounded() -> None:
    assert frequency_score(0, 100) == 0.0
    assert 0.0 < frequency_score(3, 100) < frequency_score(50, 100) < 1.0
    assert frequency_score(100, 100) == 1.0


def test_stage_one_weights_are_configurable_and_scores_bounded() -> None:
    now = utc_now()
    relevant_old = Candidate(memory("relevant", age_days=100), 1.0)
    recent_irrelevant = Candidate(memory("recent"), -1.0)
    relevance_policy = ScoringPolicy(stage_one=StageOneWeights(1.0, 0.0, 0.0))
    recency_policy = ScoringPolicy(stage_one=StageOneWeights(0.0, 1.0, 0.0))

    assert score_stage_one([relevant_old, recent_irrelevant], now, relevance_policy)[0].memory.id == "relevant"
    assert score_stage_one([relevant_old, recent_irrelevant], now, recency_policy)[0].memory.id == "recent"
    assert all(0.0 <= result.stage_one_score <= 1.0 for result in score_stage_one([relevant_old, recent_irrelevant], now))


def test_invalid_weights_are_rejected() -> None:
    with pytest.raises(ValueError, match="sum"):
        StageOneWeights(0.5, 0.5, 0.5)
