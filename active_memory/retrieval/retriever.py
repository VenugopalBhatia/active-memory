"""Two-pass retrieval with post-seed relationship expansion."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from active_memory.models import MemoryFilters, RetrievalResult
from active_memory.retrieval.affinity import affinity_scores
from active_memory.retrieval.candidates import Candidate, ExactCandidateRetriever
from active_memory.retrieval.embeddings import EmbeddingProvider, embed_normalized
from active_memory.retrieval.scoring import ScoringPolicy, score_stage_one
from active_memory.storage.base import MemoryStore


@dataclass(frozen=True, slots=True)
class FinalWeights:
    relevance: float = 0.55
    recency: float = 0.20
    frequency: float = 0.10
    affinity: float = 0.15

    def __post_init__(self) -> None:
        values = (self.relevance, self.recency, self.frequency, self.affinity)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("final weights must be within [0, 1]")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
            raise ValueError("final weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    candidate_limit: int = 80
    seed_limit: int = 6
    neighbor_limit: int = 30
    result_limit: int = 20
    minimum_relevance: float = 0.20
    scoring: ScoringPolicy = field(default_factory=ScoringPolicy)
    final_weights: FinalWeights = field(default_factory=FinalWeights)

    def __post_init__(self) -> None:
        if min(self.candidate_limit, self.seed_limit, self.neighbor_limit, self.result_limit) < 0:
            raise ValueError("retrieval limits cannot be negative")
        if not 0.0 <= self.minimum_relevance <= 1.0:
            raise ValueError("minimum_relevance must be within [0, 1]")


class TwoPassRetriever:
    def __init__(self, store: MemoryStore, embedding_provider: EmbeddingProvider, config: RetrievalConfig | None = None) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.config = config or RetrievalConfig()
        self.candidates = ExactCandidateRetriever(store, embedding_provider)

    def retrieve(self, query: str, namespace: str, now: datetime, filters: MemoryFilters | None = None) -> list[RetrievalResult]:
        initial = self.candidates.retrieve(query, namespace, candidate_limit=self.config.candidate_limit, filters=filters)
        stage_one = [result for result in score_stage_one(initial, now, self.config.scoring) if result.relevance >= self.config.minimum_relevance]
        seeds = stage_one[: self.config.seed_limit]
        if not seeds:
            return []
        edges = self.store.get_neighbors([seed.memory.id for seed in seeds])
        neighbor_ids: list[str] = []
        initial_ids = {result.memory.id for result in stage_one}
        for edge in edges:
            for memory_id in (edge.source_id, edge.target_id):
                if memory_id not in initial_ids and memory_id not in neighbor_ids:
                    neighbor_ids.append(memory_id)
        neighbor_memories = [memory for memory in self.store.get_memories_by_ids(neighbor_ids[: self.config.neighbor_limit]) if memory.namespace == namespace and memory.status == "active"]
        if neighbor_memories:
            query_vector = np.asarray(embed_normalized(self.embedding_provider, [query])[0], dtype=np.float32)
            neighbor_candidates = [Candidate(memory, float(np.clip(np.dot(memory.embedding, query_vector), -1.0, 1.0))) for memory in neighbor_memories]
            combined = initial + [candidate for candidate in neighbor_candidates if candidate.memory.id not in {item.memory.id for item in initial}]
            stage_one = score_stage_one(combined, now, self.config.scoring)

        affinities, affinity_reasons = affinity_scores(stage_one, seeds, edges)
        weights = self.config.final_weights
        for result in stage_one:
            result.affinity = affinities.get(result.memory.id, 0.0)
            result.final_score = min(1.0, max(0.0,
                weights.relevance * result.relevance + weights.recency * result.recency
                + weights.frequency * result.frequency + weights.affinity * result.affinity
            ))
            reason = affinity_reasons.get(result.memory.id)
            if reason:
                result.retrieval_reason.append(reason)
        ranked = sorted(stage_one, key=lambda result: (-result.final_score, -result.stage_one_score, result.memory.created_at, result.memory.id))
        return ranked[: self.config.result_limit]

