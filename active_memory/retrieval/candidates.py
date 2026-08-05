"""Exact cosine candidate retrieval over normalized memory embeddings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from active_memory.models import Memory, MemoryFilters
from active_memory.retrieval.embeddings import EmbeddingDimensionError, EmbeddingProvider, embed_normalized
from active_memory.storage.base import MemoryStore


@dataclass(frozen=True, slots=True)
class Candidate:
    memory: Memory
    cosine_similarity: float


class ExactCandidateRetriever:
    def __init__(self, store: MemoryStore, embedding_provider: EmbeddingProvider) -> None:
        self.store = store
        self.embedding_provider = embedding_provider

    def retrieve(self, query: str, namespace: str, *, candidate_limit: int = 80, minimum_cosine: float = -1.0, filters: MemoryFilters | None = None) -> list[Candidate]:
        if candidate_limit <= 0:
            return []
        memories = self.store.get_active_memories(namespace, filters)
        if not memories:
            return []
        dimension = self.embedding_provider.dimension
        wrong = [memory.id for memory in memories if len(memory.embedding) != dimension]
        if wrong:
            raise EmbeddingDimensionError(f"stored memories have the wrong embedding dimension: {', '.join(wrong[:5])}")

        query_vector = np.asarray(embed_normalized(self.embedding_provider, [query])[0], dtype=np.float32)
        matrix = np.asarray([memory.embedding for memory in memories], dtype=np.float32)
        similarities = matrix @ query_vector
        limit = min(candidate_limit, len(memories))
        if limit == len(memories):
            indices = np.arange(len(memories))
        else:
            indices = np.argpartition(similarities, -limit)[-limit:]
        candidates = [
            Candidate(memories[int(index)], float(np.clip(similarities[int(index)], -1.0, 1.0)))
            for index in indices
            if similarities[int(index)] >= minimum_cosine
        ]
        return sorted(candidates, key=lambda item: (-item.cosine_similarity, item.memory.created_at, item.memory.id))

