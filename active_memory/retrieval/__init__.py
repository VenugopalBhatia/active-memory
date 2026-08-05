"""Exact retrieval and inspectable ranking."""

from .candidates import Candidate, ExactCandidateRetriever
from .embeddings import (
    DeterministicTestEmbeddingProvider,
    EmbeddingDimensionError,
    EmbeddingProvider,
    LocalSentenceTransformerProvider,
    OpenAIEmbeddingProvider,
    create_embedding_provider,
    embed_normalized,
    normalize_vector,
)
from .scoring import (
    ScoringPolicy,
    StageOneWeights,
    frequency_score,
    recency_score,
    relevance_score,
    score_stage_one,
)

__all__ = [
    "Candidate", "ExactCandidateRetriever", "EmbeddingProvider",
    "DeterministicTestEmbeddingProvider", "LocalSentenceTransformerProvider",
    "OpenAIEmbeddingProvider", "EmbeddingDimensionError",
    "create_embedding_provider", "embed_normalized", "normalize_vector",
    "ScoringPolicy", "StageOneWeights", "relevance_score", "recency_score",
    "frequency_score", "score_stage_one",
]

