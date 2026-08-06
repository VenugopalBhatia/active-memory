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
from .retriever import FinalWeights, RetrievalConfig, TwoPassRetriever
from .scoring import (
    ScoringPolicy,
    StageOneWeights,
    frequency_score,
    recency_score,
    relevance_score,
    score_stage_one,
)

__all__ = [
    "Candidate",
    "DeterministicTestEmbeddingProvider",
    "EmbeddingDimensionError",
    "EmbeddingProvider",
    "ExactCandidateRetriever",
    "FinalWeights",
    "LocalSentenceTransformerProvider",
    "OpenAIEmbeddingProvider",
    "RetrievalConfig",
    "ScoringPolicy",
    "StageOneWeights",
    "TwoPassRetriever",
    "create_embedding_provider",
    "embed_normalized",
    "frequency_score",
    "normalize_vector",
    "recency_score",
    "relevance_score",
    "score_stage_one",
]
