"""Embedding provider abstraction and normalized provider adapters."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np


class EmbeddingError(RuntimeError):
    pass


class EmbeddingDimensionError(EmbeddingError):
    pass


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


def normalize_vector(vector: Sequence[float], dimension: int, epsilon: float = 1e-12) -> list[float]:
    array = np.asarray(vector, dtype=np.float32)
    if array.ndim != 1 or len(array) != dimension:
        raise EmbeddingDimensionError(f"expected dimension {dimension}, received shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise EmbeddingError("embedding contains non-finite values")
    norm = float(np.linalg.norm(array))
    if norm <= epsilon:
        raise EmbeddingError("zero-length embedding cannot be normalized")
    return (array / norm).astype(np.float32).tolist()


def embed_normalized(provider: EmbeddingProvider, texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = provider.embed_texts(texts)
    if len(vectors) != len(texts):
        raise EmbeddingError(f"provider returned {len(vectors)} vectors for {len(texts)} texts")
    return [normalize_vector(vector, provider.dimension) for vector in vectors]


class DeterministicTestEmbeddingProvider:
    """Stable lexical test provider. It is explicitly not semantic."""

    semantic = False

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = np.zeros(self.dimension, dtype=np.float32)
            words = [word for word in text.lower().replace("_", " ").split() if word]
            for word in words or [text]:
                digest = hashlib.sha256(word.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                sign = 1.0 if digest[4] & 1 else -1.0
                vector[index] += sign
            if not np.any(vector):
                vector[0] = 1.0
            vectors.append(vector.tolist())
        return vectors


class LocalSentenceTransformerProvider:
    semantic = True

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self._client = SentenceTransformer(model)
        self._dimension = int(self._client.get_sentence_embedding_dimension())

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._client.encode(list(texts), normalize_embeddings=False)
        return np.asarray(vectors, dtype=np.float32).tolist()


class OpenAIEmbeddingProvider:
    semantic = True

    _KNOWN_DIMENSIONS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small", *, client: Any | None = None, dimension: int | None = None) -> None:
        if client is None:
            from openai import OpenAI

            client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self._client = client
        self._dimension = dimension or self._KNOWN_DIMENSIONS.get(model)
        if self._dimension is None:
            raise ValueError("dimension is required for unknown OpenAI embedding models")

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in response.data]


def create_embedding_provider(provider: str = "local", *, model: str | None = None, dimension: int = 64, client: Any | None = None) -> EmbeddingProvider:
    chosen = provider.lower()
    if chosen == "local":
        return LocalSentenceTransformerProvider(model or "sentence-transformers/all-MiniLM-L6-v2")
    if chosen == "openai":
        return OpenAIEmbeddingProvider(model or "text-embedding-3-small", client=client)
    if chosen in {"test", "lexical", "hash"}:
        return DeterministicTestEmbeddingProvider(dimension)
    raise ValueError(f"unknown embedding provider: {provider}")

