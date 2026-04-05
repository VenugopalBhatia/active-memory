"""Embedding provider helpers for production and development paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import Embedder, Embedding, HashEmbedder


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """OpenAI-backed embedder that satisfies the local Embedder protocol."""

    def __init__(
        self,
        model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
        client: Any | None = None,
    ) -> None:
        from openai import OpenAI

        self.model = model
        self._client = client or OpenAI()
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        # The protocol requires a dimension; we lazily discover it.
        return self._dim or 1536

    def embed(self, texts: list[str]) -> list[Embedding]:
        response = self._client.embeddings.create(input=texts, model=self.model)
        vectors: list[Embedding] = []
        for item in response.data:
            vec = np.array(item.embedding, dtype=np.float32)
            if self._dim is None:
                self._dim = int(vec.shape[0])
            vectors.append(vec)
        return vectors


@dataclass
class EmbedderSpec:
    provider: str
    embedder: Embedder
    description: str
    semantic: bool


def create_embedder(
    provider: str = "auto",
    *,
    dim: int = 64,
    openai_model: str = DEFAULT_OPENAI_EMBEDDING_MODEL,
    verbose: bool = False,
    stream: Any = None,
) -> EmbedderSpec:
    """Create an embedder with a sensible production-facing default.

    `auto` prefers OpenAI embeddings when the package and API key are
    available, otherwise falls back to the deterministic hash embedder.
    """
    stream = stream or sys.stderr
    chosen = provider.lower()

    if chosen in {"auto", "openai"}:
        api_key = os.environ.get("OPENAI_API_KEY")
        try:
            if chosen == "openai" and not api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            if api_key:
                embedder = OpenAIEmbedder(model=openai_model)
                return EmbedderSpec(
                    provider="openai",
                    embedder=embedder,
                    description=f"OpenAI embeddings via {openai_model}",
                    semantic=True,
                )
        except Exception as exc:
            if chosen == "openai":
                raise RuntimeError(f"Failed to initialize OpenAI embedder: {exc}") from exc
            if verbose:
                print(
                    f"  [embeddings] Falling back to hash embedder: {exc}",
                    file=stream,
                )

    return EmbedderSpec(
        provider="hash",
        embedder=HashEmbedder(dim=dim),
        description=f"Deterministic hash embedder (dim={dim}, non-semantic)",
        semantic=False,
    )
