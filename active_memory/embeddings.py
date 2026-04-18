"""Embedding provider helpers for production and development paths."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import Embedder, Embedding, HashEmbedder, LocalModelEmbedder, DEFAULT_LOCAL_MODEL


DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004"


class GeminiEmbedder:
    """Google Gemini embedding provider."""

    def __init__(
        self,
        model: str = DEFAULT_GEMINI_EMBEDDING_MODEL,
        api_key: str | None = None,
    ) -> None:
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        return self._dim or 768

    def embed(self, texts: list[str]) -> list[Embedding]:
        response = self._client.models.embed_content(
            model=self.model,
            contents=texts,
        )
        vectors: list[Embedding] = []
        for emb in response.embeddings:
            vec = np.array(emb.values, dtype=np.float32)
            if self._dim is None:
                self._dim = int(vec.shape[0])
            vectors.append(vec)
        return vectors


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
    local_model: str = DEFAULT_LOCAL_MODEL,
    verbose: bool = False,
    stream: Any = None,
) -> EmbedderSpec:
    """Create an embedder with a sensible production-facing default.

    ``auto`` prefers OpenAI embeddings when the package and API key are
    available, then tries a local sentence-transformer model, and
    finally falls back to the deterministic hash embedder.
    """
    stream = stream or sys.stderr
    chosen = provider.lower()

    # --- OpenAI (explicit or auto) ---
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
                    f"  [embeddings] OpenAI unavailable, trying local model: {exc}",
                    file=stream,
                )

    # --- Gemini embeddings (explicit or auto) ---
    if chosen in {"auto", "gemini"}:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        try:
            if chosen == "gemini" and not api_key:
                raise RuntimeError(
                    "GEMINI_API_KEY or GOOGLE_API_KEY not set"
                )
            if api_key:
                gemini_embedder = GeminiEmbedder(api_key=api_key)
                return EmbedderSpec(
                    provider="gemini",
                    embedder=gemini_embedder,
                    description=f"Google Gemini embeddings via {gemini_embedder.model}",
                    semantic=True,
                )
        except Exception as exc:
            if chosen == "gemini":
                raise RuntimeError(f"Failed to initialize Gemini embedder: {exc}") from exc
            if verbose:
                print(
                    f"  [embeddings] Gemini unavailable: {exc}",
                    file=stream,
                )

    # --- Local sentence-transformer model (explicit or auto) ---
    if chosen in {"auto", "local"}:
        try:
            embedder_local = LocalModelEmbedder(model_name=local_model)
            return EmbedderSpec(
                provider="local",
                embedder=embedder_local,
                description=f"Local sentence-transformer ({local_model})",
                semantic=True,
            )
        except Exception as exc:
            if chosen == "local":
                raise RuntimeError(
                    f"Failed to initialize local embedder: {exc}. "
                    "Install with: pip install sentence-transformers"
                ) from exc
            if verbose:
                print(
                    f"  [embeddings] Local model unavailable, falling back to hash: {exc}",
                    file=stream,
                )

    # --- Hash fallback (always available) ---
    return EmbedderSpec(
        provider="hash",
        embedder=HashEmbedder(dim=dim),
        description=f"Deterministic hash embedder (dim={dim}, non-semantic)",
        semantic=False,
    )
