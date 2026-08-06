"""Core types for the active memory system."""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


# -- Embedding abstraction --------------------------------------------------

Embedding = NDArray[np.float32]  # 1-D float32 vector


class Embedder(Protocol):
    """Produces embeddings for text chunks. Swap in any provider."""

    @property
    def dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        ...

    def embed(self, texts: list[str]) -> list[Embedding]:
        """Return one embedding per input text."""
        ...


class HashEmbedder:
    """Deterministic pseudo-embedder for testing. NOT semantic."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big") % 2**31
            rng = np.random.RandomState(seed)
            vec = rng.randn(self._dim).astype(np.float32)
            vec /= np.linalg.norm(vec) + 1e-9
            out.append(vec)
        return out


DEFAULT_LOCAL_MODEL = "all-MiniLM-L6-v2"


class LocalModelEmbedder:
    """Semantic embedder using a local sentence-transformer model.

    Provides real semantic embeddings without requiring an API key.
    Uses ``sentence-transformers`` under the hood (lazy-imported so
    the dependency is only required when this embedder is selected).
    """

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        # encode returns an ndarray of shape (len(texts), dim)
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return [np.asarray(vec, dtype=np.float32) for vec in embeddings]


# -- KV Tuple ---------------------------------------------------------------

@dataclass
class KVTuple:
    """
    Atomic unit of memory: a semantic key paired with content.

    key_text  : natural-language label ("user prefers dark mode")
    value_text: the actual content chunk
    key_emb   : embedding of key_text (set by the tree on insert)
    """

    key_text: str
    value_text: str
    key_emb: Embedding | None = field(default=None, repr=False)

    # --- bookkeeping ---
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    hit_count: int = 0
    token_cost: int = 0  # approximate token count of value_text

    # --- structural references (from call graph analysis) ---
    # IDs of other tuples that this tuple references (calls, imports)
    references: list[str] = field(default_factory=list)
    # IDs of other tuples that reference this tuple
    referenced_by: list[str] = field(default_factory=list)
    # Metadata tags for filtering (e.g. "code", "conversation", "doc")
    tags: list[str] = field(default_factory=list)

    def touch(self) -> None:
        """Record an access."""
        self.last_accessed = time.time()
        self.hit_count += 1


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


def cosine_sim(a: Embedding, b: Embedding) -> float:
    """Cosine similarity between two vectors."""
    dot = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    if norm < 1e-9:
        return 0.0
    return dot / norm
