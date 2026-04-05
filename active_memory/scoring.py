"""Scoring engine for KV tuples and nodes.

Produces a single scalar score blending:
  - recency   : exponential decay from last access
  - frequency  : log-scaled hit count
  - relevance  : cosine similarity to current query (optional)
  - affinity   : structural relationship to other queried tuples
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .types import Embedding, KVTuple, cosine_sim


@dataclass
class ScoringConfig:
    """Tunable weights and decay constants."""

    recency_weight: float = 0.20
    frequency_weight: float = 0.20
    relevance_weight: float = 0.45
    affinity_weight: float = 0.15   # structural relationship bonus

    # Half-life for recency decay (seconds). After this many seconds
    # since last access, the recency component drops to 0.5.
    recency_half_life: float = 1800.0  # 30 minutes

    # Floor score: nothing drops below this so even cold tuples can
    # be found by a highly relevant query.
    floor: float = 0.05


class Scorer:
    """Scores KV tuples for eviction and context assembly."""

    def __init__(self, config: ScoringConfig | None = None) -> None:
        self.cfg = config or ScoringConfig()
        self._lambda = math.log(2) / self.cfg.recency_half_life
        # Set of tuple IDs that are in the current "active set" — used
        # to compute structural affinity (set by the assembler before scoring)
        self._active_ids: set[str] = set()

    def set_active_context(self, tuple_ids: set[str]) -> None:
        """Set the IDs of tuples currently in the active context.
        Used to compute structural affinity — tuples that reference
        or are referenced by active tuples get a boost."""
        self._active_ids = tuple_ids

    def score(
        self,
        t: KVTuple,
        query_emb: Embedding | None = None,
        now: float | None = None,
    ) -> float:
        """Compute composite score for a single tuple."""
        now = now or time.time()

        # -- recency --
        age = max(0.0, now - t.last_accessed)
        recency = math.exp(-self._lambda * age)

        # -- frequency (log-scaled, +1 to handle 0 hits) --
        frequency = math.log1p(t.hit_count) / math.log1p(100)  # normalise
        frequency = min(frequency, 1.0)

        # -- relevance --
        if query_emb is not None and t.key_emb is not None:
            relevance = max(0.0, cosine_sim(query_emb, t.key_emb))
        else:
            relevance = 0.5  # neutral when no query

        # -- structural affinity --
        affinity = self._compute_affinity(t)

        raw = (
            self.cfg.recency_weight * recency
            + self.cfg.frequency_weight * frequency
            + self.cfg.relevance_weight * relevance
            + self.cfg.affinity_weight * affinity
        )
        return max(raw, self.cfg.floor)

    def _compute_affinity(self, t: KVTuple) -> float:
        """Compute structural affinity score.

        A tuple scores high on affinity if it references (or is
        referenced by) tuples that are in the current active context.
        This catches the hidden dependency problem: function A calls
        helper B, so when A is queried, B gets a boost even if its
        name is semantically unrelated.
        """
        if not self._active_ids:
            return 0.0

        # Count how many of this tuple's references are active
        ref_overlap = 0
        total_refs = len(t.references) + len(t.referenced_by)

        if total_refs == 0:
            return 0.0

        for ref_id in t.references:
            if ref_id in self._active_ids:
                ref_overlap += 1
        for ref_id in t.referenced_by:
            if ref_id in self._active_ids:
                ref_overlap += 1

        # Normalise: what fraction of this tuple's structural
        # relationships are currently active?
        return min(1.0, ref_overlap / max(1, min(total_refs, 5)))
