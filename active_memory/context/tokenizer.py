"""Deterministic token counting used for local budgeting.

The approximation is deliberately conservative and provider-independent.
Callers may inject a model-specific implementation with the same interface.
"""

from __future__ import annotations

import math
from typing import Protocol


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    def __init__(self, characters_per_token: float = 3.5) -> None:
        if characters_per_token <= 0:
            raise ValueError("characters_per_token must be positive")
        self.characters_per_token = characters_per_token

    def count(self, text: str) -> int:
        return 0 if not text else max(1, math.ceil(len(text) / self.characters_per_token))

