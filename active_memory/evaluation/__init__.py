"""Offline evaluation APIs with lazy imports for module execution."""

from __future__ import annotations

from typing import Any


def run_all() -> dict[str, object]:
    from .end_to_end import run_all as implementation

    return implementation()


def run_retrieval_benchmark(k: int = 3) -> dict[str, dict[str, float]]:
    from .end_to_end import run_retrieval_benchmark as implementation

    return implementation(k)


def run_latency_benchmark(*args: Any, **kwargs: Any) -> list[dict[str, float | int]]:
    from .end_to_end import run_latency_benchmark as implementation

    return implementation(*args, **kwargs)


__all__ = ["run_all", "run_latency_benchmark", "run_retrieval_benchmark"]
