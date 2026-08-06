"""Exact and conservative semantic duplicate detection."""

from __future__ import annotations

import numpy as np

from active_memory.models import Memory


def find_duplicate(content: str, embedding: list[float], memory_type: str, metadata: dict, existing: list[Memory], semantic_threshold: float = 0.97) -> Memory | None:
    normalized = " ".join(content.casefold().split())
    scope = (metadata.get("file_path"), tuple(sorted(metadata.get("entities", []))))
    for memory in existing:
        if " ".join(memory.content.casefold().split()) == normalized:
            return memory
    for memory in existing:
        other_scope = (memory.metadata.get("file_path"), tuple(sorted(memory.metadata.get("entities", []))))
        if memory.memory_type != memory_type or scope != other_scope or len(memory.embedding) != len(embedding):
            continue
        if float(np.dot(memory.embedding, embedding)) >= semantic_threshold:
            return memory
    return None

