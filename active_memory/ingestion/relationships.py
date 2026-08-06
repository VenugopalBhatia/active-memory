"""Deterministic relationship construction from explicit memory metadata."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

from active_memory.models import Memory, MemoryEdge

DEFAULT_EDGE_WEIGHTS: Mapping[str, float] = {
    "resolves": 1.00,
    "supersedes": 1.00,
    "depends_on": 0.90,
    "same_function": 0.85,
    "same_file": 0.75,
    "reply_to": 0.70,
    "same_task": 0.65,
    "shared_entity": 0.40,
    "adjacent_turn": 0.25,
}


def build_relationships(new_memory: Memory, existing: Sequence[Memory], created_at: datetime, weights: Mapping[str, float] = DEFAULT_EDGE_WEIGHTS) -> list[MemoryEdge]:
    edges: dict[tuple[str, str, str], MemoryEdge] = {}

    def add(target: str, edge_type: str, metadata: dict | None = None) -> None:
        key = (new_memory.id, target, edge_type)
        edges[key] = MemoryEdge(new_memory.id, target, edge_type, weights[edge_type], created_at, metadata or {})

    explicit = {
        "resolves_memory_id": "resolves",
        "supersedes_memory_id": "supersedes",
        "depends_on_memory_id": "depends_on",
        "reply_to_memory_id": "reply_to",
        "previous_memory_id": "adjacent_turn",
    }
    known_ids = {memory.id for memory in existing}
    for field, edge_type in explicit.items():
        target = new_memory.metadata.get(field)
        if isinstance(target, str) and target in known_ids:
            add(target, edge_type, {"source": field})

    for other in existing:
        if new_memory.metadata.get("file_path") and new_memory.metadata.get("file_path") == other.metadata.get("file_path"):
            add(other.id, "same_file", {"file_path": new_memory.metadata["file_path"]})
        if new_memory.metadata.get("function") and new_memory.metadata.get("function") == other.metadata.get("function"):
            add(other.id, "same_function", {"function": new_memory.metadata["function"]})
        if new_memory.metadata.get("task_id") and new_memory.metadata.get("task_id") == other.metadata.get("task_id"):
            add(other.id, "same_task", {"task_id": new_memory.metadata["task_id"]})
        shared = sorted(set(new_memory.metadata.get("entities", [])) & set(other.metadata.get("entities", [])))
        if shared:
            add(other.id, "shared_entity", {"entities": shared})
    return sorted(edges.values(), key=lambda edge: (edge.target_id, edge.edge_type))

