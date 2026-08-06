from __future__ import annotations

from active_memory.ingestion.relationships import build_relationships
from active_memory.models import Memory, RetrievalResult, utc_now
from active_memory.retrieval.affinity import affinity_scores


def memory(memory_id: str, **metadata) -> Memory:
    now = utc_now()
    return Memory(memory_id, "project", "s1", "msg", "code_change", memory_id, [1.0, 0.0], now, now, 1, "tool_observed", metadata=metadata)


def result(item: Memory, score: float) -> RetrievalResult:
    return RetrievalResult(item, score, 1.0, 0.0, 0.0, score, score, [])


def test_relationships_are_deterministic_and_metadata_based() -> None:
    old = memory("old", file_path="auth.py", function="verify", entities=["token"])
    new = memory("new", file_path="auth.py", function="verify", entities=["token"])
    edges = build_relationships(new, [old], utc_now())
    assert [edge.edge_type for edge in edges] == ["same_file", "same_function", "shared_entity"]
    assert {edge.edge_type: edge.weight for edge in edges}["same_function"] == 0.85


def test_affinity_is_zero_without_seed_link_and_uses_edge_weight() -> None:
    seed = result(memory("seed"), 0.8)
    linked = result(memory("linked"), 0.4)
    scores, _ = affinity_scores([seed, linked], [seed], [])
    assert scores[linked.memory.id] == 0.0

    edge = build_relationships(linked.memory, [seed.memory], utc_now())
    assert edge == []  # embedding/content similarity never creates affinity
    linked.memory.metadata["file_path"] = "auth.py"
    seed.memory.metadata["file_path"] = "auth.py"
    edge = build_relationships(linked.memory, [seed.memory], utc_now())
    scores, reasons = affinity_scores([seed, linked], [seed], edge)
    assert scores["linked"] == 0.75 * 0.8
    assert "same_file" in reasons["linked"]

