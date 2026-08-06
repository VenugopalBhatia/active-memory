"""Small serializable retrieval trace model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(slots=True)
class RetrievalTrace:
    query_hash: str
    namespace: str
    candidate_count: int
    seed_ids: list[str] = field(default_factory=list)
    selected_ids: list[str] = field(default_factory=list)
    excluded: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

