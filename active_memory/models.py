"""Typed records shared by persistence, retrieval, and context assembly."""

from __future__ import annotations

import hashlib
import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


MESSAGE_ROLES = {"user", "assistant", "system", "tool"}
MEMORY_TYPES = {
    "decision", "preference", "fact", "task", "tool_observation",
    "code_change", "error", "resolution", "episode", "summary",
}
TRUST_LEVELS = {
    "user_confirmed", "tool_observed", "assistant_generated",
    "external_untrusted",
}
MEMORY_STATUSES = {"active", "superseded", "deleted", "quarantined"}
EDGE_TYPES = {
    "resolves", "supersedes", "depends_on", "same_task", "same_file",
    "same_function", "reply_to", "shared_entity", "adjacent_turn",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _aware(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")


@dataclass(frozen=True, slots=True)
class Message:
    """Immutable event from a source conversation."""

    id: str
    session_id: str
    role: str
    content: str
    created_at: datetime
    token_count: int
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.session_id:
            raise ValueError("message id and session_id are required")
        if self.role not in MESSAGE_ROLES:
            raise ValueError(f"unsupported message role: {self.role}")
        if self.token_count < 0:
            raise ValueError("token_count cannot be negative")
        if self.content_hash != content_hash(self.content):
            raise ValueError("content_hash does not match content")
        _aware(self.created_at, "created_at")


@dataclass(slots=True)
class Memory:
    """Derived, retrievable record with provenance and temporal state."""

    id: str
    namespace: str
    session_id: str | None
    source_message_id: str
    memory_type: str
    content: str
    embedding: list[float]
    created_at: datetime
    updated_at: datetime
    token_count: int
    trust_level: str
    status: str = "active"
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    superseded_by: str | None = None
    inclusion_count: int = 0
    last_included_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.id or not self.namespace or not self.source_message_id:
            raise ValueError("memory id, namespace, and source_message_id are required")
        if self.memory_type not in MEMORY_TYPES:
            raise ValueError(f"unsupported memory type: {self.memory_type}")
        if self.trust_level not in TRUST_LEVELS:
            raise ValueError(f"unsupported trust level: {self.trust_level}")
        if self.status not in MEMORY_STATUSES:
            raise ValueError(f"unsupported memory status: {self.status}")
        if self.token_count < 0 or self.inclusion_count < 0:
            raise ValueError("token and inclusion counts cannot be negative")
        if not all(math.isfinite(float(value)) for value in self.embedding):
            raise ValueError("embedding values must be finite")
        for name in ("created_at", "updated_at", "valid_from", "valid_until", "last_included_at"):
            value = getattr(self, name)
            if value is not None:
                _aware(value, name)


@dataclass(frozen=True, slots=True)
class MemoryEdge:
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_id == self.target_id:
            raise ValueError("self-referential memory edges are not allowed")
        if self.edge_type not in EDGE_TYPES:
            raise ValueError(f"unsupported edge type: {self.edge_type}")
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError("edge weight must be within [0, 1]")
        _aware(self.created_at, "created_at")


@dataclass(frozen=True, slots=True)
class MemoryFilters:
    session_id: str | None = None
    memory_types: frozenset[str] | None = None
    trust_levels: frozenset[str] | None = None
    statuses: frozenset[str] = frozenset({"active"})
    valid_at: datetime | None = None
    source_role: str | None = None
    file_path: str | None = None
    entity: str | None = None


@dataclass(slots=True)
class RetrievalResult:
    memory: Memory
    relevance: float
    recency: float
    frequency: float
    affinity: float
    stage_one_score: float
    final_score: float
    retrieval_reason: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for name in ("relevance", "recency", "frequency", "affinity", "stage_one_score", "final_score"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")

