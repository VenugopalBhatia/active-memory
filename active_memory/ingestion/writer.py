"""Transactional raw-event and derived-memory writer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Sequence

from active_memory.models import Memory, MemoryEdge, Message, content_hash, new_id, utc_now
from active_memory.storage.sqlite_store import SQLiteMemoryStore


class MemoryWriter:
    def __init__(self, store: SQLiteMemoryStore, token_counter: Any) -> None:
        self.store = store
        self.token_counter = token_counter

    def make_message(self, session_id: str, role: str, content: str, *, created_at: datetime | None = None, metadata: dict[str, Any] | None = None, message_id: str | None = None) -> Message:
        return Message(
            id=message_id or new_id("msg"), session_id=session_id, role=role, content=content,
            created_at=created_at or utc_now(), token_count=self.token_counter.count(content),
            content_hash=content_hash(content), metadata=metadata or {},
        )

    def write_raw(self, session_id: str, role: str, content: str, **kwargs: Any) -> Message:
        message = self.make_message(session_id, role, content, **kwargs)
        self.store.add_message(message)
        return message

    def write(self, message: Message, memories: Sequence[Memory], edges: Sequence[MemoryEdge] = ()) -> None:
        if any(memory.source_message_id != message.id for memory in memories):
            raise ValueError("all derived memories must reference the raw message")
        self.store.add_message_and_memories(message, memories, edges)

