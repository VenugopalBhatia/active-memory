"""Transactional raw-event and derived-memory writer."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from active_memory.ingestion.classification import classify_segment
from active_memory.ingestion.deduplication import find_duplicate
from active_memory.ingestion.redaction import SecretRedactor
from active_memory.ingestion.relationships import build_relationships
from active_memory.ingestion.segmentation import segment_message
from active_memory.models import Memory, MemoryEdge, Message, content_hash, new_id, utc_now
from active_memory.retrieval.embeddings import EmbeddingProvider, embed_normalized
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

    def write(self, message: Message, memories: Sequence[Memory], edges: Sequence[MemoryEdge] = (), supersessions: Sequence[tuple[str, str, datetime]] = ()) -> None:
        if any(memory.source_message_id != message.id for memory in memories):
            raise ValueError("all derived memories must reference the raw message")
        self.store.add_message_and_memories(message, memories, edges, supersessions)


class MemoryIngestor:
    """Explicit redaction, segmentation, classification, dedupe, and write pipeline."""

    def __init__(self, store: SQLiteMemoryStore, embedding_provider: EmbeddingProvider, token_counter: Any, *, minimum_segment_tokens: int = 12, maximum_segment_tokens: int = 500, semantic_duplicate_threshold: float = 0.97, storage_enabled: bool = True, store_assistant_generated: bool = True, redactor: SecretRedactor | None = None) -> None:
        self.store = store
        self.embedding_provider = embedding_provider
        self.token_counter = token_counter
        self.writer = MemoryWriter(store, token_counter)
        self.minimum_segment_tokens = minimum_segment_tokens
        self.maximum_segment_tokens = maximum_segment_tokens
        self.semantic_duplicate_threshold = semantic_duplicate_threshold
        self.storage_enabled = storage_enabled
        self.store_assistant_generated = store_assistant_generated
        self.redactor = redactor or SecretRedactor()

    def ingest(self, session_id: str, namespace: str, role: str, content: str, *, message_id: str | None = None, created_at: datetime | None = None, metadata: dict[str, Any] | None = None) -> tuple[Message | None, list[Memory]]:
        if not self.storage_enabled:
            return None, []
        if message_id and self.store.get_message(message_id):
            return self.store.get_message(message_id), []
        redacted, redaction_count = self.redactor.redact(content)
        event_metadata = dict(metadata or {})
        event_metadata.update({"redaction_count": redaction_count, "ingestion_status": "pending"})
        message = self.writer.make_message(session_id, role, redacted, created_at=created_at, metadata=event_metadata, message_id=message_id)
        segments = segment_message(redacted, self.token_counter, minimum_tokens=self.minimum_segment_tokens, maximum_tokens=self.maximum_segment_tokens)
        classified = [(segment, classify_segment(role, segment.content, {**segment.metadata, "source_role": role, "start_offset": segment.start_offset, "end_offset": segment.end_offset})) for segment in segments]
        eligible = [
            (segment, classification) for segment, classification in classified
            if classification is not None
            and (self.store_assistant_generated or classification.trust_level != "assistant_generated")
        ]
        try:
            embeddings = embed_normalized(self.embedding_provider, [segment.content for segment, _ in eligible])
        except Exception:
            message.metadata["ingestion_status"] = "embedding_failed"
            self.store.add_message(message)
            raise

        existing = self.store.get_active_memories(namespace)
        memories: list[Memory] = []
        edges: list[MemoryEdge] = []
        supersessions: list[tuple[str, str, datetime]] = []
        timestamp = message.created_at
        for (segment, classification), embedding in zip(eligible, embeddings):
            assert classification is not None
            duplicate = find_duplicate(segment.content, embedding, classification.memory_type, classification.metadata, existing + memories, self.semantic_duplicate_threshold)
            if duplicate:
                continue
            memory = Memory(
                new_id("mem"), namespace, session_id, message.id, classification.memory_type,
                segment.content, embedding, timestamp, timestamp, self.token_counter.count(segment.content),
                classification.trust_level, valid_from=timestamp, metadata=dict(classification.metadata),
            )
            conflict_key = memory.metadata.get("conflict_key")
            if conflict_key:
                conflicting = [item for item in existing if item.metadata.get("conflict_key") == conflict_key and item.content.casefold() != memory.content.casefold()]
                if conflicting:
                    old = max(conflicting, key=lambda item: (item.created_at, item.id))
                    memory.metadata["supersedes_memory_id"] = old.id
                    supersessions.append((old.id, memory.id, timestamp))
            memory_edges = build_relationships(memory, existing + memories, timestamp)
            edges.extend(memory_edges)
            memories.append(memory)
        message.metadata["ingestion_status"] = "complete"
        self.writer.write(message, memories, edges, supersessions)
        return message, memories
