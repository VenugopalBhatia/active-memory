from __future__ import annotations

from active_memory.context import ApproximateTokenCounter
from active_memory.ingestion.writer import MemoryIngestor
from active_memory.retrieval.embeddings import DeterministicTestEmbeddingProvider
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def test_secret_redaction_precedes_persistence_and_memory_extraction(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    ingestor = MemoryIngestor(store, DeterministicTestEmbeddingProvider(), ApproximateTokenCounter())
    message, memories = ingestor.ingest("s1", "project", "user", "api_key=sk-supersecretvalue and we use PostgreSQL")
    assert message is not None
    persisted = store.get_message(message.id)
    assert "supersecret" not in persisted.content
    assert "[REDACTED]" in persisted.content
    assert memories == []
    store.close()


def test_assistant_speculation_is_not_promoted_to_memory(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    ingestor = MemoryIngestor(store, DeterministicTestEmbeddingProvider(), ApproximateTokenCounter())
    message, memories = ingestor.ingest("s1", "project", "assistant", "It may be a database issue causing the auth failure.")
    assert message is not None
    assert memories == []
    assert store.get_recent_messages("s1", 10)[0].role == "assistant"
    store.close()

