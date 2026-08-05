from __future__ import annotations

import sqlite3
from datetime import timedelta

import pytest

from active_memory.context.tokenizer import ApproximateTokenCounter
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, MemoryFilters, new_id, utc_now
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def make_memory(message, namespace: str = "project") -> Memory:
    return Memory(
        id=new_id("mem"), namespace=namespace, session_id=message.session_id,
        source_message_id=message.id, memory_type="decision", content=message.content,
        embedding=[1.0, 0.0], created_at=message.created_at,
        updated_at=message.created_at, token_count=message.token_count,
        trust_level="user_confirmed", valid_from=message.created_at,
        metadata={"source_role": message.role},
    )


def test_messages_and_memories_survive_restart(tmp_path) -> None:
    path = tmp_path / "memory.db"
    store = SQLiteMemoryStore(path)
    writer = MemoryWriter(store, ApproximateTokenCounter())
    message = writer.make_message("s1", "user", "Use PostgreSQL.")
    memory = make_memory(message)
    writer.write(message, [memory])
    store.close()

    reopened = SQLiteMemoryStore(path)
    assert reopened.get_recent_messages("s1", 10) == [message]
    assert reopened.get_memories_by_ids([memory.id])[0].content == memory.content
    reopened.close()


def test_message_and_memory_transaction_rolls_back(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter(store, ApproximateTokenCounter())
    message = writer.make_message("s1", "user", "A transactional decision.")
    memory = make_memory(message)
    memory.source_message_id = "not-the-message"

    with pytest.raises(ValueError):
        writer.write(message, [memory])
    assert store.get_recent_messages("s1", 10) == []
    store.close()


def test_duplicate_hashes_are_separate_immutable_events(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter(store, ApproximateTokenCounter())
    first = writer.write_raw("s1", "user", "Repeated content")
    second = writer.write_raw("s1", "user", "Repeated content")

    assert first.id != second.id
    assert first.content_hash == second.content_hash
    assert len(store.get_recent_messages("s1", 10)) == 2
    with pytest.raises(sqlite3.IntegrityError):
        store.add_message(first)
    store.close()


def test_namespace_isolation_and_supersession(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    writer = MemoryWriter(store, ApproximateTokenCounter())
    old_message = writer.make_message("s1", "user", "Use MongoDB.")
    old = make_memory(old_message, "alpha")
    writer.write(old_message, [old])
    other_message = writer.make_message("s2", "user", "Keep SQLite.")
    other = make_memory(other_message, "beta")
    writer.write(other_message, [other])
    new_message = writer.make_message("s3", "user", "We migrated to PostgreSQL.")
    new = make_memory(new_message, "alpha")
    writer.write(new_message, [new])

    when = utc_now() + timedelta(seconds=1)
    store.supersede_memory(old.id, new.id, when)

    assert [m.id for m in store.get_active_memories("alpha")] == [new.id]
    historical = store.get_active_memories("alpha", MemoryFilters(statuses=frozenset({"active", "superseded"})))
    by_id = {memory.id: memory for memory in historical}
    assert by_id[old.id].valid_until == when
    assert by_id[old.id].superseded_by == new.id
    assert [m.id for m in store.get_active_memories("beta")] == [other.id]
    assert store.get_neighbors([new.id])[0].edge_type == "supersedes"
    store.close()
