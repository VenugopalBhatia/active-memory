from __future__ import annotations

from active_memory.context import ApproximateTokenCounter, BudgetConfig, ContextAssembler
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, RetrievalResult, new_id, utc_now
from active_memory.retrieval.embeddings import DeterministicTestEmbeddingProvider, embed_normalized
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def setup_memory(store, text: str, score: float, *, group: str | None = None) -> RetrievalResult:
    counter = ApproximateTokenCounter(4.0)
    writer = MemoryWriter(store, counter)
    message = writer.make_message("s1", "user", text)
    provider = DeterministicTestEmbeddingProvider(32)
    item = Memory(
        new_id("mem"), "project", "s1", message.id, "decision", text,
        embed_normalized(provider, [text])[0], message.created_at, message.created_at,
        message.token_count, "user_confirmed", valid_from=message.created_at,
        metadata={"dependency_group": group} if group else {},
    )
    writer.write(message, [item])
    return RetrievalResult(item, score, 1.0, 0.0, 0.0, score, score, ["test"])


def test_pinned_latest_and_recent_fit_hard_budget(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    config = BudgetConfig(model_context_limit=700, reserved_response_tokens=50, safety_margin_tokens=30, recent_turn_fraction=0.25, memory_fraction=0.35, recent_message_limit=2)
    assembler = ContextAssembler(store, ApproximateTokenCounter(4.0), config)
    memories = [setup_memory(store, "Use PostgreSQL for persistent storage.", 0.95)]
    messages = [
        {"role": "user", "content": "Earlier question"},
        {"role": "assistant", "content": "Earlier answer"},
        {"role": "user", "content": "What database do we use now?"},
    ]
    result = assembler.assemble(session_id="s1", namespace="project", system="Be precise.", tools=[{"name": "lookup"}], messages=messages, ranked_memories=memories, assembled_at=utc_now())
    assert result.messages[-1] == messages[-1]
    assert messages[-2] in result.messages
    assert result.input_tokens <= result.available_input_tokens
    assert result.input_tokens + config.reserved_response_tokens + config.safety_margin_tokens <= config.model_context_limit
    assert "trust=\"user_confirmed\"" in result.memory_context
    store.close()


def test_utility_packing_excludes_low_value_and_updates_only_included(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    high = setup_memory(store, "PostgreSQL is current.", 0.99)
    low = setup_memory(store, "Old low value detail " * 30, 0.05)
    config = BudgetConfig(model_context_limit=420, reserved_response_tokens=40, safety_margin_tokens=20, recent_turn_fraction=0.1, memory_fraction=0.55, recent_message_limit=0)
    assembler = ContextAssembler(store, ApproximateTokenCounter(4.0), config)
    result = assembler.assemble(session_id="s1", namespace="project", system="", tools=None, messages=[{"role": "user", "content": "database?"}], ranked_memories=[low, high], assembled_at=utc_now())

    assert [item.memory.id for item in result.included] == [high.memory.id]
    stored_high, stored_low = store.get_memories_by_ids([high.memory.id, low.memory.id])
    assert stored_high.inclusion_count == 1
    assert stored_low.inclusion_count == 0
    assert result.excluded[low.memory.id] == "memory budget exhausted"
    store.close()


def test_recent_content_deduplicates_memory_and_dependency_groups_are_atomic(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    duplicate = setup_memory(store, "The token fixture expired.", 1.0)
    first = setup_memory(store, "First dependency.", 0.8, group="auth")
    second = setup_memory(store, "Second dependency. " * 20, 0.8, group="auth")
    config = BudgetConfig(model_context_limit=280, reserved_response_tokens=40, safety_margin_tokens=20, recent_turn_fraction=0.3, memory_fraction=0.35, recent_message_limit=2)
    assembler = ContextAssembler(store, ApproximateTokenCounter(4.0), config)
    messages = [
        {"role": "assistant", "content": "The token fixture expired."},
        {"role": "user", "content": "Why did auth fail?"},
    ]
    result = assembler.assemble(session_id="s1", namespace="project", system="", tools=None, messages=messages, ranked_memories=[duplicate, first, second], assembled_at=utc_now())
    assert result.excluded[duplicate.memory.id] == "already represented in recent turns"
    included_ids = {item.memory.id for item in result.included}
    assert included_ids.isdisjoint({first.memory.id, second.memory.id}) or {first.memory.id, second.memory.id}.issubset(included_ids)
    store.close()
