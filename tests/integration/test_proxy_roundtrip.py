from __future__ import annotations

from active_memory.context import BudgetConfig
from active_memory.proxy import MemoryEngine, RequestState
from active_memory.retrieval import DeterministicTestEmbeddingProvider
from active_memory.storage.sqlite_store import SQLiteMemoryStore


class FailingProvider:
    dimension = 8

    def embed_texts(self, texts):
        raise RuntimeError("embedding offline")


def test_memory_failure_persists_raw_event_and_returns_original_shape(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store, FailingProvider(), budget_config=BudgetConfig(model_context_limit=500, reserved_response_tokens=50, safety_margin_tokens=20))
    original = {"model": "claude-test", "max_tokens": 50, "system": [{"type": "text", "text": "System"}], "messages": [{"role": "user", "content": "Decision: we use PostgreSQL."}]}
    transformed, _ = engine.transform_anthropic_request(original, session_id="s1", namespace="project")
    assert transformed == original
    messages = store.get_recent_messages("s1", 10)
    assert any(message.role == "user" and "PostgreSQL" in message.content for message in messages)
    assert engine.last_trace["status"] == "fallback"
    store.close()


def test_assistant_response_is_persisted_with_content_blocks(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store, DeterministicTestEmbeddingProvider(), budget_config=BudgetConfig(model_context_limit=500, reserved_response_tokens=50, safety_margin_tokens=20))
    state = RequestState("s1", "project", 1)
    engine.ingest_anthropic_response({"id": "msg_upstream", "content": [{"type": "text", "text": "Decision: we use SQLite."}, {"type": "tool_use", "id": "t1", "name": "check", "input": {"path": "db.py"}}]}, state)
    messages = store.get_recent_messages("s1", 10)
    assert len(messages) == 1
    assert "Decision: we use SQLite." in messages[0].content
    assert "tool_use check" in messages[0].content
    store.close()
