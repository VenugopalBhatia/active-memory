from __future__ import annotations

from active_memory.context import BudgetConfig
from active_memory.proxy import MemoryEngine
from active_memory.retrieval import DeterministicTestEmbeddingProvider, RetrievalConfig
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def test_cross_session_update_supersedes_old_fact_in_context(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(
        store, DeterministicTestEmbeddingProvider(256),
        retrieval_config=RetrievalConfig(candidate_limit=20, seed_limit=4, result_limit=10, minimum_relevance=0.0),
        budget_config=BudgetConfig(model_context_limit=1200, reserved_response_tokens=100, safety_margin_tokens=50, recent_message_limit=4),
    )
    engine.transform_anthropic_request({"model": "test", "messages": [{"role": "user", "content": "We are using MongoDB."}]}, session_id="s1", namespace="project")
    engine.transform_anthropic_request({"model": "test", "messages": [{"role": "user", "content": "We have migrated to PostgreSQL."}]}, session_id="s2", namespace="project")
    transformed, _ = engine.transform_anthropic_request({"model": "test", "tools": [{"name": "lookup"}], "messages": [{"role": "user", "content": "Which database does the project use now?"}]}, session_id="s3", namespace="project")

    active = store.get_active_memories("project")
    assert any("PostgreSQL" in memory.content for memory in active)
    assert not any("MongoDB" in memory.content for memory in active)
    assert "PostgreSQL" in transformed["system"]
    assert "MongoDB" not in transformed["system"]
    assert transformed["tools"] == [{"name": "lookup"}]
    assert transformed["messages"][-1]["content"] == "Which database does the project use now?"
    store.close()


def test_confirmed_tool_cause_outranks_skipped_assistant_speculation(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    engine = MemoryEngine(store, DeterministicTestEmbeddingProvider(256), retrieval_config=RetrievalConfig(minimum_relevance=0.0), budget_config=BudgetConfig(model_context_limit=1200, reserved_response_tokens=100, safety_margin_tokens=50))
    request = {
        "messages": [
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Tests failed in auth/test_tokens.py."}]},
            {"role": "assistant", "content": "It may be a database issue."},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "The token fixture expired."}]},
            {"role": "user", "content": "Yes, the expired fixture was the cause."},
        ]
    }
    engine.transform_anthropic_request(request, session_id="s1", namespace="project")
    transformed, _ = engine.transform_anthropic_request({"messages": [{"role": "user", "content": "Why did the authentication tests fail?"}]}, session_id="s2", namespace="project")
    assert "expired fixture" in transformed["system"].lower()
    assert "database issue" not in transformed["system"].lower()
    assert "tool_observed" in transformed["system"] or "user_confirmed" in transformed["system"]
    store.close()

