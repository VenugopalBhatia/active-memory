from __future__ import annotations

from active_memory.context.tokenizer import ApproximateTokenCounter
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, MemoryEdge, new_id, utc_now
from active_memory.retrieval.embeddings import DeterministicTestEmbeddingProvider, embed_normalized
from active_memory.retrieval.retriever import RetrievalConfig, TwoPassRetriever
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def test_linked_neighbor_expands_after_seed_selection(tmp_path) -> None:
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    provider = DeterministicTestEmbeddingProvider(512)
    writer = MemoryWriter(store, ApproximateTokenCounter())

    def add(text: str, metadata: dict) -> Memory:
        message = writer.make_message("s1", "user", text)
        item = Memory(new_id("mem"), "project", "s1", message.id, "code_change", text,
                      embed_normalized(provider, [text])[0], message.created_at, message.created_at,
                      message.token_count, "user_confirmed", metadata=metadata)
        writer.write(message, [item])
        return item

    seed = add("authentication token validation", {"file_path": "auth.py"})
    neighbor = add("helper implementation details", {"file_path": "auth.py"})
    unrelated = add("authentication documentation", {"file_path": "docs.md"})
    store.add_edges([MemoryEdge(seed.id, neighbor.id, "same_file", 0.75, utc_now())])

    config = RetrievalConfig(candidate_limit=1, seed_limit=1, neighbor_limit=5, result_limit=5, minimum_relevance=0.0)
    results = TwoPassRetriever(store, provider, config).retrieve("authentication token", "project", utc_now())
    by_id = {result.memory.id: result for result in results}
    assert seed.id in by_id
    assert neighbor.id in by_id
    assert by_id[neighbor.id].affinity > 0.0
    assert "linked to seed" in " ".join(by_id[neighbor.id].retrieval_reason)
    assert unrelated.id not in by_id
    store.close()
