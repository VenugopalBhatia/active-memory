from __future__ import annotations

from active_memory.context.tokenizer import ApproximateTokenCounter
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, MemoryFilters, new_id
from active_memory.retrieval.candidates import ExactCandidateRetriever
from active_memory.retrieval.embeddings import DeterministicTestEmbeddingProvider, EmbeddingDimensionError, embed_normalized
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def add_memory(store, provider, *, namespace="project", session="s1", text="postgres database", memory_type="decision", trust="user_confirmed", metadata=None):
    writer = MemoryWriter(store, ApproximateTokenCounter())
    message = writer.make_message(session, "user", text)
    memory = Memory(
        id=new_id("mem"), namespace=namespace, session_id=session, source_message_id=message.id,
        memory_type=memory_type, content=text, embedding=embed_normalized(provider, [text])[0],
        created_at=message.created_at, updated_at=message.created_at, token_count=message.token_count,
        trust_level=trust, valid_from=message.created_at, metadata=metadata or {"source_role": "user"},
    )
    writer.write(message, [memory])
    return memory


def test_embeddings_are_normalized_and_exact_search_ranks_match(tmp_path) -> None:
    provider = DeterministicTestEmbeddingProvider(256)
    vector = embed_normalized(provider, ["postgres database"])[0]
    assert abs(sum(value * value for value in vector) - 1.0) < 1e-6
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    expected = add_memory(store, provider, text="postgres database decision")
    add_memory(store, provider, text="redis cache policy")

    results = ExactCandidateRetriever(store, provider).retrieve("postgres database", "project", candidate_limit=2)
    assert results[0].memory.id == expected.id
    assert results[0].cosine_similarity > results[1].cosine_similarity
    store.close()


def test_candidate_filters_apply_before_search(tmp_path) -> None:
    provider = DeterministicTestEmbeddingProvider(128)
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    add_memory(store, provider, namespace="other", text="postgres database")
    expected = add_memory(store, provider, session="s2", text="postgres database", memory_type="fact", metadata={"source_role": "user", "file_path": "db.py", "entities": ["postgres"]})
    add_memory(store, provider, session="s1", text="postgres database", memory_type="decision")

    filters = MemoryFilters(session_id="s2", memory_types=frozenset({"fact"}), file_path="db.py", entity="postgres")
    results = ExactCandidateRetriever(store, provider).retrieve("postgres database", "project", filters=filters)
    assert [result.memory.id for result in results] == [expected.id]
    store.close()


def test_retrieval_is_read_only_and_rejects_dimension_mismatch(tmp_path) -> None:
    provider = DeterministicTestEmbeddingProvider(32)
    store = SQLiteMemoryStore(tmp_path / "memory.db")
    memory = add_memory(store, provider, text="authentication token")
    retriever = ExactCandidateRetriever(store, provider)
    retriever.retrieve("authentication", "project")
    assert store.get_memories_by_ids([memory.id])[0].inclusion_count == 0

    wrong_provider = DeterministicTestEmbeddingProvider(16)
    try:
        ExactCandidateRetriever(store, wrong_provider).retrieve("authentication", "project")
    except EmbeddingDimensionError:
        pass
    else:
        raise AssertionError("dimension mismatch was not rejected")
    store.close()
