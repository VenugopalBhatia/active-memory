"""Behavior-focused tests for the active memory system.

These tests are intentionally more opinionated than the original smoke
tests. They validate retrieval behavior, anchoring, dependency pull,
context resets, and embedding-provider selection.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from active_memory import (
    AnthropicModelClient,
    AssemblerConfig,
    BTreeConfig,
    ContextAssembler,
    GeneratedResponse,
    KVTuple,
    OpenAIModelClient,
    Scorer,
    ScoringConfig,
    SemanticBTree,
    cosine_sim,
    create_model_client,
    create_embedder,
    estimate_tokens,
)
from active_memory.cli import ActiveMemoryCLI
from active_memory.eval import (
    FrozenScenario,
    FrozenTurn,
    Probe,
    ActiveMemoryAdapter,
    FullContextAdapter,
    SlidingWindowAdapter,
    run_prompt_benchmark,
)
from active_memory.grounding import GroundedAssembler, GroundingConfig
from active_memory.middleware import ActiveMemoryMiddleware, MiddlewareConfig
from active_memory.mcp_server import ActiveMemoryMCPServer
from active_memory.long_context_bench import build_long_context_scenario
from active_memory.prepare_release import prepare
from active_memory.proxy import ContextManager, ProxyConfig, ProxyHandler
from active_memory.types import HashEmbedder


class SemanticTestEmbedder:
    """Small deterministic embedder with stable topic groupings for tests."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self._topics = {
            0: {"database", "db", "postgres", "clickhouse", "catalog", "sql"},
            1: {"cache", "caching", "redis", "session", "ttl"},
            2: {"auth", "pkce", "token", "refresh", "oauth"},
            3: {"deploy", "docker", "release", "build"},
            4: {"caller", "callee", "helper", "dependency"},
            5: {"budget", "cost", "finance"},
        }

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            lower = text.lower().replace("-", " ")
            for idx, keywords in self._topics.items():
                if any(word in lower for word in keywords):
                    vec[idx] += 2.0
            # Stable fallback signal so unrelated strings are still distinct.
            for ch in lower:
                vec[(ord(ch) % (self._dim - 1)) + 1] += 0.05
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec[0] = 1.0
                norm = 1.0
            out.append((vec / norm).astype(np.float32))
        return out


def make_tree(
    *,
    max_tuples: int = 4,
    threshold: float = 0.1,
) -> tuple[SemanticBTree, SemanticTestEmbedder, Scorer]:
    embedder = SemanticTestEmbedder()
    scorer = Scorer(ScoringConfig(recency_half_life=60.0))
    tree = SemanticBTree(
        embedder=embedder,
        scorer=scorer,
        config=BTreeConfig(max_tuples=max_tuples, compress_threshold=threshold),
    )
    return tree, embedder, scorer


class TestCoreUtilities(unittest.TestCase):
    def test_touch_updates_access_metadata(self) -> None:
        t = KVTuple(key_text="test", value_text="hello")
        before = t.last_accessed
        time.sleep(0.001)
        t.touch()
        self.assertEqual(t.hit_count, 1)
        self.assertGreaterEqual(t.last_accessed, before)

    def test_estimate_tokens_scales_with_text_size(self) -> None:
        self.assertEqual(estimate_tokens("a" * 400), 100)
        self.assertGreater(estimate_tokens("a" * 800), estimate_tokens("a" * 400))

    def test_cosine_similarity_handles_identity_and_orthogonality(self) -> None:
        a = np.array([1, 0, 0], dtype=np.float32)
        b = np.array([0, 1, 0], dtype=np.float32)
        self.assertAlmostEqual(cosine_sim(a, a), 1.0, places=5)
        self.assertAlmostEqual(cosine_sim(a, b), 0.0, places=5)

    def test_hash_embedder_is_stable_across_processes(self) -> None:
        script = (
            "from active_memory.types import HashEmbedder; "
            "vec = HashEmbedder(dim=8).embed(['database choice'])[0]; "
            "print(','.join(f'{x:.6f}' for x in vec))"
        )
        common = {
            "cwd": "/Users/venugopalbhatia/Documents/active_memory_v0.1",
            "text": True,
        }
        first = subprocess.check_output([sys.executable, "-c", script], **common).strip()
        second = subprocess.check_output([sys.executable, "-c", script], **common).strip()
        self.assertEqual(first, second)


class TestScorerBehavior(unittest.TestCase):
    def test_recency_and_frequency_increase_score(self) -> None:
        embedder = SemanticTestEmbedder()
        scorer = Scorer(ScoringConfig(recency_half_life=60.0))
        emb = embedder.embed(["database"])[0]

        fresh = KVTuple(key_text="database", value_text="postgres", key_emb=emb)
        fresh.touch()
        fresh.touch()

        stale = KVTuple(key_text="database", value_text="postgres", key_emb=emb)
        stale.last_accessed = time.time() - 600

        self.assertGreater(scorer.score(fresh, emb), scorer.score(stale, emb))

    def test_structural_affinity_boosts_related_tuple(self) -> None:
        embedder = SemanticTestEmbedder()
        scorer = Scorer()
        active = KVTuple(key_text="caller", value_text="calls helper", key_emb=embedder.embed(["caller"])[0])
        related = KVTuple(key_text="helper", value_text="shared dependency", key_emb=embedder.embed(["helper"])[0])
        unrelated = KVTuple(key_text="budget", value_text="cost cap", key_emb=embedder.embed(["budget"])[0])

        active.references.append(related.id)
        related.referenced_by.append(active.id)
        scorer.set_active_context({active.id})

        query = embedder.embed(["helper dependency"])[0]
        self.assertGreater(scorer.score(related, query), scorer.score(unrelated, query))


class TestSemanticBTreeBehavior(unittest.TestCase):
    def test_query_prefers_semantically_matching_tuple(self) -> None:
        tree, embedder, _ = make_tree()
        tree.insert("database choice", "Use PostgreSQL 16 for metadata.")
        tree.insert("caching policy", "Use Redis with short TTLs.")
        tree.insert("auth flow", "Use PKCE and refresh token rotation.")

        query = embedder.embed(["database postgres decision"])[0]
        results = tree.query(query, top_k=2)

        self.assertEqual(results[0][1].key_text, "database choice")

    def test_prune_evicts_cold_tuples_but_keeps_hot_tuple(self) -> None:
        tree, embedder, _ = make_tree(threshold=0.2)
        cold_1 = tree.insert("database old note", "legacy mysql decision")
        cold_2 = tree.insert("auth old note", "legacy auth setting")
        hot = tree.insert("database current choice", "postgres for metadata")

        for t in (cold_1, cold_2):
            t.last_accessed = time.time() - 100_000
            t.hit_count = 0
        hot.touch()
        hot.touch()

        query = embedder.embed(["database postgres"])[0]
        evicted = tree.prune(query)

        self.assertGreaterEqual(len(evicted), 1)
        self.assertIn(hot.id, {t.id for t in tree.all_tuples()})

    def test_compress_cold_subtrees_replaces_raw_tuples_with_summary(self) -> None:
        tree, embedder, _ = make_tree(max_tuples=3, threshold=0.35)
        original_ids: set[str] = set()
        for i in range(8):
            t = tree.insert(f"old database note {i}", f"cold database fact {i}")
            t.last_accessed = time.time() - 100_000
            original_ids.add(t.id)

        compressed = tree.compress_cold_subtrees(
            summariser=lambda tuples: "summary of cold database facts",
            query_emb=embedder.embed(["fresh auth topic"])[0],
        )

        all_tuples = tree.all_tuples()
        self.assertGreaterEqual(compressed, 1)
        self.assertTrue(any(t.key_text.startswith("summary:") for t in all_tuples))
        self.assertNotEqual(original_ids, {t.id for t in all_tuples})

    def test_insert_after_compression_replaces_summary_with_live_tuple(self) -> None:
        tree, embedder, _ = make_tree(max_tuples=2, threshold=0.9)
        for i in range(5):
            t = tree.insert(f"old note {i}", f"cold fact {i}")
            t.last_accessed = time.time() - 100_000

        tree.compress_cold_subtrees(
            query_emb=embedder.embed(["fresh topic"])[0],
        )
        tree.insert("new note", "fresh fact after compression")

        all_keys = {t.key_text for t in tree.all_tuples()}
        self.assertIn("new note", all_keys)
        self.assertTrue(any("new note" in {t.key_text for t in cluster} for cluster in tree.tuples_by_cluster()))


class TestAssemblerBehavior(unittest.TestCase):
    def test_anchor_includes_relevant_low_priority_tuple(self) -> None:
        tree, embedder, _ = make_tree()
        anchored = tree.insert("database policy", "PostgreSQL 16 is the catalog DB.")
        anchored.last_accessed = time.time() - 100_000
        anchored.hit_count = 0
        tree.insert("cache policy", "Redis caches sessions.")
        tree.insert("auth policy", "PKCE with rotated refresh tokens.")

        assembler = ContextAssembler(
            tree=tree,
            config=AssemblerConfig(
                total_budget=400,
                pinned_reserve=50,
                recency_window=1,
                managed_top_k=0,
                anchor_relevance_threshold=0.7,
            ),
        )
        conversation = [{"role": "user", "content": "Which database are we using?"}]
        result = assembler.assemble(conversation, embedder.embed(["postgres database"])[0])

        self.assertTrue(any(block.source == "anchored" for block in result.managed_blocks))
        self.assertIn("database policy", {block.key for block in result.managed_blocks})

    def test_dependency_pull_includes_linked_tuple(self) -> None:
        tree, embedder, _ = make_tree()
        caller = tree.insert("caller function", "The caller delegates to helper.")
        callee = tree.insert("helper dependency", "The helper validates tokens.")
        caller.references.append(callee.id)
        callee.referenced_by.append(caller.id)

        assembler = ContextAssembler(
            tree=tree,
            config=AssemblerConfig(
                total_budget=400,
                pinned_reserve=50,
                recency_window=1,
                managed_top_k=1,
                dependency_pull=True,
                dependency_budget_fraction=0.5,
                anchor_relevance_threshold=1.1,
            ),
        )
        conversation = [{"role": "user", "content": "Show me the caller flow"}]
        result = assembler.assemble(conversation, embedder.embed(["caller flow"])[0])

        self.assertIn("caller function", {block.key for block in result.managed_blocks})
        self.assertIn("helper dependency", {block.key for block in result.managed_blocks})
        self.assertTrue(any(block.source == "dependency" for block in result.managed_blocks))

    def test_to_messages_places_retrieved_context_before_latest_user_message(self) -> None:
        tree, embedder, _ = make_tree()
        for i in range(6):
            tree.insert(f"database fact {i}", f"Database detail {i} about postgres.")

        assembler = ContextAssembler(
            tree=tree,
            config=AssemblerConfig(total_budget=600, pinned_reserve=100, recency_window=2),
        )
        conversation = [
            {"role": "user", "content": "Earlier we discussed the database."},
            {"role": "assistant", "content": "Yes, we narrowed it down."},
            {"role": "user", "content": "Remind me of the decision."},
        ]
        assembled = assembler.assemble(conversation, embedder.embed(["database decision postgres"])[0])
        messages = assembler.to_messages("System prompt", assembled)

        self.assertEqual(messages[-1]["content"], "Remind me of the decision.")
        self.assertTrue(any("<retrieved_context>" in msg.get("content", "") for msg in messages))


class TestContextManagerAndProxyBehavior(unittest.TestCase):
    def test_context_manager_passthroughs_small_conversation_but_indexes_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                ProxyConfig(
                    token_budget=1_000,
                    embedder_provider="hash",
                    state_dir=tmp,
                )
            )
            messages = [{"role": "user", "content": "Short message about database choice."}]

            result = manager.process_messages(messages)

            self.assertEqual(result, messages)
            self.assertGreater(manager.tree.size, 0)

    def test_context_manager_builds_reset_briefing_for_large_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                ProxyConfig(
                    token_budget=120,
                    reset_threshold=0.5,
                    reset_briefing_budget=60,
                    reset_recency_turns=1,
                    embedder_provider="hash",
                    state_dir=tmp,
                )
            )
            messages = [
                {"role": "user", "content": "Database decision: use PostgreSQL for metadata. " * 4},
                {"role": "assistant", "content": "Acknowledged. PostgreSQL is selected. " * 4},
                {"role": "user", "content": "Caching decision: use Redis with short TTLs. " * 4},
                {"role": "assistant", "content": "Acknowledged. Redis is selected. " * 4},
            ]

            fresh = manager.process_messages(messages)

            self.assertGreaterEqual(len(fresh), 3)
            self.assertIn("[CONTEXT RELOAD:", fresh[0]["content"])
            self.assertEqual(fresh[-1]["content"], messages[-1]["content"])

    def test_context_manager_dedupes_across_restart_using_saved_message_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ProxyConfig(
                token_budget=1_000,
                embedder_provider="hash",
                state_dir=tmp,
            )
            messages = [
                {"role": "user", "content": "Database decision: use PostgreSQL for metadata."},
                {"role": "assistant", "content": "Acknowledged. PostgreSQL is selected."},
            ]

            manager = ContextManager(config)
            manager.process_messages(messages)
            initial_tree_size = manager.tree.size
            manager._save_state()

            reloaded = ContextManager(config)
            reloaded.process_messages(messages)

            self.assertEqual(reloaded.tree.size, initial_tree_size)

    def test_context_manager_persists_passthrough_turn_count_and_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = ProxyConfig(
                token_budget=1_000,
                embedder_provider="hash",
                state_dir=tmp,
            )
            manager = ContextManager(config)
            messages = [{"role": "user", "content": "Short message about database choice."}]

            manager.process_messages(messages)

            reloaded = ContextManager(config)
            self.assertEqual(reloaded._turn_count, 1)
            self.assertGreater(reloaded.tree.size, 0)

    def test_context_manager_reembeds_saved_state_when_embedding_dim_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = ProxyConfig(
                token_budget=1_000,
                embedder_provider="hash",
                embed_dim=64,
                state_dir=tmp,
            )
            second = ProxyConfig(
                token_budget=1_000,
                embedder_provider="hash",
                embed_dim=24,
                state_dir=tmp,
            )
            messages = [{"role": "user", "content": "Short message about database choice."}]

            manager = ContextManager(first)
            manager.process_messages(messages)
            manager._save_state()

            reloaded = ContextManager(second)

            self.assertGreater(reloaded.tree.size, 0)
            for node in reloaded.tree.all_nodes():
                if node.centroid is not None:
                    self.assertEqual(int(node.centroid.shape[0]), 24)
            for tuple_ in reloaded.tree.all_tuples():
                self.assertIsNotNone(tuple_.key_emb)
                self.assertEqual(int(tuple_.key_emb.shape[0]), 24)

    def test_context_manager_ingests_repeated_identical_content_as_new_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                ProxyConfig(
                    token_budget=1_000,
                    embedder_provider="hash",
                    state_dir=tmp,
                )
            )
            first_turn = [{"role": "user", "content": "Repeat this exact note about the database choice."}]
            second_turn = first_turn + [
                {"role": "assistant", "content": "Noted."},
                {"role": "user", "content": "Repeat this exact note about the database choice."},
            ]

            manager.process_messages(first_turn)
            tree_size_after_first = manager.tree.size
            manager.process_messages(second_turn)

            self.assertGreater(manager.tree.size, tree_size_after_first)

    def test_context_manager_ingests_assistant_response_immediately_without_reingesting_next_turn(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manager = ContextManager(
                ProxyConfig(
                    token_budget=1_000,
                    embedder_provider="hash",
                    state_dir=tmp,
                )
            )
            user_one = [{"role": "user", "content": "User note " * 8}]
            assistant_response = {
                "content": [{"type": "text", "text": "Assistant decision " * 5}]
            }
            next_turn = user_one + [
                {"role": "assistant", "content": "Assistant decision " * 5},
                {"role": "user", "content": "Follow-up question " * 4},
            ]

            manager.process_messages(user_one)
            self.assertEqual(manager.tree.size, 1)

            manager.ingest_assistant_response(assistant_response)
            self.assertEqual(manager.tree.size, 2)

            manager.process_messages(next_turn)
            self.assertEqual(manager.tree.size, 3)

    def test_proxy_handler_matches_messages_route_with_query_string(self) -> None:
        handler = ProxyHandler.__new__(ProxyHandler)
        handler.path = "/v1/messages?beta=true"
        handler.headers = {"Content-Length": "2"}
        handler.rfile = io.BytesIO(b"{}")

        with patch.object(handler, "_handle_messages") as handle_messages, patch.object(
            handler, "_handle_config_update"
        ) as handle_config_update, patch.object(handler, "_proxy_passthrough") as proxy_passthrough:
            handler.do_POST()

        handle_messages.assert_called_once_with(b"{}")
        handle_config_update.assert_not_called()
        proxy_passthrough.assert_not_called()


class TestAssemblerAndGroundingBehavior(unittest.TestCase):
    def test_assembler_counts_text_blocks_in_pinned_messages(self) -> None:
        tree, embedder, _ = make_tree()
        assembler = ContextAssembler(
            tree=tree,
            config=AssemblerConfig(total_budget=500, pinned_reserve=100, recency_window=1),
        )
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "x" * 200}],
            },
            {"role": "assistant", "content": "short reply"},
        ]

        assembled = assembler.assemble(conversation, embedder.embed(["topic"])[0])

        self.assertGreaterEqual(assembled.total_tokens, 50)

    def test_grounded_assembler_uses_configured_thresholds(self) -> None:
        tree = SemanticBTree(
            embedder=HashEmbedder(dim=8),
            scorer=Scorer(),
            config=BTreeConfig(max_tuples=8),
        )
        config = GroundingConfig(grounding_threshold=0.99, contradiction_threshold=0.99)
        grounded = GroundedAssembler(
            tree=tree,
            embedder=HashEmbedder(dim=8),
            scorer=Scorer(),
            config=config,
        )

        self.assertEqual(grounded.verifier.grounding_threshold, 0.99)
        self.assertEqual(grounded.verifier.contradiction_threshold, 0.99)


class TestEmbeddingProviderSelection(unittest.TestCase):
    def test_create_embedder_auto_falls_back_to_hash_without_openai_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            spec = create_embedder("auto", dim=24)

        self.assertEqual(spec.provider, "hash")
        self.assertFalse(spec.semantic)
        self.assertIn("non-semantic", spec.description)

    def test_create_embedder_openai_requires_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                create_embedder("openai")

    def test_create_embedder_auto_reports_fallback_reason(self) -> None:
        stream = io.StringIO()
        with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
            with patch("active_memory.embeddings.OpenAIEmbedder", side_effect=RuntimeError("boom")):
                spec = create_embedder("auto", verbose=True, stream=stream)

        self.assertEqual(spec.provider, "hash")
        self.assertIn("Falling back to hash embedder", stream.getvalue())


class _FakeAnthropicResponse:
    def __init__(self) -> None:
        self.content = [type("Block", (), {"text": "anthropic ok"})()]
        self.usage = type("Usage", (), {"input_tokens": 12, "output_tokens": 5})()


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = type(
            "Messages",
            (),
            {"create": lambda _self, **kwargs: _FakeAnthropicResponse()},
        )()


class _FakeOpenAIResponse:
    def __init__(self) -> None:
        message = type("Message", (), {"content": "openai ok"})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = type("Usage", (), {"prompt_tokens": 21, "completion_tokens": 8})()


class _FakeOpenAIClient:
    def __init__(self) -> None:
        chat = type(
            "Chat",
            (),
            {"completions": type("Completions", (), {"create": lambda _self, **kwargs: _FakeOpenAIResponse()})()},
        )()
        self.chat = chat


class _FakeModelClient:
    def generate(self, **kwargs):
        return GeneratedResponse(text="ok", raw={"ok": True}, usage={})


class _RecordingModelClient:
    def __init__(self, responses: list[str]) -> None:
        self.calls: list[dict] = []
        self._responses = list(responses)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self._responses.pop(0)
        return GeneratedResponse(text=text, raw={"text": text})


class _FakeGrounder:
    def __init__(self) -> None:
        self.verified: list[str] = []

    def build_grounded_prompt(self, system_prompt, conversation, query_emb, token_budget, recency_window):
        return system_prompt, list(conversation)

    def verify_response(self, response_text: str):
        self.verified.append(response_text)
        grounded = "corrected" in response_text
        return type(
            "Verification",
            (),
            {
                "response_text": response_text,
                "overall_grounding": 1.0 if grounded else 0.0,
                "contradictions": [],
                "ungrounded_claims": [],
            },
        )()

    def build_correction_prompt(self, verification):
        return "Please correct the answer." if verification.overall_grounding < 0.5 else None


class TestCliAndMcpPersistence(unittest.TestCase):
    def test_cli_load_replaces_existing_tree_and_preserves_tuple_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            with patch("active_memory.cli.SESSION_DIR", session_dir):
                cli = ActiveMemoryCLI(
                    client=_FakeModelClient(),
                    embedder=HashEmbedder(dim=8),
                    session_name="demo",
                )
                cli.session_path = session_dir / "demo.json"
                cli.mw._conversation = [{"role": "user", "content": "saved turn"}]
                cli.mw._turn_count = 1
                saved_tuple = cli.mw.tree.insert("code:helper", "helper body")
                saved_tuple.references.append("callee-id")
                saved_tuple.referenced_by.append("caller-id")
                saved_tuple.tags.append("code")
                cli._cmd_save()

                restored = ActiveMemoryCLI(
                    client=_FakeModelClient(),
                    embedder=HashEmbedder(dim=8),
                    session_name="demo",
                )
                restored.session_path = session_dir / "demo.json"
                restored.mw.tree.insert("extra", "should disappear")

                restored._cmd_load()

                self.assertEqual(restored.mw.tree.size, 1)
                loaded_tuple = restored.mw.tree.all_tuples()[0]
                self.assertEqual(loaded_tuple.references, ["callee-id"])
                self.assertEqual(loaded_tuple.referenced_by, ["caller-id"])
                self.assertEqual(loaded_tuple.tags, ["code"])
                self.assertEqual(restored.mw._turn_count, 1)

    def test_mcp_server_honors_configured_state_dir_and_preserves_tuple_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            config_path = tmp_path / "config.json"
            config_path.write_text(json.dumps({
                "embedder": "hash",
                "embed_dim": 23,
                "state_dir": str(tmp_path / "state"),
            }))

            server = ActiveMemoryMCPServer(config_path=str(config_path))
            stored = server.tree.insert("module:auth", "auth flow")
            stored.references.append("token-id")
            stored.referenced_by.append("login-id")
            stored.tags.append("code")
            server._save_state()

            reloaded = ActiveMemoryMCPServer(config_path=str(config_path))

            self.assertEqual(reloaded.embedder.dim, 23)
            self.assertEqual(reloaded._state_path, tmp_path / "state" / "mcp_state.json")
            self.assertEqual(reloaded.tree.size, 1)
            loaded = reloaded.tree.all_tuples()[0]
            self.assertEqual(loaded.references, ["token-id"])
            self.assertEqual(loaded.referenced_by, ["login-id"])
            self.assertEqual(loaded.tags, ["code"])


class TestUtilityScripts(unittest.TestCase):
    def test_prepare_release_uses_repo_root_and_readme_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = prepare(output_dir=tmp, repo_slug="owner/repo")
            out_path = Path(out)

            self.assertTrue((out_path / "README.md").exists())
            self.assertTrue((out_path / "LICENSE").exists())
            self.assertTrue((out_path / "install.sh").exists())

    def test_long_context_bench_plants_first_fact(self) -> None:
        scenario = build_long_context_scenario(2_000, filler_repetitions=2)

        self.assertTrue(any(p.fact_id == "database" and p.planted_turn == 1 for p in scenario.probes))


class TestMiddlewareBehavior(unittest.TestCase):
    def test_auto_correct_preserves_api_kwargs_and_recomputes_verification(self) -> None:
        cfg = MiddlewareConfig()
        cfg.grounding.enabled = True
        cfg.grounding.provenance_injection = True
        cfg.grounding.post_verification = True
        cfg.grounding.auto_correct = True
        client = _RecordingModelClient(["initial answer", "corrected answer"])
        mw = ActiveMemoryMiddleware(client=client, embedder=HashEmbedder(dim=8), config=cfg)
        mw.grounder = _FakeGrounder()
        for i in range(6):
            mw.tree.insert(f"seed:{i}", f"seed fact {i}")

        response = mw.send("What did we decide?", temperature=0.2)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["temperature"], 0.2)
        self.assertEqual(client.calls[1]["temperature"], 0.2)
        self.assertEqual(mw.grounder.verified, ["initial answer", "corrected answer"])
        self.assertEqual(response.text, "corrected answer")
        self.assertEqual(response.verification.response_text, "corrected answer")
        self.assertEqual(response.grounding_rate, 1.0)

    def test_zero_maintenance_intervals_disable_modulo_checks(self) -> None:
        cfg = MiddlewareConfig(prune_interval=0, compress_interval=0)
        cfg.grounding.enabled = False
        client = _RecordingModelClient(["plain answer"])
        mw = ActiveMemoryMiddleware(client=client, embedder=HashEmbedder(dim=8), config=cfg)

        response = mw.send("A message that should not crash maintenance.")

        self.assertEqual(response.text, "plain answer")
        self.assertEqual(mw._turn_count, 1)

    def test_grounded_prompt_reports_nonzero_tuple_stats(self) -> None:
        cfg = MiddlewareConfig()
        cfg.grounding.enabled = True
        cfg.grounding.provenance_injection = True
        cfg.grounding.post_verification = False
        client = _RecordingModelClient(["grounded answer"])
        mw = ActiveMemoryMiddleware(client=client, embedder=HashEmbedder(dim=8), config=cfg)
        for i in range(6):
            mw.tree.insert(f"seed:{i}", f"seed fact {i}")

        response = mw.send("What did we decide?")

        self.assertGreater(response.context_stats.tuples_considered, 0)
        self.assertEqual(
            response.context_stats.tuples_considered,
            response.context_stats.tuples_included,
        )


class TestModelClientProviders(unittest.TestCase):
    def test_anthropic_model_client_normalizes_response(self) -> None:
        client = AnthropicModelClient(client=_FakeAnthropicClient())
        response = client.generate(
            model="claude-test",
            max_tokens=32,
            messages=[{"role": "user", "content": "hello"}],
        )

        self.assertIsInstance(response, GeneratedResponse)
        self.assertEqual(response.text, "anthropic ok")
        self.assertEqual(response.input_tokens, 12)
        self.assertEqual(response.output_tokens, 5)

    def test_openai_model_client_normalizes_response(self) -> None:
        client = OpenAIModelClient(client=_FakeOpenAIClient())
        response = client.generate(
            model="gpt-test",
            max_tokens=32,
            system="be precise",
            messages=[{"role": "user", "content": "hello"}],
        )

        self.assertEqual(response.text, "openai ok")
        self.assertEqual(response.input_tokens, 21)
        self.assertEqual(response.output_tokens, 8)

    def test_create_model_client_requires_openai_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                create_model_client("openai")

    def test_create_model_client_requires_anthropic_key(self) -> None:
        with patch.dict("os.environ", {}, clear=True), \
             patch("active_memory.model_clients._read_claude_code_oauth", return_value=None):
            with self.assertRaises(RuntimeError):
                create_model_client("anthropic")


class TestOfflineBenchmarking(unittest.TestCase):
    def test_offline_benchmark_exposes_recall_gap_between_strategies(self) -> None:
        scenario = FrozenScenario(
            name="tiny_scenario",
            description="A small scenario for prompt coverage tests.",
            transcript=[
                FrozenTurn(1, "user", "Database choice: use PostgreSQL 16.", True, "database"),
                FrozenTurn(2, "assistant", "PostgreSQL 16 is selected."),
                FrozenTurn(3, "user", "We also use Redis for caching."),
                FrozenTurn(4, "assistant", "Redis is selected."),
                FrozenTurn(5, "user", "Let's discuss UI color palettes and onboarding copy."),
                FrozenTurn(6, "assistant", "We can iterate on the visual design later."),
            ],
            probes=[
                Probe(
                    fact_id="database",
                    query="Which database did we choose?",
                    expected_keywords=["postgresql"],
                    planted_at_turn=1,
                )
            ],
        )
        strategies = {
            "full_context": FullContextAdapter(),
            "sliding_window_1": SlidingWindowAdapter(1),
            "active_memory": ActiveMemoryAdapter(SemanticTestEmbedder()),
        }

        results = run_prompt_benchmark(scenario, strategies)
        by_name = {result.strategy: result for result in results}

        self.assertEqual(by_name["full_context"].overall_recall, 1.0)
        self.assertEqual(by_name["sliding_window_1"].overall_recall, 0.0)
        self.assertEqual(by_name["active_memory"].overall_recall, 1.0)


if __name__ == "__main__":
    unittest.main()
