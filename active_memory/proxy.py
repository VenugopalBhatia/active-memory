#!/usr/bin/env python3
"""Transparent context management proxy for Claude Code.

Sits between Claude Code and the Anthropic API. Claude Code doesn't
know it's there — no MCP tools, no extra tokens, no overhead. The
proxy intercepts API calls, ingests the conversation into the B-tree,
reassembles context within a token budget, and forwards the optimised
request to Anthropic.

Usage:
    # Terminal 1: Start the proxy
    python -m active_memory.proxy --port 8080

    # Terminal 2: Point Claude Code at it
    ANTHROPIC_BASE_URL=http://localhost:8080 claude

    # Or for any Anthropic SDK client:
    client = Anthropic(base_url="http://localhost:8080")

How it works:
    Claude Code sends: [system + 200 turns of conversation]
    Proxy does:
      1. Ingest new turns into the B-tree
      2. Score all tuples against the latest user message
      3. Assemble a new message list:
         [system + managed context block + last N turns]
      4. Forward the optimised request to Anthropic
      5. Return the response unchanged to Claude Code

    Claude Code receives: normal API response (unaware of proxy)

The result: Claude Code's effective context never rots because the
proxy continuously curates what goes to the API. Old decisions stay
retrievable even 100+ turns in. Token costs drop because redundant
and irrelevant history is replaced with scored, compressed summaries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import threading

from .types import KVTuple, estimate_tokens


class ProxyConfig:
    """Configuration for the proxy."""
    def __init__(
        self,
        upstream_url: str = "https://api.anthropic.com",
        api_key: str | None = None,
        token_budget: int = 18_000,
        pinned_reserve: int = 4_000,
        recency_window: int = 3,
        embedder_provider: str = "auto",
        embed_model: str = "text-embedding-3-small",
        embed_dim: int = 64,
        max_tuples: int = 16,
        prune_interval: int = 4,
        compress_interval: int = 8,
        state_dir: str | None = None,
        verbose: bool = False,
        # Context reset settings
        reset_threshold: float = 0.70,  # reset when raw tokens exceed this fraction of budget
        reset_briefing_budget: int = 3_000,  # max tokens for the briefing after reset
        reset_recency_turns: int = 2,  # conversation turns to keep verbatim after reset
        reset_cooldown_turns: int = 8,  # min turns between resets
        reset_hysteresis_ratio: float = 0.10,  # required raw growth after a reset
    ):
        self.upstream_url = upstream_url
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.token_budget = token_budget
        self.pinned_reserve = pinned_reserve
        self.recency_window = recency_window
        self.embedder_provider = embedder_provider
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.max_tuples = max_tuples
        self.prune_interval = prune_interval
        self.compress_interval = compress_interval
        self.state_dir = state_dir or str(Path.home() / ".active-memory" / "proxy")
        self.verbose = verbose
        self.reset_threshold = reset_threshold
        self.reset_briefing_budget = reset_briefing_budget
        self.reset_recency_turns = reset_recency_turns
        self.reset_cooldown_turns = reset_cooldown_turns
        self.reset_hysteresis_ratio = reset_hysteresis_ratio


class ContextManager:
    """Manages the B-tree and context assembly for the proxy."""

    def __init__(self, config: ProxyConfig) -> None:
        from .embeddings import create_embedder
        from .scoring import Scorer, ScoringConfig
        from .btree import SemanticBTree, BTreeConfig
        from .assembler import ContextAssembler, AssemblerConfig

        self.cfg = config
        embedder_spec = create_embedder(
            config.embedder_provider,
            dim=config.embed_dim,
            openai_model=config.embed_model,
            verbose=config.verbose,
            stream=sys.stderr,
        )
        self.embedder = embedder_spec.embedder
        self.embedder_spec = embedder_spec
        self.scorer = Scorer(ScoringConfig())
        self.tree = SemanticBTree(
            embedder=self.embedder,
            scorer=self.scorer,
            config=BTreeConfig(max_tuples=config.max_tuples),
        )
        self.assembler = ContextAssembler(
            tree=self.tree,
            config=AssemblerConfig(
                total_budget=config.token_budget,
                pinned_reserve=config.pinned_reserve,
                recency_window=config.recency_window,
            ),
        )

        self._turn_count = 0
        self._message_fingerprints: list[str] = []  # ordered raw-message history for dedupe
        self._state_path = Path(config.state_dir) / "proxy_state.json"
        self._reset_count = 0  # how many times we've done a full context reset
        self._last_reset_turn = -10_000
        self._last_reset_raw_tokens = 0
        self._activation_turn: int | None = None
        self._usage_history: list[dict[str, Any]] = []
        self._operational_facts: list[dict[str, Any]] = []
        self._last_debug: dict = {}  # last assembly decision for /debug

        # Load previous state if exists
        self._load_state()

    def _embedding_state(self) -> dict[str, Any]:
        """Serialize the current embedding configuration for state checks."""
        return {
            "provider": self.embedder_spec.provider,
            "model": getattr(self.embedder, "model", None),
            "dim": int(self.embedder.dim),
        }

    def _state_requires_reembed(self, state: dict[str, Any]) -> bool:
        """Return True when saved tuple embeddings no longer match this embedder."""
        tuples = state.get("tuples", [])
        if not tuples:
            return False

        expected = self._embedding_state()
        saved = state.get("embedding")
        if saved is None:
            return any(
                td.get("key_emb") is not None and len(td["key_emb"]) != expected["dim"]
                for td in tuples
            )

        return any(
            saved.get(key) != expected.get(key)
            for key in ("provider", "model", "dim")
        )

    def _batch_insert_segments(
        self,
        segments: list[tuple[str, str]],
        *,
        batch_size: int = 256,
    ) -> None:
        """Embed and insert many segments efficiently.

        This avoids one embedding API call per tuple when using a real
        provider such as OpenAI.
        """
        if not segments:
            return

        keys = [key for key, _ in segments]
        values = [value for _, value in segments]

        for start in range(0, len(keys), batch_size):
            batch_keys = keys[start : start + batch_size]
            batch_values = values[start : start + batch_size]
            embeddings = self.embedder.embed(batch_keys)
            for key_text, value_text, emb in zip(batch_keys, batch_values, embeddings):
                self.tree.insert_tuple(
                    KVTuple(
                        key_text=key_text,
                        value_text=value_text,
                        key_emb=emb,
                        token_cost=estimate_tokens(value_text),
                    )
                )

    def _segment_message(self, role: str, content: str) -> list[tuple[str, str]]:
        """Split a message into indexable segments."""
        if not content.strip():
            return []

        pending_segments: list[tuple[str, str]] = []
        segments = re.split(r'(?<=[.!?\n])\s+', content)
        buf = ""
        for seg in segments:
            buf += (" " if buf else "") + seg
            if len(buf) >= 60:
                key = f"{role}:{buf[:60]}"
                pending_segments.append((key, buf))
                buf = ""
        if buf and len(buf.strip()) >= 10:
            key = f"{role}:{buf[:60]}"
            pending_segments.append((key, buf))
        return pending_segments

    def _extract_operational_facts(self, role: str, content: str) -> list[dict[str, Any]]:
        """Extract path/file/location hints that should survive resets."""
        path_pattern = re.compile(r'(?<!\w)(?:~?/|/|\./|\.\./)[^\s,;:()]+')
        file_pattern = re.compile(
            r'\b[\w.-]+\.(?:py|js|ts|tsx|jsx|json|md|yaml|yml|toml|html|css|sh|rb|go|java|rs)\b'
        )

        facts: list[dict[str, Any]] = []
        seen: set[str] = set()
        for pattern in (path_pattern, file_pattern):
            for match in pattern.finditer(content):
                text = match.group(0).strip("'\"")
                if len(text) < 3 or text in seen:
                    continue
                seen.add(text)
                facts.append({
                    "text": text,
                    "role": role,
                    "turn": self._turn_count + 1,
                })
        return facts

    def _update_operational_facts(self, messages: list[dict]) -> None:
        """Track recent explicit file/directory mentions for coding continuity."""
        for msg in messages:
            content = self._extract_text(msg)
            role = msg.get("role", "user")
            for fact in self._extract_operational_facts(role, content):
                self._operational_facts = [
                    existing for existing in self._operational_facts
                    if existing["text"] != fact["text"]
                ]
                self._operational_facts.append(fact)
        if len(self._operational_facts) > 24:
            self._operational_facts = self._operational_facts[-24:]

    def _build_operational_context(self) -> str:
        """Return a compact high-priority block of recent path/file hints."""
        if not self._operational_facts:
            return ""

        recent = self._operational_facts[-8:]
        lines = []
        for fact in reversed(recent):
            role = fact["role"]
            turn = fact["turn"]
            lines.append(f"- turn {turn} {role}: {fact['text']}")
        return (
            "[OPERATIONAL CONTEXT]\n"
            "Prefer the most recent file/path references when choosing where to edit.\n"
            "If multiple paths conflict, prefer the newest one rather than older semantically related directories.\n"
            + "\n".join(lines)
        )

    @staticmethod
    def _normalize_message_content(content: Any) -> Any:
        """Normalize content for stable dedupe across SDK message shapes."""
        if isinstance(content, str):
            return [{"type": "text", "text": content.strip()}]
        if isinstance(content, list):
            normalized: list[Any] = []
            for block in content:
                if isinstance(block, str):
                    normalized.append({"type": "text", "text": block.strip()})
                elif isinstance(block, dict) and block.get("type") == "text":
                    block_copy = dict(block)
                    block_copy["text"] = str(block_copy.get("text", "")).strip()
                    normalized.append(block_copy)
                else:
                    normalized.append(block)
            return normalized
        return content

    def process_messages(
        self,
        messages: list[dict],
        system: str | list | None = None,
    ) -> list[dict]:
        """Ingest conversation, assemble optimised context, return new messages.

        This is the core of the proxy. It takes the full message list
        that Claude Code is trying to send, ingests any new messages
        into the B-tree, then returns an optimised message list.
        """
        # -- 1. Ingest new messages --
        pending_segments: list[tuple[str, str]] = []
        incoming_fingerprints = [self._fingerprint_message(msg) for msg in messages]
        prefix_len = 0
        max_prefix = min(len(self._message_fingerprints), len(incoming_fingerprints))
        while (
            prefix_len < max_prefix
            and self._message_fingerprints[prefix_len] == incoming_fingerprints[prefix_len]
        ):
            prefix_len += 1

        for msg in messages[prefix_len:]:
            content = self._extract_text(msg)
            role = msg.get("role", "user")
            pending_segments.extend(self._segment_message(role, content))

        self._batch_insert_segments(pending_segments)
        self._update_operational_facts(messages[prefix_len:])
        self._message_fingerprints = incoming_fingerprints

        if prefix_len < len(incoming_fingerprints):
            self._turn_count += 1
        if self.cfg.verbose:
            print(
                f"  [proxy] registered turn {self._turn_count} "
                f"(messages={len(messages)}, new_segments={len(pending_segments)})",
                file=sys.stderr,
            )

        # -- 2. Bypass if conversation is small --
        # No point managing context if it already fits comfortably.
        # Only activate when raw conversation exceeds 50% of budget.
        # The system prompt is forwarded as a top-level field but still
        # consumes context-window budget upstream, so include it here
        # too — otherwise a large system prompt can push the request
        # over budget while the proxy stays in passthrough mode.
        system_tokens = estimate_tokens(self._system_text(system))
        raw_tokens = system_tokens + sum(
            estimate_tokens(self._extract_text(m)) for m in messages
        )
        activation_threshold = self.cfg.token_budget // 2

        if raw_tokens < activation_threshold:
            self._last_debug = {
                "turn": self._turn_count,
                "action": "passthrough",
                "original_tokens": raw_tokens,
                "activation_threshold": activation_threshold,
                "tree_size": self.tree.size,
            }
            self._save_state()
            if self.cfg.verbose:
                print(
                    f"  [proxy] turn {self._turn_count}: "
                    f"passthrough ({raw_tokens:,} < {activation_threshold:,} threshold) | "
                    f"tree: {self.tree.size} tuples (indexing only)",
                    file=sys.stderr,
                )
            self._record_usage(
                action="passthrough",
                raw_tokens=raw_tokens,
                sent_tokens=raw_tokens,
            )
            return messages  # pass through unchanged, but tree is still building

        # -- 3. Check if we should do a FULL CONTEXT RESET --
        # When the raw conversation gets too large, instead of trying to
        # trim it, we throw it away entirely and rebuild from the B-tree.
        # From the model's perspective, it's a brand new conversation with
        # a curated briefing. Context rot literally cannot survive this.
        reset_threshold_tokens = int(self.cfg.token_budget * self.cfg.reset_threshold)
        reset_growth_tokens = int(self.cfg.token_budget * self.cfg.reset_hysteresis_ratio)
        reset_growth_floor = max(reset_threshold_tokens, self._last_reset_raw_tokens + reset_growth_tokens)
        reset_cooldown_active = (
            self._reset_count > 0
            and (self._turn_count - self._last_reset_turn) <= self.cfg.reset_cooldown_turns
        )
        reset_suppressed_this_turn = False

        if raw_tokens > reset_threshold_tokens:
            if reset_cooldown_active or raw_tokens < reset_growth_floor:
                reset_suppressed_this_turn = True
                self._last_debug = {
                    "turn": self._turn_count,
                    "action": "managed_reset_suppressed",
                    "original_tokens": raw_tokens,
                    "reset_threshold": reset_threshold_tokens,
                    "reset_growth_floor": reset_growth_floor,
                    "reset_cooldown_turns": self.cfg.reset_cooldown_turns,
                    "turns_since_reset": self._turn_count - self._last_reset_turn,
                    "tree_size": self.tree.size,
                }
                if self.cfg.verbose:
                    reason = (
                        "cooldown"
                        if reset_cooldown_active
                        else "insufficient growth"
                    )
                    print(
                        f"  [proxy] turn {self._turn_count}: reset suppressed "
                        f"({reason}; raw={raw_tokens:,}, floor={reset_growth_floor:,})",
                        file=sys.stderr,
                    )
            else:
                return self._context_reset(messages, system)

        # -- 4. Periodic maintenance --
        if self._turn_count % self.cfg.prune_interval == 0:
            self.tree.prune()
        if self._turn_count % self.cfg.compress_interval == 0:
            self.tree.compress_cold_subtrees()

        # -- 3. Assemble optimised context --
        # Get the latest user message for relevance scoring
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = self._extract_text(msg)
                break

        if last_user_msg and self.tree.size > 0:
            query_emb = self.embedder.embed([last_user_msg])[0]
            assembled = self.assembler.assemble(messages, query_emb, system=system)
        else:
            # Not enough context to manage — pass through
            self._last_debug = {
                "turn": self._turn_count,
                "action": "passthrough",
                "original_tokens": raw_tokens,
                "reason": "empty_tree_or_no_user_message",
                "tree_size": self.tree.size,
            }
            self._record_usage(
                action="passthrough",
                raw_tokens=raw_tokens,
                sent_tokens=raw_tokens,
                reset_suppressed=reset_suppressed_this_turn,
            )
            self._save_state()
            return messages

        # -- 4. Build optimised message list --
        optimised = self.assembler.to_messages(
            system,
            assembled,
        )
        operational_context = self._build_operational_context()
        if operational_context:
            operational_tokens = estimate_tokens(operational_context)
            total_budget = self.assembler.cfg.total_budget
            # Drop lowest-priority managed blocks until the operational
            # context block fits.  We rebuild ``optimised`` after dropping
            # so the prompt actually reflects the accounting — popping
            # from ``assembled.managed_blocks`` alone does not remove the
            # blocks from a previously-built ``optimised`` list.
            dropped_any = False
            while (
                assembled.managed_blocks
                and assembled.total_tokens + operational_tokens > total_budget
            ):
                dropped = assembled.managed_blocks.pop()
                assembled.total_tokens -= dropped.token_cost
                assembled.budget_remaining += dropped.token_cost
                if dropped.source == "anchored":
                    assembled.anchored_count = max(0, assembled.anchored_count - 1)
                elif dropped.source == "dependency":
                    assembled.dependency_count = max(0, assembled.dependency_count - 1)
                dropped_any = True
            assembled.tuples_included = len(assembled.managed_blocks)

            if dropped_any:
                # Rebuild the message list now that managed_blocks shrank.
                # to_messages is idempotent w.r.t. wrapper accounting.
                optimised = self.assembler.to_messages(system, assembled)

            if assembled.total_tokens + operational_tokens <= total_budget:
                optimised.insert(0, {
                    "role": "user",
                    "content": operational_context,
                })
                assembled.total_tokens += operational_tokens
                assembled.budget_remaining -= operational_tokens

        # -- 5. Capture debug info --
        original_tokens = system_tokens + sum(
            estimate_tokens(self._extract_text(m)) for m in messages
        )
        selected = [
            {
                "key": b.key,
                "score": round(b.score, 4),
                "tokens": b.token_cost,
                "source": b.source,
            }
            for b in sorted(
                assembled.managed_blocks,
                key=lambda b: b.score,
                reverse=True,
            )
        ]
        self._last_debug = {
            "turn": self._turn_count,
            "action": "managed",
            "original_tokens": original_tokens,
            "assembled_tokens": assembled.total_tokens,
            "savings": max(0, original_tokens - assembled.total_tokens),
            "budget": self.assembler.cfg.total_budget,
            "budget_remaining": assembled.budget_remaining,
            "tree_size": self.tree.size,
            "tree_depth": self.tree.depth(),
            "tuples_considered": assembled.tuples_considered,
            "tuples_included": assembled.tuples_included,
            "anchored_count": assembled.anchored_count,
            "dependency_count": assembled.dependency_count,
            "selected_tuples": selected,
            "query": last_user_msg[:200],
        }

        # -- 6. Save state --
        self._save_state()

        if self.cfg.verbose:
            savings = self._last_debug["savings"]
            print(
                f"  [proxy] turn {self._turn_count}: "
                f"{original_tokens:,} → {assembled.total_tokens:,} tokens "
                f"(-{savings:,}) | "
                f"tree: {self.tree.size} tuples, depth {self.tree.depth()} | "
                f"anchored: {assembled.anchored_count}, "
                f"deps: {assembled.dependency_count}",
                file=sys.stderr,
            )
            # Show top-5 selected tuples
            for entry in selected[:5]:
                print(
                    f"    {entry['score']:.3f}  [{entry['source']:<10}] "
                    f"{entry['key'][:60]}  ({entry['tokens']} tok)",
                    file=sys.stderr,
                )

        self._record_usage(
            action=self._last_debug["action"],
            raw_tokens=original_tokens,
            sent_tokens=assembled.total_tokens,
            reset_suppressed=reset_suppressed_this_turn,
        )
        return optimised

    def _context_reset(
        self,
        messages: list[dict],
        system: str | list | None = None,
    ) -> list[dict]:
        """Perform a full context reset — garbage collection for conversations.

        Stop-the-world, compact, resume with a clean heap. Instead of trimming
        the conversation or asking the LLM to summarize itself (like /compact),
        we throw the conversation away entirely and reload from the B-tree.

        Why this beats /compact:
          - /compact asks the LLM to summarize its own context (expensive,
            single-pass, lossy — summary-of-summary degrades over time)
          - We reload from the B-tree, which has been scoring and organizing
            everything across the entire session. The tree is the source of
            truth, not the LLM's memory.

        The model receives:
          1. A topic-grouped briefing assembled from the tree's clusters
          2. The last N actual conversation turns (for continuity)

        From the model's perspective, this is a brand new conversation with
        excellent background knowledge. Context rot literally cannot accumulate
        because there IS no accumulated context — just a curated briefing from
        a well-organized index.

        The B-tree survives every reset. It has been indexing from turn 1.
        """
        self._reset_count += 1
        self._last_reset_turn = self._turn_count

        # -- 1. Aggressive maintenance before reset --
        self.tree.prune()
        self.tree.compress_cold_subtrees()

        # -- 2. Build a topic-grouped briefing from the tree --
        last_user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user_msg = self._extract_text(msg)
                break

        query_emb = None
        if last_user_msg:
            query_emb = self.embedder.embed([last_user_msg])[0]

        # Use the tree's cluster structure for topic grouping
        clusters = self.tree.tuples_by_cluster()

        # Score each cluster: average score of its tuples, and track the
        # best individual tuples within each cluster
        scored_clusters: list[tuple[float, list[tuple[float, 'KVTuple']]]] = []
        for cluster in clusters:
            scored_tuples = []
            for t in cluster:
                s = (
                    self.tree.scorer.score(t, query_emb)
                    if query_emb is not None
                    else self.tree.scorer.score(t)
                )
                scored_tuples.append((s, t))
            scored_tuples.sort(key=lambda x: x[0], reverse=True)
            avg_score = sum(s for s, _ in scored_tuples) / max(len(scored_tuples), 1)
            scored_clusters.append((avg_score, scored_tuples))

        # Sort clusters by average score (most relevant topic clusters first)
        scored_clusters.sort(key=lambda x: x[0], reverse=True)

        # Pack into briefing budget, preserving topic grouping
        briefing_sections: list[str] = []
        briefing_tokens = 0
        briefing_budget = self.cfg.reset_briefing_budget
        total_facts = 0

        for _cluster_score, scored_tuples in scored_clusters:
            if briefing_tokens >= briefing_budget:
                break

            # Use the highest-scored tuple's key as the topic label
            topic_key = scored_tuples[0][1].key_text if scored_tuples else "misc"
            # Strip role prefix for cleaner display
            topic_label = topic_key.split(":", 1)[-1].strip()[:80]

            section_lines: list[str] = []
            for score, t in scored_tuples:
                if briefing_tokens + t.token_cost > briefing_budget:
                    continue
                section_lines.append(f"  - {t.value_text}")
                briefing_tokens += t.token_cost
                total_facts += 1
                t.touch()

            if section_lines:
                briefing_sections.append(
                    f"[{topic_label}]\n" + "\n".join(section_lines)
                )

        # -- 3. Construct fresh message list --
        fresh_messages: list[dict] = []
        operational_context = self._build_operational_context()
        if operational_context:
            fresh_messages.append({
                "role": "user",
                "content": operational_context,
            })

        if briefing_sections:
            briefing_text = "\n\n".join(briefing_sections)
            fresh_messages.append({
                "role": "user",
                "content": (
                    "[CONTEXT RELOAD: The session history has been compacted. "
                    "Below is a curated briefing of everything important from "
                    "this session, organized by topic. This is authoritative — "
                    "these facts were established in earlier conversation.]\n\n"
                    + briefing_text
                ),
            })
            fresh_messages.append({
                "role": "assistant",
                "content": (
                    f"Got it. I have {total_facts} key facts across "
                    f"{len(briefing_sections)} topics from our session. "
                    "Continuing with full context."
                ),
            })

        # -- 4. Append the last N actual conversation turns --
        recency = self.cfg.reset_recency_turns * 2  # user + assistant
        recent_turns = messages[-recency:] if len(messages) > recency else messages
        fresh_messages.extend(recent_turns)

        # -- 5. Capture debug info and log the reset --
        system_tokens = estimate_tokens(self._system_text(system))
        original_tokens = system_tokens + sum(
            estimate_tokens(self._extract_text(m)) for m in messages
        )
        self._last_reset_raw_tokens = original_tokens
        reset_tokens = system_tokens + sum(
            estimate_tokens(self._extract_text(m)) for m in fresh_messages
        )
        self._last_debug = {
            "turn": self._turn_count,
            "action": "context_reset",
            "reset_number": self._reset_count,
            "original_tokens": original_tokens,
            "reset_tokens": reset_tokens,
            "reduction_pct": round((1 - reset_tokens / max(1, original_tokens)) * 100),
            "topics": len(briefing_sections),
            "facts_loaded": total_facts,
            "tree_size": self.tree.size,
            "recent_turns_kept": len(recent_turns),
        }

        if self.cfg.verbose:
            print(
                f"\n  [proxy] ╔══ CONTEXT RESET #{self._reset_count} ══╗\n"
                f"  [proxy] ║ Raw conversation:  {original_tokens:>8,} tokens\n"
                f"  [proxy] ║ After reset:       {reset_tokens:>8,} tokens\n"
                f"  [proxy] ║ Reduction:         {original_tokens - reset_tokens:>8,} tokens ({self._last_debug['reduction_pct']}%)\n"
                f"  [proxy] ║ Topics:            {len(briefing_sections):>8} clusters\n"
                f"  [proxy] ║ Facts:             {total_facts:>8} loaded\n"
                f"  [proxy] ║ Tree retained:     {self.tree.size:>8} tuples\n"
                f"  [proxy] ║ Recent turns kept: {len(recent_turns):>8} messages\n"
                f"  [proxy] ╚{'═'*30}╝",
                file=sys.stderr,
            )

        self._record_usage(
            action="context_reset",
            raw_tokens=original_tokens,
            sent_tokens=reset_tokens,
        )
        self._save_state()
        return fresh_messages

    @staticmethod
    def _system_text(system: Any) -> str:
        """Normalize an Anthropic ``system`` payload into plain text.

        ``system`` may be ``None``, a plain string, or a list of text
        blocks.  Mirrors :py:meth:`ContextAssembler._system_text` so the
        proxy and assembler agree on what counts as system tokens.
        """
        if system is None:
            return ""
        if isinstance(system, str):
            return system
        if isinstance(system, list):
            parts: list[str] = []
            for block in system:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
            return "\n".join(parts)
        return str(system)

    def _extract_text(self, msg: dict) -> str:
        """Extract text content from a message, including tool blocks.

        Handles all Anthropic content block types:
          - string content (plain text)
          - text blocks (type=text)
          - tool_use blocks (type=tool_use): extracts tool name + JSON input
          - tool_result blocks (type=tool_result): extracts text content
        """
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict):
                    block_type = block.get("type", "")
                    if block_type == "text":
                        parts.append(block.get("text", ""))
                    elif block_type == "tool_use":
                        # Index the tool name and a compact repr of its input
                        name = block.get("name", "unknown_tool")
                        tool_input = block.get("input", {})
                        input_repr = json.dumps(tool_input, separators=(",", ":"))
                        # Cap the input repr to avoid bloating the index
                        if len(input_repr) > 500:
                            input_repr = input_repr[:500] + "..."
                        parts.append(f"[tool_use:{name}] {input_repr}")
                    elif block_type == "tool_result":
                        # tool_result content can be string or list of blocks
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            parts.append(result_content)
                        elif isinstance(result_content, list):
                            for sub in result_content:
                                if isinstance(sub, str):
                                    parts.append(sub)
                                elif isinstance(sub, dict) and sub.get("type") == "text":
                                    parts.append(sub.get("text", ""))
            return " ".join(p for p in parts if p)
        return str(content)

    def _fingerprint_message(self, msg: dict) -> str:
        """Stable message fingerprint for cross-request dedupe.

        Normalizes plain-text messages so that ``"hi"`` and
        ``[{"type":"text","text":"hi"}]`` hash identically — the
        Anthropic API returns assistant turns as a list of blocks, but
        client SDKs sometimes replay them as bare strings. Tool-bearing
        messages keep their structural distinctness because the
        non-text blocks remain in the normalized payload.
        """
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        normalized = self._normalize_message_content(content)
        payload = json.dumps(
            {"role": role, "content": normalized},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _usage_bar(value: float, width: int = 24) -> str:
        value = max(0.0, min(1.0, value))
        filled = int(round(value * width))
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _usage_chart(
        raw_values: list[float],
        sent_values: list[float],
        markers: list[str],
        *,
        max_value: float,
        height: int = 8,
    ) -> list[str]:
        """Render a taller ASCII chart for raw vs sent tokens."""
        if not raw_values or not sent_values:
            return []

        n = min(len(raw_values), len(sent_values), len(markers))
        raw_values = raw_values[-n:]
        sent_values = sent_values[-n:]
        markers = markers[-n:]
        height = max(4, height)
        max_value = max(1.0, max_value)

        canvas = [[" " for _ in range(n)] for _ in range(height)]

        def to_row(value: float) -> int:
            ratio = max(0.0, min(1.0, value / max_value))
            return max(0, min(height - 1, int(round((1.0 - ratio) * (height - 1)))))

        for idx, (raw, sent) in enumerate(zip(raw_values, sent_values)):
            raw_row = to_row(raw)
            sent_row = to_row(sent)
            if raw_row == sent_row:
                canvas[raw_row][idx] = "◆"
            else:
                canvas[raw_row][idx] = "H"
                canvas[sent_row][idx] = "P"

        lines: list[str] = []
        for row_idx, row in enumerate(canvas):
            level = int(round(max_value * (height - 1 - row_idx) / max(1, height - 1)))
            lines.append(f"  {level:>7,} | {''.join(row)}")
        lines.append("          +" + "-" * (n + 2))
        lines.append("           " + "".join(markers))
        lines.append("           " + "".join("^" if i % 5 == 0 else " " for i in range(n)))
        lines.append("           Legend: H raw history  P proxy sent  ◆ overlap")
        return lines

    def _record_usage(
        self,
        *,
        action: str,
        raw_tokens: int,
        sent_tokens: int,
        reset_suppressed: bool = False,
    ) -> None:
        """Persist per-turn usage for the /usage endpoint."""
        events: list[str] = []
        if self._activation_turn is None and action != "passthrough":
            self._activation_turn = self._turn_count
            events.append("A")
        if action == "context_reset":
            events.append("R")
        elif action == "passthrough":
            events.append("P")
        if reset_suppressed:
            events.append("S")

        self._usage_history.append({
            "turn": self._turn_count,
            "timestamp": time.time(),
            "action": action,
            "raw_tokens": raw_tokens,
            "sent_tokens": sent_tokens,
            "events": events,
        })
        if len(self._usage_history) > 240:
            self._usage_history = self._usage_history[-240:]

    def render_usage_report(self, last_n: int = 30) -> str:
        """Render a terminal-friendly usage graph for recent turns."""
        budget = self.cfg.token_budget
        activation_threshold = budget // 2
        reset_threshold = int(budget * self.cfg.reset_threshold)

        if not self._usage_history:
            return "active-memory usage\nNo usage samples recorded yet.\n"

        latest = self._usage_history[-1]
        raw_ratio = latest["raw_tokens"] / max(1, budget)
        sent_ratio = latest["sent_tokens"] / max(1, budget)
        raw_left = max(0, budget - latest["raw_tokens"])
        sent_left = max(0, budget - latest["sent_tokens"])
        resets = sum(1 for row in self._usage_history if row["action"] == "context_reset")
        recent = self._usage_history[-last_n:]
        raw_values = [float(row["raw_tokens"]) for row in recent]
        sent_values = [float(row["sent_tokens"]) for row in recent]
        raw_growth = 0.0
        sent_growth = 0.0
        if len(recent) >= 2:
            raw_start = max(1.0, raw_values[0])
            sent_start = max(1.0, sent_values[0])
            raw_growth = (raw_values[-1] - raw_values[0]) / raw_start
            sent_growth = (sent_values[-1] - sent_values[0]) / sent_start
        marker_row = []
        for row in recent:
            marker = "".join(row["events"])
            if "R" in marker:
                marker_row.append("R")
            elif "A" in marker:
                marker_row.append("A")
            elif row["action"] == "managed":
                marker_row.append("M")
            elif row["action"] == "managed_reset_suppressed":
                marker_row.append("S")
            else:
                marker_row.append("P")

        lines = [
            "active-memory usage",
            f"Budget:        {budget:>8,} tok",
            f"Activation:    {activation_threshold:>8,} tok",
            f"Reset:         {reset_threshold:>8,} tok",
            "",
            f"Raw history:   {latest['raw_tokens']:>8,} tok  {raw_ratio:>6.1%} used  {raw_left:>8,} left",
            f"Proxy sent:    {latest['sent_tokens']:>8,} tok  {sent_ratio:>6.1%} used  {sent_left:>8,} left",
            f"Current mode:  {latest['action']}",
            f"Activation at: {self._activation_turn if self._activation_turn is not None else '-'}",
            f"Reset count:   {resets}",
            f"Growth ({len(recent):>2} turns): raw {raw_growth:+6.1%}  sent {sent_growth:+6.1%}",
            "",
            "Trend:",
        ]

        lines.extend(
            self._usage_chart(
                raw_values,
                sent_values,
                marker_row,
                max_value=float(budget),
                height=8,
            )
        )
        lines.extend([
            "",
            "Legend: P passthrough  A activation  M managed  R reset  S reset-suppressed",
            "Recent turns:",
        ])

        for row in recent:
            ratio = row["sent_tokens"] / max(1, budget)
            bar = self._usage_bar(ratio)
            marker = "".join(row["events"]) or {
                "managed": "M",
                "context_reset": "R",
                "managed_reset_suppressed": "S",
                "passthrough": "P",
            }.get(row["action"], "?")
            savings = max(0, row["raw_tokens"] - row["sent_tokens"])
            lines.append(
                f"  t{row['turn']:>3} {bar} "
                f"{row['sent_tokens']:>7,}/{budget:,} ({ratio:>5.1%})  "
                f"raw={row['raw_tokens']:>7,}  saved={savings:>7,}  {marker}"
            )

        return "\n".join(lines) + "\n"

    def _index_assistant_blocks(self, content_blocks: list[Any]) -> None:
        """Index an assistant response and store its fingerprint.

        We fingerprint using the *original* content block list so the
        next request from the client (which replays Anthropic's exact
        response shape) prefix-matches and we don't re-ingest.
        """
        text_parts: list[str] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            btype = block.get("type", "")
            if btype == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif btype == "tool_use":
                name = block.get("name", "unknown_tool")
                tool_input = block.get("input", {})
                try:
                    input_repr = json.dumps(tool_input, separators=(",", ":"))
                except (TypeError, ValueError):
                    input_repr = str(tool_input)
                if len(input_repr) > 500:
                    input_repr = input_repr[:500] + "..."
                text_parts.append(f"[tool_use:{name}] {input_repr}")

        assistant_text = " ".join(p for p in text_parts if p).strip()
        if not assistant_text:
            return

        assistant_msg = {"role": "assistant", "content": content_blocks}
        self._batch_insert_segments(self._segment_message("assistant", assistant_text))
        self._message_fingerprints.append(self._fingerprint_message(assistant_msg))
        self._save_state()

    def ingest_assistant_response(self, response_data: dict[str, Any]) -> None:
        """Index a completed non-streaming assistant response immediately."""
        content_blocks = response_data.get("content")
        if not isinstance(content_blocks, list):
            return
        self._index_assistant_blocks(content_blocks)

    def ingest_streaming_response(self, raw_sse: str) -> None:
        """Parse an SSE stream and index the assistant response.

        Reconstructs the full ``content`` block list from
        ``content_block_start`` / ``content_block_delta`` /
        ``content_block_stop`` events so the resulting fingerprint
        matches what the client will replay on the next request.
        Handles both ``text_delta`` (text blocks) and
        ``input_json_delta`` (tool_use input).
        """
        content_blocks: list[dict] = []
        current_block: dict | None = None
        json_buf: list[str] = []

        for line in raw_sse.splitlines():
            if not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue

            event_type = event.get("type", "")
            if event_type == "content_block_start":
                block = event.get("content_block")
                if isinstance(block, dict):
                    current_block = dict(block)
                    json_buf = []
                    content_blocks.append(current_block)
            elif event_type == "content_block_delta" and current_block is not None:
                delta = event.get("delta", {})
                dtype = delta.get("type", "")
                if dtype == "text_delta":
                    current_block["text"] = (
                        current_block.get("text", "") + delta.get("text", "")
                    )
                elif dtype == "input_json_delta":
                    json_buf.append(delta.get("partial_json", ""))
            elif event_type == "content_block_stop" and current_block is not None:
                if current_block.get("type") == "tool_use" and json_buf:
                    try:
                        current_block["input"] = json.loads("".join(json_buf))
                    except json.JSONDecodeError:
                        # Leave whatever input was on the start event
                        pass
                current_block = None
                json_buf = []

        if not content_blocks:
            return
        self._index_assistant_blocks(content_blocks)

    def _save_state(self) -> None:
        state_dir = Path(self.cfg.state_dir)
        state_dir.mkdir(parents=True, exist_ok=True)
        tuples = self.tree.all_tuples()
        state = {
            "turn_count": self._turn_count,
            "reset_count": self._reset_count,
            "last_reset_turn": self._last_reset_turn,
            "last_reset_raw_tokens": self._last_reset_raw_tokens,
            "activation_turn": self._activation_turn,
            "usage_history": self._usage_history,
            "operational_facts": self._operational_facts,
            "message_fingerprints": self._message_fingerprints,
            "embedding": self._embedding_state(),
            "tuples": [
                {
                    "key_text": t.key_text,
                    "value_text": t.value_text,
                    "key_emb": t.key_emb.tolist() if t.key_emb is not None else None,
                    "id": t.id,
                    "created_at": t.created_at,
                    "last_accessed": t.last_accessed,
                    "hit_count": t.hit_count,
                    "token_cost": t.token_cost,
                    "references": t.references,
                    "referenced_by": t.referenced_by,
                    "tags": t.tags,
                }
                for t in tuples
            ],
        }
        # Atomic write: write to temp file then rename to avoid corruption
        # if the process is killed mid-write.
        tmp_path = self._state_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(state))
        tmp_path.replace(self._state_path)

    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            import numpy as np
            from .types import KVTuple

            state = json.loads(self._state_path.read_text())
            self._turn_count = state.get("turn_count", 0)
            self._reset_count = state.get("reset_count", 0)
            self._last_reset_turn = state.get("last_reset_turn", -10_000)
            self._last_reset_raw_tokens = state.get("last_reset_raw_tokens", 0)
            self._activation_turn = state.get("activation_turn")
            self._usage_history = list(state.get("usage_history", []))
            self._operational_facts = list(state.get("operational_facts", []))
            self._message_fingerprints = list(state.get("message_fingerprints", []))
            reembed = self._state_requires_reembed(state)
            if reembed and self.cfg.verbose:
                print(
                    "  [proxy] Saved embeddings do not match current embedder; re-embedding state",
                    file=sys.stderr,
                )

            tuple_dicts = list(state.get("tuples", []))
            reembedded: list[Any] | None = None
            if reembed and tuple_dicts:
                keys = [td["key_text"] for td in tuple_dicts]
                reembedded = []
                for start in range(0, len(keys), 256):
                    reembedded.extend(self.embedder.embed(keys[start : start + 256]))

            for idx, td in enumerate(tuple_dicts):
                t = KVTuple(
                    key_text=td["key_text"],
                    value_text=td["value_text"],
                    key_emb=(
                        reembedded[idx]
                        if reembedded is not None
                        else (
                            np.array(td["key_emb"], dtype=np.float32)
                            if td.get("key_emb") else None
                        )
                    ),
                    id=td["id"],
                    created_at=td["created_at"],
                    last_accessed=td["last_accessed"],
                    hit_count=td["hit_count"],
                    token_cost=td["token_cost"],
                    references=td.get("references", []),
                    referenced_by=td.get("referenced_by", []),
                    tags=td.get("tags", []),
                )
                self.tree.insert_tuple(t)

            if self.cfg.verbose:
                print(
                    f"  [proxy] Loaded state: {self.tree.size} tuples, "
                    f"turn {self._turn_count}",
                    file=sys.stderr,
                )
        except Exception as e:
            if self.cfg.verbose:
                print(f"  [proxy] Failed to load state: {e}", file=sys.stderr)


def _serialize_config(cm: ContextManager) -> dict:
    """Serialize the current running configuration as a JSON-friendly dict."""
    return {
        "budget": cm.cfg.token_budget,
        "recency_window": cm.cfg.recency_window,
        "reset_threshold": cm.cfg.reset_threshold,
        "reset_briefing_budget": cm.cfg.reset_briefing_budget,
        "reset_recency_turns": cm.cfg.reset_recency_turns,
        "reset_cooldown_turns": cm.cfg.reset_cooldown_turns,
        "reset_hysteresis_ratio": cm.cfg.reset_hysteresis_ratio,
        "embedder": cm.embedder_spec.description,
        "scoring": {
            "recency_weight": cm.scorer.cfg.recency_weight,
            "frequency_weight": cm.scorer.cfg.frequency_weight,
            "relevance_weight": cm.scorer.cfg.relevance_weight,
            "affinity_weight": cm.scorer.cfg.affinity_weight,
            "recency_half_life": cm.scorer.cfg.recency_half_life,
            "floor": cm.scorer.cfg.floor,
        },
        "assembler": {
            "total_budget": cm.assembler.cfg.total_budget,
            "pinned_reserve": cm.assembler.cfg.pinned_reserve,
            "recency_window": cm.assembler.cfg.recency_window,
            "managed_top_k": cm.assembler.cfg.managed_top_k,
            "anchor_relevance_threshold": cm.assembler.cfg.anchor_relevance_threshold,
            "anchor_budget_fraction": cm.assembler.cfg.anchor_budget_fraction,
            "budget_pressure": cm.assembler.cfg.budget_pressure,
            "dependency_pull": cm.assembler.cfg.dependency_pull,
            "dependency_budget_fraction": cm.assembler.cfg.dependency_budget_fraction,
        },
        "btree": {
            "max_tuples": cm.tree.cfg.max_tuples,
            "min_tuples": cm.tree.cfg.min_tuples,
            "compress_threshold": cm.tree.cfg.compress_threshold,
        },
    }


class ProxyHandler(BaseHTTPRequestHandler):
    """HTTP handler that proxies Anthropic API requests with context management."""

    context_manager: ContextManager
    proxy_config: ProxyConfig

    def _request_path(self) -> str:
        """Return the URL path without query/fragment for route matching."""
        return urlsplit(self.path).path

    def do_GET(self):
        """Handle GET requests (health, stats, debug, config)."""
        route = self._request_path()
        if route == "/health":
            self._respond_json(200, {
                "status": "ok",
                "tree_size": self.context_manager.tree.size,
                "tree_depth": self.context_manager.tree.depth(),
                "turn_count": self.context_manager._turn_count,
            })
        elif route == "/stats":
            tree = self.context_manager.tree
            all_tuples = tree.all_tuples()
            self._respond_json(200, {
                "tree_size": tree.size,
                "tree_depth": tree.depth(),
                "total_nodes": len(tree.all_nodes()),
                "total_tokens_stored": sum(t.token_cost for t in all_tuples),
                "total_hits": sum(t.hit_count for t in all_tuples),
                "turn_count": self.context_manager._turn_count,
            })
        elif route == "/debug":
            self._respond_json(200, self.context_manager._last_debug or {
                "message": "No turns processed yet.",
            })
        elif route == "/config":
            self._respond_json(200, _serialize_config(self.context_manager))
        elif route == "/usage":
            self._respond_text(200, self.context_manager.render_usage_report())
        else:
            self.send_error(404)

    def do_POST(self):
        """Handle POST requests (API calls, config updates)."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        route = self._request_path()

        if getattr(self, "proxy_config", None) and self.proxy_config.verbose:
            print(
                f"  [proxy] inbound POST {route} "
                f"(content-length={content_length})",
                file=sys.stderr,
            )

        if route == "/config":
            self._handle_config_update(body)
        elif route == "/v1/messages":
            self._handle_messages(body)
        else:
            # Pass through non-messages endpoints unchanged
            self._proxy_passthrough(body)

    def _handle_config_update(self, body: bytes):
        """Hot-reload tunable parameters from a JSON POST body."""
        try:
            updates = json.loads(body)
        except json.JSONDecodeError:
            self._respond_json(400, {"error": "Invalid JSON"})
            return

        cm = self.context_manager
        changed: dict[str, dict] = {}

        # -- Proxy-level settings --
        if "budget" in updates:
            old = cm.cfg.token_budget
            cm.cfg.token_budget = int(updates["budget"])
            cm.assembler.cfg.total_budget = cm.cfg.token_budget
            changed["budget"] = {"old": old, "new": cm.cfg.token_budget}

        if "recency_window" in updates:
            old = cm.cfg.recency_window
            cm.cfg.recency_window = int(updates["recency_window"])
            cm.assembler.cfg.recency_window = cm.cfg.recency_window
            changed["recency_window"] = {"old": old, "new": cm.cfg.recency_window}

        if "reset_threshold" in updates:
            old = cm.cfg.reset_threshold
            cm.cfg.reset_threshold = float(updates["reset_threshold"])
            changed["reset_threshold"] = {"old": old, "new": cm.cfg.reset_threshold}

        if "reset_briefing_budget" in updates:
            old = cm.cfg.reset_briefing_budget
            cm.cfg.reset_briefing_budget = int(updates["reset_briefing_budget"])
            changed["reset_briefing_budget"] = {"old": old, "new": cm.cfg.reset_briefing_budget}

        if "reset_recency_turns" in updates:
            old = cm.cfg.reset_recency_turns
            cm.cfg.reset_recency_turns = int(updates["reset_recency_turns"])
            changed["reset_recency_turns"] = {"old": old, "new": cm.cfg.reset_recency_turns}

        if "reset_cooldown_turns" in updates:
            old = cm.cfg.reset_cooldown_turns
            cm.cfg.reset_cooldown_turns = int(updates["reset_cooldown_turns"])
            changed["reset_cooldown_turns"] = {"old": old, "new": cm.cfg.reset_cooldown_turns}

        if "reset_hysteresis_ratio" in updates:
            old = cm.cfg.reset_hysteresis_ratio
            cm.cfg.reset_hysteresis_ratio = float(updates["reset_hysteresis_ratio"])
            changed["reset_hysteresis_ratio"] = {"old": old, "new": cm.cfg.reset_hysteresis_ratio}

        # -- Scoring weights --
        if "scoring" in updates and isinstance(updates["scoring"], dict):
            s = updates["scoring"]
            cfg = cm.scorer.cfg
            for key in ("recency_weight", "frequency_weight", "relevance_weight",
                        "affinity_weight", "recency_half_life", "floor"):
                if key in s:
                    old = getattr(cfg, key)
                    setattr(cfg, key, float(s[key]))
                    changed[f"scoring.{key}"] = {"old": old, "new": float(s[key])}
            # Recompute lambda if half-life changed
            if "recency_half_life" in s:
                import math
                cm.scorer._lambda = math.log(2) / cfg.recency_half_life

        # -- Assembler settings --
        if "assembler" in updates and isinstance(updates["assembler"], dict):
            a = updates["assembler"]
            cfg = cm.assembler.cfg
            for key in ("budget_pressure", "anchor_relevance_threshold",
                        "anchor_budget_fraction", "dependency_budget_fraction",
                        "managed_top_k", "pinned_reserve"):
                if key in a:
                    old = getattr(cfg, key)
                    val = float(a[key]) if isinstance(a[key], float) else int(a[key])
                    setattr(cfg, key, val)
                    changed[f"assembler.{key}"] = {"old": old, "new": val}

        if "dependency_pull" in updates.get("assembler", {}):
            old = cm.assembler.cfg.dependency_pull
            cm.assembler.cfg.dependency_pull = bool(updates["assembler"]["dependency_pull"])
            changed["assembler.dependency_pull"] = {"old": old, "new": cm.assembler.cfg.dependency_pull}

        if changed and cm.cfg.verbose:
            print(f"  [proxy] Config updated: {list(changed.keys())}", file=sys.stderr)

        self._respond_json(200, {
            "updated": changed,
            "current": _serialize_config(cm),
        })

    def _handle_messages(self, body: bytes):
        """Intercept /v1/messages, manage context, forward to upstream."""
        try:
            request_data = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        original_messages = request_data.get("messages", [])
        system = request_data.get("system")

        # -- Context management --
        optimised_messages = self.context_manager.process_messages(
            original_messages, system
        )

        # Replace messages in the request
        request_data["messages"] = optimised_messages

        # -- Forward to upstream Anthropic API --
        upstream_body = json.dumps(request_data).encode("utf-8")
        self._proxy_to_upstream(upstream_body, request_data.get("stream", False))

    def _build_upstream_headers(self) -> dict[str, str]:
        """Build headers for the upstream request.

        Forwards auth headers from the incoming request (x-api-key or
        Authorization) so the proxy works with both API keys and OAuth
        login sessions. Falls back to the configured API key if the
        client didn't send auth headers.
        """
        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "accept": self.headers.get("accept", "application/json"),
            "anthropic-version": self.headers.get(
                "anthropic-version", "2023-06-01"
            ),
        }

        # Forward auth: prefer what the client sent, fall back to config
        client_api_key = self.headers.get("x-api-key")
        client_auth = self.headers.get("authorization")
        if client_api_key:
            headers["x-api-key"] = client_api_key
        elif client_auth:
            headers["authorization"] = client_auth
        elif self.proxy_config.api_key:
            headers["x-api-key"] = self.proxy_config.api_key

        # Copy other relevant Anthropic headers
        for key in ("anthropic-beta", "anthropic-dangerous-direct-browser-access"):
            val = self.headers.get(key)
            if val:
                headers[key] = val

        return headers

    def _proxy_to_upstream(self, body: bytes, is_stream: bool):
        """Forward the request to the real Anthropic API."""
        url = f"{self.proxy_config.upstream_url}{self.path}"
        headers = self._build_upstream_headers()

        req = Request(url, data=body, headers=headers, method="POST")

        try:
            with urlopen(req) as response:
                status = response.status
                resp_headers = dict(response.headers)
                self.send_response(status)
                for key, val in response.headers.items():
                    key_lower = key.lower()
                    if key_lower in {
                        "connection",
                        "keep-alive",
                        "proxy-authenticate",
                        "proxy-authorization",
                        "te",
                        "trailers",
                        "transfer-encoding",
                        "upgrade",
                    }:
                        continue
                    if key_lower == "content-length" and is_stream:
                        continue
                    if val:
                        self.send_header(key, val)
                self.end_headers()
                if is_stream:
                    stream_buf = bytearray()
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        stream_buf.extend(chunk)
                        self.wfile.write(chunk)
                        self.wfile.flush()
                    # Parse accumulated SSE stream and ingest assistant text
                    try:
                        self.context_manager.ingest_streaming_response(
                            stream_buf.decode("utf-8", errors="replace")
                        )
                    except Exception:
                        if self.proxy_config.verbose:
                            print(
                                "  [proxy] Skipped assistant ingestion from stream",
                                file=sys.stderr,
                            )
                else:
                    response_body = response.read()
                    try:
                        self.context_manager.ingest_assistant_response(
                            json.loads(response_body)
                        )
                    except json.JSONDecodeError:
                        if self.proxy_config.verbose:
                            print(
                                "  [proxy] Skipped assistant ingestion: upstream body was not valid JSON",
                                file=sys.stderr,
                            )
                    self.wfile.write(response_body)

        except HTTPError as e:
            if self.proxy_config.verbose:
                print(
                    f"  [proxy] upstream HTTP {e.code} for {self.path}",
                    file=sys.stderr,
                )
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except URLError as e:
            self._respond_json(502, {
                "type": "upstream_connection_error",
                "message": f"Failed to reach upstream Anthropic API: {e.reason}",
            })
        except Exception as e:
            self._respond_json(502, {
                "type": "proxy_error",
                "message": f"Unexpected proxy error: {e}",
            })

    def _proxy_passthrough(self, body: bytes):
        """Forward non-messages requests unchanged."""
        url = f"{self.proxy_config.upstream_url}{self.path}"
        headers = self._build_upstream_headers()
        req = Request(url, data=body, headers=headers, method="POST")

        try:
            with urlopen(req) as response:
                self.send_response(response.status)
                for key, val in response.headers.items():
                    key_lower = key.lower()
                    if key_lower in {
                        "connection",
                        "keep-alive",
                        "proxy-authenticate",
                        "proxy-authorization",
                        "te",
                        "trailers",
                        "transfer-encoding",
                        "upgrade",
                    }:
                        continue
                    if val:
                        self.send_header(key, val)
                self.end_headers()
                self.wfile.write(response.read())
        except HTTPError as e:
            self.send_response(e.code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(e.read())
        except URLError as e:
            self._respond_json(502, {
                "type": "upstream_connection_error",
                "message": f"Failed to reach upstream Anthropic API: {e.reason}",
            })
        except Exception as e:
            self._respond_json(502, {
                "type": "proxy_error",
                "message": f"Unexpected proxy error: {e}",
            })

    def _respond_json(self, status: int, data: dict):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())

    def _respond_text(self, status: int, text: str):
        self.send_response(status)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(text.encode("utf-8"))

    def log_message(self, format, *args):
        """Suppress default access logs unless verbose."""
        if self.proxy_config.verbose:
            super().log_message(format, *args)


def main():
    parser = argparse.ArgumentParser(
        description="active-memory proxy for Claude Code",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    # Terminal 1
    python -m active_memory.proxy --port 8080 --verbose

    # Terminal 2
    ANTHROPIC_BASE_URL=http://localhost:8080 claude
        """,
    )
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--budget", type=int, default=18_000,
                        help="Token budget for managed context")
    parser.add_argument("--recency", type=int, default=3,
                        help="Number of recent turns to always include")
    parser.add_argument(
        "--embedder",
        choices=["auto", "hash", "openai"],
        default="auto",
        help="Embedding provider (default: auto, prefers OpenAI when configured)",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="OpenAI embedding model when using auto/openai",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
        help="Embedding dimension for hash embedder fallback",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--upstream", default="https://api.anthropic.com",
                        help="Upstream Anthropic API URL")
    parser.add_argument(
        "--state-dir",
        default=None,
        help="Directory for proxy state (default: ~/.active-memory/proxy)",
    )
    parser.add_argument(
        "--reset-threshold",
        type=float,
        default=0.70,
        help="Fraction of budget that triggers a full context reset",
    )
    parser.add_argument(
        "--reset-briefing-budget",
        type=int,
        default=3_000,
        help="Token budget for the reset briefing",
    )
    parser.add_argument(
        "--reset-recency-turns",
        type=int,
        default=2,
        help="Conversation turns to keep verbatim after reset",
    )
    parser.add_argument(
        "--reset-cooldown-turns",
        type=int,
        default=8,
        help="Minimum turns between consecutive context resets",
    )
    parser.add_argument(
        "--reset-hysteresis-ratio",
        type=float,
        default=0.10,
        help="Required additional raw token growth after a reset before resetting again",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a JSON config file (overrides defaults, CLI flags override config)",
    )

    args = parser.parse_args()

    # -- Load config file (explicit path or auto-discover) --
    from .config import load_config, apply_overrides

    file_overrides = load_config(args.config)
    if file_overrides and (args.verbose or file_overrides.get("verbose")):
        src = args.config or "~/.active-memory/config.json"
        print(f"  [proxy] Loaded config from {src}", file=sys.stderr)

    proxy_section = file_overrides.get("proxy", {})

    # Resolve: defaults < config file < CLI flags
    # argparse defaults are used only when the user didn't pass the flag.
    # We detect "user passed this flag" by comparing to the default.
    def _resolve(cli_val, cli_default, file_key, convert=None):
        """Pick: CLI flag if explicitly set, else config file, else default."""
        if cli_val != cli_default:
            return cli_val  # user explicitly passed this flag
        # Check proxy section first, then top-level
        for source in (proxy_section, file_overrides):
            if file_key in source:
                val = source[file_key]
                return convert(val) if convert else val
        return cli_val  # fall back to default

    config = ProxyConfig(
        upstream_url=_resolve(args.upstream, "https://api.anthropic.com", "upstream"),
        token_budget=_resolve(args.budget, 18_000, "budget", int),
        recency_window=_resolve(args.recency, 3, "recency_window", int),
        embedder_provider=_resolve(args.embedder, "auto", "embedder"),
        embed_model=_resolve(args.embed_model, "text-embedding-3-small", "embed_model"),
        embed_dim=_resolve(args.embed_dim, 64, "embed_dim", int),
        state_dir=_resolve(args.state_dir, None, "state_dir"),
        verbose=args.verbose or file_overrides.get("verbose", False) or proxy_section.get("verbose", False),
        reset_threshold=_resolve(args.reset_threshold, 0.70, "reset_threshold", float),
        reset_briefing_budget=_resolve(args.reset_briefing_budget, 3_000, "reset_briefing_budget", int),
        reset_recency_turns=_resolve(args.reset_recency_turns, 2, "reset_recency_turns", int),
        reset_cooldown_turns=_resolve(args.reset_cooldown_turns, 8, "reset_cooldown_turns", int),
        reset_hysteresis_ratio=_resolve(args.reset_hysteresis_ratio, 0.10, "reset_hysteresis_ratio", float),
    )

    if not config.api_key:
        print(
            "  Note: ANTHROPIC_API_KEY not set. The proxy will forward\n"
            "  auth headers from the client (works with Claude Code login).",
            file=sys.stderr,
        )

    # Create context manager
    ctx_manager = ContextManager(config)

    # Apply scoring/btree/assembler/grounding overrides from config file
    if "scoring" in file_overrides:
        apply_overrides(ctx_manager.scorer.cfg, file_overrides["scoring"])
        import math
        ctx_manager.scorer._lambda = math.log(2) / ctx_manager.scorer.cfg.recency_half_life

    if "btree" in file_overrides:
        apply_overrides(ctx_manager.tree.cfg, file_overrides["btree"])

    if "assembler" in file_overrides:
        apply_overrides(ctx_manager.assembler.cfg, file_overrides["assembler"])

    # Wire into handler
    ProxyHandler.context_manager = ctx_manager
    ProxyHandler.proxy_config = config

    server = HTTPServer((args.host, args.port), ProxyHandler)

    auth_mode = "API key" if config.api_key else "passthrough (client auth)"
    config_note = f"\n  Config:     {args.config}" if args.config else ""
    print(f"""
  ╔═══════════════════════════════════════════════╗
  ║         active-memory proxy v0.1.0            ║
  ╚═══════════════════════════════════════════════╝
  Listening:  http://{args.host}:{args.port}
  Upstream:   {config.upstream_url}
  Auth:       {auth_mode}
  Budget:     {config.token_budget:,} tokens
  Recency:    {config.recency_window} turns pinned
  Embedding:  {ctx_manager.embedder_spec.description}
  State:      {config.state_dir}{config_note}

  Endpoints:
    /health     Health check + tree stats
    /stats      Detailed metrics
    /usage      Terminal-friendly usage graph
    /debug      Last assembly decision (selected tuples, scores)
    /config     GET current config / POST to hot-reload parameters

  Start Claude Code with:
    ANTHROPIC_BASE_URL=http://{args.host}:{args.port} claude
""", file=sys.stderr)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down...", file=sys.stderr)
        ctx_manager._save_state()
        server.shutdown()


if __name__ == "__main__":
    main()
