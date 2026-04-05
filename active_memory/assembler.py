"""Context assembler — builds a token-budgeted prompt from the B-tree.

The assembler combines:
  1. A *pinned* section (system prompt, recent turns) — always included.
  2. A *managed* section assembled by querying the B-tree for the most
     relevant + highest-scored KV tuples, packed greedily into the
     remaining token budget.
  3. A *ground-truth anchor* pass that force-includes highly relevant
     tuples even if they scored low on recency/frequency — prevents
     hallucination by ensuring the model has stored facts when answering
     queries about those topics.
  4. A *dependency pull* pass that includes structurally related tuples
     (call graph neighbors) — if function A is included and it calls B,
     B gets pulled in too.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .btree import SemanticBTree
from .types import Embedding, KVTuple, cosine_sim, estimate_tokens


@dataclass
class AssemblerConfig:
    total_budget: int = 100_000       # max tokens for the full prompt
    pinned_reserve: int = 8_000       # tokens reserved for pinned content
    recency_window: int = 4           # last N conversation turns always kept
    managed_top_k: int = 200          # max tuples to score from the tree

    # Ground-truth anchoring: force-include tuples with relevance above
    # this threshold, even if their composite score is low.
    anchor_relevance_threshold: float = 0.60
    # Max tokens to spend on anchored tuples (prevents runaway anchoring)
    anchor_budget_fraction: float = 0.25

    # Dependency pulling: include structural neighbors of selected tuples
    dependency_pull: bool = True
    # Max tokens to spend on pulled dependencies
    dependency_budget_fraction: float = 0.10

    # Budget pressure: intentionally use LESS than the full budget.
    # A value of 0.6 means "fill only 60% of the available budget."
    # Lower values = shorter context = stronger per-token attention,
    # but higher risk of missing relevant information.
    # Range: 0.3 (very aggressive) to 1.0 (fill everything).
    budget_pressure: float = 0.85


@dataclass
class ManagedBlock:
    """A block of retrieved context ready for injection into the prompt."""
    key: str
    value: str
    score: float
    token_cost: int
    source: str = "scored"  # "scored", "anchored", or "dependency"


@dataclass
class AssembledContext:
    """The output of a context assembly pass."""
    pinned_messages: list[dict]       # recent turns (always included)
    managed_blocks: list[ManagedBlock]  # from B-tree, ranked by score
    total_tokens: int
    budget_remaining: int
    tuples_considered: int
    tuples_included: int
    anchored_count: int = 0           # tuples force-included by anchoring
    dependency_count: int = 0         # tuples pulled in by call graph


class ContextAssembler:
    """Assembles a prompt from pinned messages + B-tree managed context."""

    def __init__(
        self,
        tree: SemanticBTree,
        config: AssemblerConfig | None = None,
    ) -> None:
        self.tree = tree
        self.cfg = config or AssemblerConfig()

    def assemble(
        self,
        conversation: list[dict],
        query_emb: Embedding | None = None,
    ) -> AssembledContext:
        """Build a context-managed prompt.

        Parameters
        ----------
        conversation : list[dict]
            Full conversation history (role/content dicts).
        query_emb : Embedding, optional
            Embedding of the current user query for relevance scoring.

        Returns
        -------
        AssembledContext with pinned messages + managed blocks.
        """
        # -- 1. Pin recent turns --
        pinned = conversation[-self.cfg.recency_window * 2 :]
        pinned_tokens = sum(
            estimate_tokens(self._message_text(m)) for m in pinned
        )

        # -- 2. Compute remaining budget for managed context --
        managed_budget = (
            self.cfg.total_budget - self.cfg.pinned_reserve - pinned_tokens
        )
        managed_budget = max(0, managed_budget)

        # Apply budget pressure: intentionally use less than the full
        # budget. Shorter context means each token gets more attention
        # from the model, reducing "lost in the middle" effects.
        managed_budget = int(managed_budget * self.cfg.budget_pressure)

        # -- 3. Query tree for top-k tuples --
        if query_emb is not None and self.tree.size > 0:
            scored = self.tree.query(query_emb, top_k=self.cfg.managed_top_k)
        else:
            scored = []

        # -- 4. Ground-truth anchoring --
        # Force-include tuples that are highly relevant to the query,
        # regardless of their composite score. This prevents hallucination
        # by ensuring stored facts are in context when the query is about them.
        anchor_budget = int(managed_budget * self.cfg.anchor_budget_fraction)
        anchored_blocks: list[ManagedBlock] = []
        anchored_ids: set[str] = set()
        included_keys: set[str] = set()

        if query_emb is not None:
            all_tuples = self.tree.all_tuples()
            for t in all_tuples:
                if t.key_emb is None:
                    continue
                rel = cosine_sim(query_emb, t.key_emb)
                if rel >= self.cfg.anchor_relevance_threshold:
                    if t.token_cost <= anchor_budget and t.key_text not in included_keys:
                        anchored_blocks.append(ManagedBlock(
                            key=t.key_text,
                            value=t.value_text,
                            score=rel,
                            token_cost=t.token_cost,
                            source="anchored",
                        ))
                        anchor_budget -= t.token_cost
                        anchored_ids.add(t.id)
                        included_keys.add(t.key_text)
                        t.touch()

        # -- 5. Greedy knapsack: fill budget by score --
        scored_budget = managed_budget - sum(b.token_cost for b in anchored_blocks)
        blocks: list[ManagedBlock] = list(anchored_blocks)
        scored_ids: set[str] = set(anchored_ids)
        used = sum(b.token_cost for b in blocks)

        for score, t in scored:
            if t.id in scored_ids:
                continue
            if used + t.token_cost > managed_budget:
                continue
            blocks.append(
                ManagedBlock(
                    key=t.key_text,
                    value=t.value_text,
                    score=score,
                    token_cost=t.token_cost,
                    source="scored",
                )
            )
            scored_ids.add(t.id)
            included_keys.add(t.key_text)
            used += t.token_cost

        # -- 6. Dependency pull --
        # If included tuples have structural references (call graph),
        # pull in their dependencies too.
        dep_count = 0
        if self.cfg.dependency_pull:
            dep_budget = int(managed_budget * self.cfg.dependency_budget_fraction)
            dep_ids = self._collect_dependency_ids(scored_ids)

            for t in self.tree.all_tuples():
                if t.id in dep_ids and t.id not in scored_ids:
                    if t.token_cost <= dep_budget and t.key_text not in included_keys:
                        blocks.append(ManagedBlock(
                            key=t.key_text,
                            value=t.value_text,
                            score=0.0,
                            token_cost=t.token_cost,
                            source="dependency",
                        ))
                        dep_budget -= t.token_cost
                        used += t.token_cost
                        scored_ids.add(t.id)
                        included_keys.add(t.key_text)
                        dep_count += 1
                        t.touch()

        # -- 7. Update scorer's active context for affinity computation --
        self.tree.scorer.set_active_context(scored_ids)

        return AssembledContext(
            pinned_messages=pinned,
            managed_blocks=blocks,
            total_tokens=pinned_tokens + used,
            budget_remaining=managed_budget - used,
            tuples_considered=len(scored),
            tuples_included=len(blocks),
            anchored_count=len(anchored_blocks),
            dependency_count=dep_count,
        )

    def _collect_dependency_ids(self, included_ids: set[str]) -> set[str]:
        """Collect IDs of tuples that are structurally related to
        the included set (one hop in the call graph)."""
        dep_ids: set[str] = set()
        for t in self.tree.all_tuples():
            if t.id in included_ids:
                dep_ids.update(t.references)
                dep_ids.update(t.referenced_by)
        return dep_ids - included_ids

    def to_messages(
        self,
        system_prompt: str,
        assembled: AssembledContext,
    ) -> list[dict]:
        """Convert an AssembledContext into Anthropic-style messages.

        Uses position-aware layout to fight attention degradation:

        Models attend best to the START and END of the context window
        and worst to the MIDDLE ("lost in the middle" problem). So we
        arrange the managed context in a U-shaped attention pattern:

            [SYSTEM PROMPT]                         ← always attended
            [HIGH-PRIORITY: anchored facts]         ← start = strong
            [MEDIUM-PRIORITY: scored context]        ← middle = weaker
            [LOW-PRIORITY: dependencies]             ← middle = weaker
            ... pinned recent turns ...
            [CRITICAL RECAP: top facts restated]     ← end = strong
            [USER'S LATEST MESSAGE]                  ← always attended

        The critical recap before the user's message ensures that the
        most important facts are in the attention "hot zone" at the
        end of the sequence, even if they were originally from turn 2.
        """
        messages: list[dict] = []

        if assembled.managed_blocks:
            # Partition blocks by source for positional placement
            anchored = [b for b in assembled.managed_blocks if b.source == "anchored"]
            scored = [b for b in assembled.managed_blocks if b.source == "scored"]
            deps = [b for b in assembled.managed_blocks if b.source == "dependency"]

            # ── Position 1: High-priority context (START of window) ──
            # Anchored facts go first — they are directly relevant to
            # the current query and placing them at the start ensures
            # strong attention.
            ctx_parts = []
            if anchored:
                ctx_parts.append("## Key facts relevant to this query")
                for b in anchored:
                    ctx_parts.append(f"- [{b.key}]: {b.value}")

            # ── Position 2: Scored context (MIDDLE of window) ──
            # General context ranked by score. Gets weaker attention
            # but provides background knowledge.
            if scored:
                ctx_parts.append("\n## Session context")
                for b in scored:
                    ctx_parts.append(f"[{b.key}]\n{b.value}")

            # ── Position 3: Dependencies (MIDDLE of window) ──
            if deps:
                ctx_parts.append("\n## Related context")
                for b in deps:
                    ctx_parts.append(f"[{b.key}]\n{b.value}")

            context_block = (
                "<retrieved_context>\n"
                + "\n\n".join(ctx_parts)
                + "\n</retrieved_context>"
            )

            messages.append({
                "role": "user",
                "content": (
                    "[SYSTEM: The following is retrieved context relevant "
                    "to this conversation. Use it if helpful.]\n\n"
                    + context_block
                ),
            })
            messages.append({
                "role": "assistant",
                "content": "Understood. I have the retrieved context available.",
            })

        # ── Position 4: Pinned recent turns ──
        # These go after managed context but before the recap.
        # Separate the last user message so we can inject recap before it.
        if assembled.pinned_messages:
            all_pinned = assembled.pinned_messages
            # Everything except the last message
            preceding = all_pinned[:-1]
            last_msg = all_pinned[-1]

            messages.extend(preceding)

            # ── Position 5: Critical recap (END of window, just before user query) ──
            # Restate the top 3 most important facts in condensed form.
            # This exploits the recency bias in attention — facts placed
            # right before the query get disproportionately strong attention.
            anchored_blocks = [b for b in assembled.managed_blocks if b.source == "anchored"]
            top_scored = sorted(
                [b for b in assembled.managed_blocks if b.source == "scored"],
                key=lambda b: b.score,
                reverse=True,
            )[:3]
            recap_items = anchored_blocks + top_scored

            if recap_items and len(assembled.managed_blocks) > 5:
                # Only add recap when there's enough context that
                # middle items might get lost
                recap_lines = [f"- {b.key}: {b.value[:100]}" for b in recap_items[:5]]
                recap_text = (
                    "[Key context reminder]\n"
                    + "\n".join(recap_lines)
                )

                # Inject as the assistant's last thought before the user speaks
                if preceding and preceding[-1].get("role") == "assistant":
                    # Append to the last assistant message
                    pass  # Don't modify existing messages
                else:
                    # Add as a new exchange
                    messages.append({
                        "role": "user",
                        "content": recap_text,
                    })
                    messages.append({
                        "role": "assistant",
                        "content": "Noted, I have these key points in mind.",
                    })

            # ── Position 6: The actual user message (VERY END = maximum attention) ──
            messages.append(last_msg)
        else:
            # No pinned messages — just the managed context
            pass

        return messages

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Normalize Anthropic-style message content into plain text."""
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)
