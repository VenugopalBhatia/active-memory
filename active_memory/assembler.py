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


# Wrapper overhead estimates used to reserve budget *before* knapsacking.
# These cover text that ``to_messages`` adds beyond the raw block content:
#   - the SYSTEM hint preface + ack message exchange
#   - the optional "key context reminder" recap exchange
#   - per-block formatting (``[key]\n value`` lines)
# Reserving up front prevents the assembled prompt from exceeding
# total_budget after wrappers are added.
_WRAPPER_FIXED_OVERHEAD = 80      # SYSTEM hint + ack + section headers
_WRAPPER_RECAP_OVERHEAD = 60      # recap user/assistant exchange
_WRAPPER_PER_BLOCK_OVERHEAD = 8   # formatting tokens per included block


@dataclass
class AssemblerConfig:
    total_budget: int = 18_000        # max tokens for the full prompt
    pinned_reserve: int = 4_000       # tokens reserved for pinned content
    recency_window: int = 3           # last N conversation turns always kept
    managed_top_k: int = 120          # max tuples to score from the tree

    # Ground-truth anchoring: force-include tuples with relevance above
    # this threshold, even if their composite score is low.
    anchor_relevance_threshold: float = 0.62
    # Max tokens to spend on anchored tuples (prevents runaway anchoring)
    anchor_budget_fraction: float = 0.20

    # Dependency pulling: include structural neighbors of selected tuples
    dependency_pull: bool = True
    # Max tokens to spend on pulled dependencies
    dependency_budget_fraction: float = 0.08

    # Budget pressure: intentionally use LESS than the full budget.
    # A value of 0.6 means "fill only 60% of the available budget."
    # Lower values = shorter context = stronger per-token attention,
    # but higher risk of missing relevant information.
    # Range: 0.3 (very aggressive) to 1.0 (fill everything).
    budget_pressure: float = 0.80


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
    wrapper_tokens: int = 0           # tokens added by synthetic wrapper messages
    system_tokens: int = 0            # tokens consumed by the system prompt


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
        system: Any = None,
    ) -> AssembledContext:
        """Build a context-managed prompt.

        Parameters
        ----------
        conversation : list[dict]
            Full conversation history (role/content dicts).
        query_emb : Embedding, optional
            Embedding of the current user query for relevance scoring.
        system : str | list | None, optional
            The system prompt that will be forwarded to the model.
            Anthropic accepts ``system`` as a top-level field rather
            than a message, but its tokens still count toward the
            context window — so we deduct them from the available
            budget here to avoid overflowing the upstream request.

        Returns
        -------
        AssembledContext with pinned messages + managed blocks.
        """
        # -- 1. Pin recent turns --
        if self.cfg.recency_window > 0:
            pinned = conversation[-self.cfg.recency_window * 2 :]
        else:
            pinned = []
        pinned_tokens = sum(
            estimate_tokens(self._message_text(m)) for m in pinned
        )

        # System prompt tokens are forwarded as a top-level field but
        # still consume context-window budget.
        system_tokens = estimate_tokens(self._system_text(system))

        # -- 2. Compute remaining budget for managed context --
        # Reserve wrapper overhead (SYSTEM hint, ack, recap) up front
        # so the final assembled prompt cannot exceed total_budget once
        # to_messages() finishes adding wrapper text.
        managed_budget = (
            self.cfg.total_budget
            - self.cfg.pinned_reserve
            - pinned_tokens
            - system_tokens
            - _WRAPPER_FIXED_OVERHEAD
            - _WRAPPER_RECAP_OVERHEAD
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
        all_tuples = self.tree.all_tuples() if query_emb is not None else []

        if query_emb is not None:
            for t in all_tuples:
                if t.key_emb is None:
                    continue
                rel = cosine_sim(query_emb, t.key_emb)
                if rel >= self.cfg.anchor_relevance_threshold:
                    effective_cost = t.token_cost + _WRAPPER_PER_BLOCK_OVERHEAD
                    if effective_cost <= anchor_budget and t.key_text not in included_keys:
                        anchored_blocks.append(ManagedBlock(
                            key=t.key_text,
                            value=t.value_text,
                            score=rel,
                            token_cost=t.token_cost,
                            source="anchored",
                        ))
                        anchor_budget -= effective_cost
                        anchored_ids.add(t.id)
                        included_keys.add(t.key_text)
                        t.touch()

        # -- 5. Greedy knapsack: fill budget by score --
        # Each block carries a small per-block formatting overhead
        # (the ``[key]\n value`` lines wrap around its raw content).
        # Charge it to the budget so the final wrapped prompt fits.
        blocks: list[ManagedBlock] = list(anchored_blocks)
        scored_ids: set[str] = set(anchored_ids)
        used = sum(b.token_cost + _WRAPPER_PER_BLOCK_OVERHEAD for b in blocks)

        for score, t in scored:
            if t.id in scored_ids:
                continue
            effective_cost = t.token_cost + _WRAPPER_PER_BLOCK_OVERHEAD
            if used + effective_cost > managed_budget:
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
            used += effective_cost

        # -- 6. Dependency pull --
        # If included tuples have structural references (call graph),
        # pull in their dependencies too.
        dep_count = 0
        if self.cfg.dependency_pull:
            dep_budget = int(managed_budget * self.cfg.dependency_budget_fraction)
            dep_ids = self._collect_dependency_ids(scored_ids, tuples=all_tuples)

            for t in all_tuples:
                if t.id in dep_ids and t.id not in scored_ids:
                    effective_cost = t.token_cost + _WRAPPER_PER_BLOCK_OVERHEAD
                    if (
                        effective_cost <= dep_budget
                        and used + effective_cost <= managed_budget
                        and t.key_text not in included_keys
                    ):
                        blocks.append(ManagedBlock(
                            key=t.key_text,
                            value=t.value_text,
                            score=0.0,
                            token_cost=t.token_cost,
                            source="dependency",
                        ))
                        dep_budget -= effective_cost
                        used += effective_cost
                        scored_ids.add(t.id)
                        included_keys.add(t.key_text)
                        dep_count += 1
                        t.touch()

        # -- 7. Update scorer's active context for affinity computation --
        self.tree.scorer.set_active_context(scored_ids)

        # ``total_tokens`` here is provisional; ``to_messages`` will
        # overwrite it with an authoritative count once the actual
        # message list is built.
        raw_block_tokens = sum(b.token_cost for b in blocks)
        return AssembledContext(
            pinned_messages=pinned,
            managed_blocks=blocks,
            total_tokens=pinned_tokens + system_tokens + raw_block_tokens,
            budget_remaining=managed_budget - used,
            tuples_considered=len(scored),
            tuples_included=len(blocks),
            anchored_count=len(anchored_blocks),
            dependency_count=dep_count,
            system_tokens=system_tokens,
        )

    def _collect_dependency_ids(
        self,
        included_ids: set[str],
        tuples: list[KVTuple] | None = None,
    ) -> set[str]:
        """Collect IDs of tuples that are structurally related to
        the included set (one hop in the call graph)."""
        dep_ids: set[str] = set()
        source = tuples if tuples is not None else self.tree.all_tuples()
        for t in source:
            if t.id in included_ids:
                dep_ids.update(t.references)
                dep_ids.update(t.referenced_by)
        return dep_ids - included_ids

    def to_messages(
        self,
        system_prompt: str | list | None,
        assembled: AssembledContext,
    ) -> list[dict]:
        """Convert an AssembledContext into Anthropic-style messages.

        Note: *system_prompt* is accepted for token accounting but is NOT
        injected into the returned message list because the Anthropic API
        takes ``system`` as a separate top-level field.  The caller is
        responsible for forwarding it in ``request_data["system"]``.

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

        This method is idempotent: calling it more than once on the
        same ``AssembledContext`` will not double-count wrapper tokens.
        Callers may modify ``assembled.managed_blocks`` between calls
        (e.g. to drop blocks for budget pressure) and re-invoke safely.
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
            ack_text = "Understood. I have the retrieved context available."
            messages.append({
                "role": "assistant",
                "content": ack_text,
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
                    recap_ack = "Noted, I have these key points in mind."
                    messages.append({
                        "role": "user",
                        "content": recap_text,
                    })
                    messages.append({
                        "role": "assistant",
                        "content": recap_ack,
                    })

            # ── Position 6: The actual user message (VERY END = maximum attention) ──
            messages.append(last_msg)
        else:
            # No pinned messages — just the managed context
            pass

        # -- Authoritative token accounting --
        # Compute total_tokens from the actual built message list plus
        # the system prompt (which is forwarded as a top-level field
        # but still counts toward the model's context window). This
        # avoids drift between estimates and what is actually sent.
        system_tokens = estimate_tokens(self._system_text(system_prompt))
        message_tokens = sum(
            estimate_tokens(self._message_text(m)) for m in messages
        )
        raw_block_tokens = sum(b.token_cost for b in assembled.managed_blocks)
        pinned_tokens = sum(
            estimate_tokens(self._message_text(m))
            for m in assembled.pinned_messages
        )
        # Wrapper overhead = everything in the message list that is not
        # pinned content and not raw block content.
        assembled.system_tokens = system_tokens
        assembled.wrapper_tokens = max(
            0, message_tokens - pinned_tokens - raw_block_tokens
        )
        assembled.total_tokens = message_tokens + system_tokens
        assembled.budget_remaining = self.cfg.total_budget - assembled.total_tokens

        return messages

    @staticmethod
    def _system_text(system: Any) -> str:
        """Normalize an Anthropic ``system`` payload into plain text.

        ``system`` may be ``None``, a plain string, or a list of text
        blocks (``[{"type": "text", "text": "..."}]``).
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

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Normalize Anthropic-style message content into plain text.

        Handles text, tool_use, and tool_result blocks so that token
        estimates cover the full content, not just text blocks.
        """
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        parts.append(str(item.get("text", "")))
                    elif item_type == "tool_use":
                        name = item.get("name", "")
                        inp = item.get("input", {})
                        parts.append(f"[tool_use:{name}] {inp}")
                    elif item_type == "tool_result":
                        rc = item.get("content", "")
                        if isinstance(rc, str):
                            parts.append(rc)
                        elif isinstance(rc, list):
                            for sub in rc:
                                if isinstance(sub, str):
                                    parts.append(sub)
                                elif isinstance(sub, dict) and sub.get("type") == "text":
                                    parts.append(str(sub.get("text", "")))
            return "\n".join(part for part in parts if part)
        return str(content)
