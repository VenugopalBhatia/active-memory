"""Middleware that wraps the Anthropic Messages API with active memory.

Usage
-----
```python
from anthropic import Anthropic
from active_memory import ActiveMemoryMiddleware

client = Anthropic()
mw = ActiveMemoryMiddleware(client, embedder=my_embedder)

# Use it like the normal client — memory is managed automatically
response = mw.send("What database should I use for time-series data?")
print(response.text)

# Later in the conversation it remembers and prunes automatically
response = mw.send("Remind me what we decided about the DB")
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .assembler import AssemblerConfig, ContextAssembler
from .btree import BTreeConfig, SemanticBTree, _Summariser
from .grounding import (
    GroundedAssembler, GroundingConfig, GroundingReport, VerificationResult
)
from .scoring import Scorer, ScoringConfig
from .model_clients import ModelClient
from .types import Embedder, KVTuple, estimate_tokens


@dataclass
class MiddlewareConfig:
    system_prompt: str = "You are a helpful assistant."
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 4096

    # How often (in turns) to run pruning/compression
    prune_interval: int = 5
    compress_interval: int = 10

    # Whether to auto-ingest assistant responses as KV tuples
    auto_ingest_responses: bool = True

    # Grounding layer
    grounding: GroundingConfig = field(default_factory=GroundingConfig)

    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    btree: BTreeConfig = field(default_factory=BTreeConfig)
    assembler: AssemblerConfig = field(default_factory=AssemblerConfig)


class ActiveMemoryMiddleware:
    """Drop-in middleware that manages context via a semantic B-tree."""

    def __init__(
        self,
        client: ModelClient,
        embedder: Embedder,
        config: MiddlewareConfig | None = None,
        summariser: _Summariser | None = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.cfg = config or MiddlewareConfig()
        self.summariser = summariser

        self.scorer = Scorer(self.cfg.scoring)
        self.tree = SemanticBTree(
            embedder=embedder,
            scorer=self.scorer,
            config=self.cfg.btree,
        )
        self.assembler = ContextAssembler(
            tree=self.tree,
            config=self.cfg.assembler,
        )

        # Grounding layer
        self.grounder = GroundedAssembler(
            tree=self.tree,
            embedder=embedder,
            scorer=self.scorer,
            config=self.cfg.grounding,
        ) if self.cfg.grounding.enabled else None

        self._conversation: list[dict] = []
        self._turn_count: int = 0

    # -- public API -----------------------------------------------------

    def send(self, user_message: str, **api_kwargs: Any) -> MiddlewareResponse:
        """Send a message through the managed pipeline.

        1. Ingest the user message as KV tuples
        2. Assemble context from tree + recent turns
        3. Call Anthropic API
        4. Ingest the response
        5. Periodically prune / compress
        """
        # -- ingest user message --
        self._conversation.append({"role": "user", "content": user_message})
        self._ingest_message("user", user_message)
        self._turn_count += 1

        # -- assemble context --
        query_emb = self.embedder.embed([user_message])[0]

        grounding_report = None
        verification = None
        tuples_considered = 0
        tuples_included = 0

        if self.grounder and self.cfg.grounding.provenance_injection:
            # Use grounded assembler with confidence tags
            system_prompt, messages = self.grounder.build_grounded_prompt(
                self.cfg.system_prompt,
                self._conversation,
                query_emb,
                token_budget=self.cfg.assembler.total_budget,
                recency_window=self.cfg.assembler.recency_window,
            )
            assembled_tokens = sum(
                estimate_tokens(self._message_text(m)) for m in messages
            )
            grounded_blocks = getattr(self.grounder, "last_blocks", [])
            tuples_considered = len(grounded_blocks)
            tuples_included = len(grounded_blocks)
        else:
            # Standard assembler
            system_prompt = self.cfg.system_prompt
            assembled = self.assembler.assemble(self._conversation, query_emb)
            messages = self.assembler.to_messages(
                self.cfg.system_prompt, assembled
            )
            assembled_tokens = assembled.total_tokens
            tuples_considered = assembled.tuples_considered
            tuples_included = assembled.tuples_included

        # -- call Anthropic --
        response = self.client.generate(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system_prompt,
            messages=messages,
            **api_kwargs,
        )
        assistant_text = response.text

        # -- post-generation verification --
        if (
            self.grounder
            and self.cfg.grounding.post_verification
            and self.tree.size > 5  # only verify once we have enough context
        ):
            verification = self.grounder.verify_response(assistant_text)

            # Auto-correct if enabled and issues found
            if (
                self.cfg.grounding.auto_correct
                and verification.overall_grounding < self.cfg.grounding.min_grounding_rate
            ):
                correction = self.grounder.build_correction_prompt(verification)
                if correction:
                    # Send correction as a follow-up
                    messages.append({"role": "assistant", "content": assistant_text})
                    messages.append({"role": "user", "content": correction})
                    corrected = self.client.generate(
                        model=self.cfg.model,
                        max_tokens=self.cfg.max_tokens,
                        system=system_prompt,
                        messages=messages,
                        **api_kwargs,
                    )
                    assistant_text = corrected.text
                    response = corrected
                    verification = self.grounder.verify_response(assistant_text)

        # -- ingest response --
        self._conversation.append(
            {"role": "assistant", "content": assistant_text}
        )
        if self.cfg.auto_ingest_responses:
            self._ingest_message("assistant", assistant_text)

        # -- maintenance --
        if self.cfg.prune_interval > 0 and self._turn_count % self.cfg.prune_interval == 0:
            self.tree.prune(query_emb)
        if self.cfg.compress_interval > 0 and self._turn_count % self.cfg.compress_interval == 0:
            self.tree.compress_cold_subtrees(
                summariser=self.summariser, query_emb=query_emb
            )

        return MiddlewareResponse(
            text=assistant_text,
            raw=response,
            context_stats=ContextStats(
                tree_size=self.tree.size,
                tree_depth=self.tree.depth(),
                tokens_assembled=assembled_tokens,
                tokens_remaining=self.cfg.assembler.total_budget - assembled_tokens,
                tuples_considered=tuples_considered,
                tuples_included=tuples_included,
                turn=self._turn_count,
            ),
            verification=verification,
        )

    def ingest(self, key: str, value: str) -> KVTuple:
        """Manually inject a KV tuple (e.g. tool output, document chunk)."""
        return self.tree.insert(key, value)

    def ingest_document(
        self, text: str, chunk_size: int = 500
    ) -> list[KVTuple]:
        """Chunk and ingest a document."""
        chunks = self._chunk_text(text, chunk_size)
        tuples = []
        for i, chunk in enumerate(chunks):
            key = f"doc_chunk_{i}"
            tuples.append(self.tree.insert(key, chunk))
        return tuples

    def ingest_code_file(self, filepath: str) -> list[KVTuple]:
        """Ingest a code file using AST-aware chunking.

        Uses AST parsing to split at function/class boundaries instead
        of sentence boundaries. Extracts call graph relationships and
        stores them as structural references between tuples, enabling
        dependency-aware context assembly.

        Returns the list of created KV tuples.
        """
        from .code_ingest import parse_code_file
        from pathlib import Path

        chunks = parse_code_file(filepath)
        if not chunks:
            return []

        filename = Path(filepath).name
        tuples: list[KVTuple] = []
        name_to_tuple: dict[str, KVTuple] = {}

        # Phase 1: Insert all chunks
        for chunk in chunks:
            key = f"code:{filename}:{chunk.name}"
            t = self.tree.insert(key, chunk.source)
            t.tags.append("code")
            t.tags.append(chunk.kind)
            tuples.append(t)
            name_to_tuple[chunk.name] = t

            # Also register by short name for method matching
            if "." in chunk.name:
                short = chunk.name.split(".")[-1]
                if short not in name_to_tuple:
                    name_to_tuple[short] = t

        # Phase 2: Wire up structural references from call graph
        for chunk in chunks:
            caller = name_to_tuple.get(chunk.name)
            if not caller:
                continue
            for call_name in chunk.calls:
                callee = name_to_tuple.get(call_name)
                if callee and callee.id != caller.id:
                    if callee.id not in caller.references:
                        caller.references.append(callee.id)
                    if caller.id not in callee.referenced_by:
                        callee.referenced_by.append(caller.id)

        return tuples

    def ingest_code_directory(
        self, dirpath: str, extensions: set[str] | None = None
    ) -> list[KVTuple]:
        """Ingest all code files in a directory tree.

        Parameters
        ----------
        dirpath : str
            Root directory to scan.
        extensions : set[str], optional
            File extensions to include. Defaults to common code extensions.

        Returns
        -------
        list[KVTuple] of all created tuples.
        """
        from pathlib import Path

        if extensions is None:
            extensions = {
                ".py", ".js", ".ts", ".jsx", ".tsx",
                ".java", ".go", ".rs", ".c", ".cpp", ".h",
            }

        root = Path(dirpath)
        all_tuples: list[KVTuple] = []
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix in extensions:
                # Skip common non-source directories
                parts = path.parts
                if any(p in {
                    "__pycache__", "node_modules", ".git",
                    "venv", ".venv", "dist", "build"
                } for p in parts):
                    continue
                all_tuples.extend(self.ingest_code_file(str(path)))

        return all_tuples

    @property
    def stats(self) -> dict:
        return {
            "tree_size": self.tree.size,
            "tree_depth": self.tree.depth(),
            "conversation_turns": self._turn_count,
            "total_nodes": len(self.tree.all_nodes()),
        }

    # -- internals ------------------------------------------------------

    def _ingest_message(self, role: str, content: str) -> None:
        """Break a message into KV tuples and insert into the tree."""
        # Split into sentences / logical units
        segments = self._segment(content)
        for seg in segments:
            if len(seg.strip()) < 10:
                continue
            key = f"{role}:{seg[:60]}"
            self.tree.insert(key, seg)

    @staticmethod
    def _segment(text: str) -> list[str]:
        """Split text into sentence-ish segments."""
        import re
        parts = re.split(r'(?<=[.!?])\s+', text)
        # Merge very short segments
        merged: list[str] = []
        buf = ""
        for p in parts:
            buf += (" " if buf else "") + p
            if len(buf) >= 80:
                merged.append(buf)
                buf = ""
        if buf:
            merged.append(buf)
        return merged

    @staticmethod
    def _chunk_text(text: str, size: int) -> list[str]:
        words = text.split()
        chunks: list[str] = []
        for i in range(0, len(words), size):
            chunks.append(" ".join(words[i : i + size]))
        return chunks

    @staticmethod
    def _message_text(message: dict[str, Any]) -> str:
        """Normalize Anthropic-style message content into plain text.

        Handles text, tool_use, and tool_result blocks.
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


# -- response types ---------------------------------------------------------

@dataclass
class ContextStats:
    tree_size: int
    tree_depth: int
    tokens_assembled: int
    tokens_remaining: int
    tuples_considered: int
    tuples_included: int
    turn: int


@dataclass
class MiddlewareResponse:
    text: str
    raw: Any  # anthropic MessageResponse
    context_stats: ContextStats
    verification: VerificationResult | None = None

    @property
    def grounding_rate(self) -> float | None:
        """How well-grounded is this response? None if verification disabled."""
        return self.verification.overall_grounding if self.verification else None

    @property
    def has_contradictions(self) -> bool:
        """Did the response contradict stored facts?"""
        return bool(self.verification and self.verification.contradictions)
