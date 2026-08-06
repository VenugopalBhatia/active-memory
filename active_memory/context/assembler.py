"""Deterministic recent-turn and memory packing under an input budget."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from active_memory.context.budgeting import BudgetConfig, ContextBudgetExceeded
from active_memory.context.formatter import content_to_text, format_memory, format_memory_context, message_to_text
from active_memory.context.tokenizer import TokenCounter
from active_memory.models import RetrievalResult
from active_memory.storage.base import MemoryStore


@dataclass(slots=True)
class AssemblyResult:
    system: str
    messages: list[dict[str, Any]]
    memory_context: str
    included: list[RetrievalResult]
    excluded: dict[str, str]
    input_tokens: int
    available_input_tokens: int
    component_tokens: dict[str, int] = field(default_factory=dict)


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


class ContextAssembler:
    def __init__(self, store: MemoryStore, token_counter: TokenCounter, config: BudgetConfig | None = None) -> None:
        self.store = store
        self.token_counter = token_counter
        self.config = config or BudgetConfig()

    def assemble(self, *, session_id: str, namespace: str, system: Any, tools: Any, messages: Sequence[dict[str, Any]], ranked_memories: Sequence[RetrievalResult], assembled_at: datetime) -> AssemblyResult:
        latest_index = next((index for index in range(len(messages) - 1, -1, -1) if messages[index].get("role") == "user"), None)
        if latest_index is None:
            raise ValueError("at least one user message is required")
        latest = dict(messages[latest_index])
        system_text = content_to_text(system)
        tool_text = json.dumps(tools, sort_keys=True, default=str) if tools else ""
        fixed_tokens = self._text_tokens(system_text) + self._text_tokens(tool_text) + self._message_tokens(latest)
        available = self.config.available_input_tokens
        if fixed_tokens > available:
            raise ContextBudgetExceeded(f"pinned system/tools/latest message require {fixed_tokens} tokens; input budget is {available}")

        recent_cap = min(int(available * self.config.recent_turn_fraction), available - fixed_tokens)
        recent_candidates = [dict(message) for index, message in enumerate(messages) if index != latest_index]
        selected_recent: list[dict[str, Any]] = []
        recent_tokens = 0
        for message in reversed(recent_candidates[-self.config.recent_message_limit :]):
            cost = self._message_tokens(message)
            if recent_tokens + cost <= recent_cap:
                selected_recent.append(message)
                recent_tokens += cost
        selected_recent.reverse()

        remaining = available - fixed_tokens - recent_tokens
        memory_cap = min(int(available * self.config.memory_fraction), remaining)
        packed, excluded, memory_context, memory_tokens = self._pack_memories(ranked_memories, selected_recent + [latest], memory_cap)
        combined_system = system_text
        if memory_context:
            combined_system = f"{system_text}\n\n{memory_context}" if system_text else memory_context
        final_messages = selected_recent + [latest]
        input_tokens = self._text_tokens(combined_system) + self._text_tokens(tool_text) + sum(self._message_tokens(message) for message in final_messages)
        if input_tokens > available:
            raise ContextBudgetExceeded(f"assembled input uses {input_tokens} tokens; budget is {available}")

        result = AssemblyResult(
            combined_system, final_messages, memory_context, packed, excluded, input_tokens, available,
            {"system_and_memory": self._text_tokens(combined_system), "tools": self._text_tokens(tool_text),
             "latest_user": self._message_tokens(latest), "recent": recent_tokens, "memory": memory_tokens},
        )
        memory_ids = [item.memory.id for item in packed]
        complete = getattr(self.store, "complete_context_assembly", None)
        if complete:
            complete(session_id, namespace, memory_ids, input_tokens, assembled_at, {"excluded": excluded})
        else:
            self.store.mark_included(memory_ids, assembled_at)
        return result

    def _pack_memories(self, ranked: Sequence[RetrievalResult], pinned_messages: Sequence[dict[str, Any]], budget: int) -> tuple[list[RetrievalResult], dict[str, str], str, int]:
        pinned_text = _normalized("\n".join(message_to_text(message) for message in pinned_messages))
        excluded: dict[str, str] = {}
        unique: list[RetrievalResult] = []
        seen: set[str] = set()
        trust = {"user_confirmed": 4, "tool_observed": 3, "assistant_generated": 2, "external_untrusted": 1}
        conflict_best: dict[str, RetrievalResult] = {}
        for result in ranked:
            memory = result.memory
            normalized = _normalized(memory.content)
            if memory.status != "active" or memory.superseded_by:
                excluded[memory.id] = "inactive or superseded"
                continue
            if not normalized or normalized in seen:
                excluded[memory.id] = "duplicate memory"
                continue
            if normalized in pinned_text:
                excluded[memory.id] = "already represented in recent turns"
                continue
            conflict_key = str(memory.metadata.get("conflict_key", ""))
            if conflict_key:
                current = conflict_best.get(conflict_key)
                if current and trust[current.memory.trust_level] >= trust[memory.trust_level]:
                    excluded[memory.id] = "higher-trust conflicting memory selected"
                    continue
                if current:
                    excluded[current.memory.id] = "higher-trust conflicting memory selected"
                    unique.remove(current)
                conflict_best[conflict_key] = result
            seen.add(normalized)
            unique.append(result)

        groups: dict[str, list[RetrievalResult]] = {}
        for result in unique:
            group = str(result.memory.metadata.get("dependency_group") or result.memory.id)
            groups.setdefault(group, []).append(result)
        scored_groups: list[tuple[float, str, list[RetrievalResult], list[str], int]] = []
        for group_id, items in groups.items():
            formatted = [format_memory(item) for item in sorted(items, key=lambda item: item.memory.id)]
            context = format_memory_context(formatted)
            cost = self._text_tokens(context)
            utility = sum(item.final_score for item in items) / math.sqrt(max(1, cost))
            scored_groups.append((utility, group_id, items, formatted, cost))
        scored_groups.sort(key=lambda item: (-item[0], item[1]))

        selected: list[RetrievalResult] = []
        formatted_selected: list[str] = []
        for _, _, items, formatted, _ in scored_groups:
            tentative = format_memory_context(formatted_selected + formatted)
            cost = self._text_tokens(tentative)
            if cost <= budget:
                selected.extend(items)
                formatted_selected.extend(formatted)
            else:
                for item in items:
                    excluded[item.memory.id] = "memory budget exhausted"
        context = format_memory_context(formatted_selected)
        return selected, excluded, context, self._text_tokens(context)

    def _text_tokens(self, text: str) -> int:
        return self.token_counter.count(text)

    def _message_tokens(self, message: dict[str, Any]) -> int:
        return self.token_counter.count(message_to_text(message)) + 4

