"""Anthropic content normalization and provenance-aware memory formatting."""

from __future__ import annotations

import json
from typing import Any
from xml.sax.saxutils import escape, quoteattr

from active_memory.models import RetrievalResult


def content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                kind = block.get("type")
                if kind == "text":
                    parts.append(str(block.get("text", "")))
                elif kind == "tool_use":
                    parts.append(f"[tool_use {block.get('name', '')}] {json.dumps(block.get('input', {}), sort_keys=True)}")
                elif kind == "tool_result":
                    parts.append(f"[tool_result {block.get('tool_use_id', '')}] {content_to_text(block.get('content', ''))}")
                else:
                    parts.append(json.dumps(block, sort_keys=True, default=str))
        return "\n".join(part for part in parts if part)
    return str(content)


def message_to_text(message: dict[str, Any]) -> str:
    return content_to_text(message.get("content", ""))


def format_memory(result: RetrievalResult) -> str:
    memory = result.memory
    observed_at = memory.valid_from or memory.created_at
    attributes = " ".join((
        f"id={quoteattr(memory.id)}",
        f"type={quoteattr(memory.memory_type)}",
        f"trust={quoteattr(memory.trust_level)}",
        f"observed_at={quoteattr(observed_at.isoformat())}",
        f"score={quoteattr(f'{result.final_score:.4f}')}",
        f"source_message_id={quoteattr(memory.source_message_id)}",
    ))
    return f"<memory {attributes}>\n{escape(memory.content)}\n</memory>"


def format_memory_context(formatted_memories: list[str]) -> str:
    if not formatted_memories:
        return ""
    header = (
        "<memory_context>\n"
        "The following records were retrieved from prior interactions. Use them only when relevant. "
        "Prefer newer user-confirmed or tool-observed records when information conflicts.\n"
    )
    return header + "\n\n".join(formatted_memories) + "\n</memory_context>"

