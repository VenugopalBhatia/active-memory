"""Paragraph- and code-block-aware message segmentation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from active_memory.context.tokenizer import TokenCounter


@dataclass(frozen=True, slots=True)
class Segment:
    content: str
    start_offset: int
    end_offset: int
    metadata: dict[str, object] = field(default_factory=dict)


def segment_message(content: str, token_counter: TokenCounter, *, minimum_tokens: int = 12, maximum_tokens: int = 500) -> list[Segment]:
    if not content.strip():
        return []
    blocks = list(re.finditer(r"```[\s\S]*?```|(?:[^`]|`(?!``))+", content))
    units: list[Segment] = []
    for match in blocks:
        text = match.group(0)
        if text.startswith("```"):
            units.append(Segment(text.strip(), match.start(), match.end(), {"code_block": True}))
            continue
        for paragraph in re.finditer(r"\S(?:[\s\S]*?\S)?(?=\n\s*\n|\Z)", text):
            value = paragraph.group(0).strip()
            start = match.start() + paragraph.start()
            if token_counter.count(value) <= maximum_tokens:
                units.append(Segment(value, start, start + len(value)))
                continue
            words = list(re.finditer(r"\S+", value))
            chunk_start = 0
            while chunk_start < len(words):
                chunk_end = chunk_start + 1
                while chunk_end < len(words) and token_counter.count(value[words[chunk_start].start():words[chunk_end].end()]) <= maximum_tokens:
                    chunk_end += 1
                end_word = words[max(chunk_start, chunk_end - 1)]
                chunk = value[words[chunk_start].start():end_word.end()]
                units.append(Segment(chunk, start + words[chunk_start].start(), start + end_word.end()))
                chunk_start = max(chunk_start + 1, chunk_end - 1)

    merged: list[Segment] = []
    for unit in units:
        if token_counter.count(unit.content) >= minimum_tokens or unit.metadata.get("code_block") or not merged:
            merged.append(unit)
        else:
            previous = merged[-1]
            merged[-1] = Segment(previous.content + "\n\n" + unit.content, previous.start_offset, unit.end_offset, {**previous.metadata, **unit.metadata})
    if len(merged) > 1 and token_counter.count(merged[0].content) < minimum_tokens:
        first, second = merged[0], merged[1]
        merged[:2] = [Segment(first.content + "\n\n" + second.content, first.start_offset, second.end_offset, {**first.metadata, **second.metadata})]
    return merged

