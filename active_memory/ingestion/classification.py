"""Conservative deterministic memory-worthiness and trust classification."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Classification:
    memory_type: str
    trust_level: str
    metadata: dict[str, object] = field(default_factory=dict)


_SECRET_MARKERS = re.compile(r"(?i)(api[_ -]?key|password|private key|access token|connection string)")
_SPECULATION = re.compile(r"(?i)\b(may|might|perhaps|possibly|I think|I guess|could be)\b")
_FILES = re.compile(r"\b[\w./-]+\.(?:py|js|ts|tsx|jsx|json|md|ya?ml|toml|go|rs|java)\b")


def classify_segment(role: str, content: str, metadata: dict | None = None) -> Classification | None:
    lower = content.lower()
    metadata_out: dict[str, object] = dict(metadata or {})
    if _SECRET_MARKERS.search(content) or "[redacted]" in lower:
        return None
    if role == "assistant" and _SPECULATION.search(content):
        return None
    if len(content.strip()) < 8 or lower.strip() in {"hello", "hi", "thanks", "thank you", "ok", "okay"}:
        return None

    trust = {"user": "user_confirmed", "tool": "tool_observed", "assistant": "assistant_generated", "system": "external_untrusted"}.get(role, "external_untrusted")
    memory_type: str | None = None
    if any(word in lower for word in ("resolved", "fixed", "root cause", "was the cause", "expired fixture")):
        memory_type = "resolution"
    elif any(word in lower for word in ("failed", "failure", "error", "exception", "broken")):
        memory_type = "error"
    elif any(word in lower for word in ("todo", "must ", "need to", "unresolved", "follow up")):
        memory_type = "task"
    elif any(word in lower for word in ("prefer", "preference", "always use", "do not use")):
        memory_type = "preference"
    elif any(word in lower for word in ("decided", "decision", "we use", "we are using", "migrated to", "switch to", "choose ")):
        memory_type = "decision"
    elif role == "tool":
        memory_type = "tool_observation"
    elif any(word in lower for word in ("implemented", "changed", "updated", "refactored", "added", "removed")) and _FILES.search(content):
        memory_type = "code_change"
    elif role == "user" and any(word in lower for word in (" is ", " are ", " uses ", "actually", "correction")):
        memory_type = "fact"
    if memory_type is None:
        return None

    files = _FILES.findall(content)
    if files:
        metadata_out["file_path"] = files[0]
    database_values = [value for value in ("mongodb", "postgresql", "postgres", "mysql", "sqlite") if value in lower]
    if database_values and memory_type == "decision":
        metadata_out["conflict_key"] = "project_database"
        metadata_out["entities"] = database_values
    if any(value in lower for value in ("auth", "authentication")) and any(value in lower for value in ("cause", "failed", "failure", "expired")):
        metadata_out["conflict_key"] = "authentication_failure_cause"
    return Classification(memory_type, trust, metadata_out)

