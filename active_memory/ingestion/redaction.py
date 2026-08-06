"""Configurable secret redaction performed before local persistence."""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_PATTERNS = (
    r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+",
    r"(?i)(password\s*[:=]\s*)[^\s,;]+",
    r"(?i)(authorization:\s*bearer\s+)[A-Za-z0-9._~+/-]+",
    r"\b(?:sk|sk-ant)-[A-Za-z0-9_-]{12,}\b",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    r"(?i)(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?):\/\/[^\s]+",
)


class SecretRedactor:
    def __init__(self, patterns: Sequence[str] = DEFAULT_PATTERNS) -> None:
        self.patterns = [re.compile(pattern) for pattern in patterns]

    def redact(self, text: str) -> tuple[str, int]:
        redacted = text
        count = 0
        for pattern in self.patterns:
            def replace(match: re.Match[str]) -> str:
                nonlocal count
                count += 1
                prefix = match.group(1) if match.lastindex else ""
                return f"{prefix}[REDACTED]"
            redacted = pattern.sub(replace, redacted)
        return redacted, count

