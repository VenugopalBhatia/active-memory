"""Structured JSON logging helpers that avoid raw memory content."""

from __future__ import annotations

import json
import logging
from typing import Any


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    safe = {key: value for key, value in fields.items() if key not in {"content", "api_key", "password", "token"}}
    logger.info(json.dumps({"event": event, **safe}, sort_keys=True, default=str))

