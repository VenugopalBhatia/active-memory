"""Unified config file loading for all active-memory interfaces.

Supports a single JSON config file that works across proxy, chat CLI,
and MCP server.  Priority: dataclass defaults < config file < CLI flags.

Auto-discovers ~/.active-memory/config.json when no explicit path is given.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path
from typing import Any, get_type_hints

_DEFAULT_CONFIG_PATH = Path.home() / ".active-memory" / "config.json"

# Top-level convenience keys that map into assembler fields
_ASSEMBLER_ALIASES = {
    "budget": "total_budget",
    "recency_window": "recency_window",
    "pinned_reserve": "pinned_reserve",
}


def load_config(
    config_path: str | None = None,
    *,
    auto_discover: bool = True,
) -> dict[str, Any]:
    """Load a JSON config file and return it as a plain dict.

    Resolution:
      1. If *config_path* is given, load that file (error if missing).
      2. Else if *auto_discover* and ~/.active-memory/config.json exists, use it.
      3. Otherwise return {}.
    """
    if config_path is not None:
        p = Path(config_path)
        if not p.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        return json.loads(p.read_text())

    if auto_discover and _DEFAULT_CONFIG_PATH.exists():
        return json.loads(_DEFAULT_CONFIG_PATH.read_text())

    return {}


def _coerce(value: Any, target_type: type) -> Any:
    """Coerce a JSON value to the target type."""
    if target_type is bool:
        return bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return str(value)
    return value


def apply_overrides(dc_instance: Any, overrides: dict[str, Any]) -> None:
    """Apply dict overrides to a dataclass instance with type coercion.

    Skips keys that don't correspond to fields on the dataclass.
    """
    hints = get_type_hints(type(dc_instance))
    for f in dataclasses.fields(dc_instance):
        if f.name in overrides:
            setattr(dc_instance, f.name, _coerce(overrides[f.name], hints[f.name]))


def build_scoring_config(raw: dict[str, Any]) -> "ScoringConfig":
    """Build a ScoringConfig from the 'scoring' section of a config dict."""
    from .scoring import ScoringConfig

    cfg = ScoringConfig()
    section = raw.get("scoring", {})
    if section:
        apply_overrides(cfg, section)
    return cfg


def build_btree_config(raw: dict[str, Any]) -> "BTreeConfig":
    """Build a BTreeConfig from the 'btree' section of a config dict."""
    from .btree import BTreeConfig

    cfg = BTreeConfig()
    section = raw.get("btree", {})
    if section:
        apply_overrides(cfg, section)
    return cfg


def build_assembler_config(raw: dict[str, Any]) -> "AssemblerConfig":
    """Build an AssemblerConfig from config dict.

    Reads the 'assembler' section, with fallback to top-level convenience
    keys ('budget' -> total_budget, 'recency_window', 'pinned_reserve').
    """
    from .assembler import AssemblerConfig

    cfg = AssemblerConfig()
    section = raw.get("assembler", {})

    # Apply top-level convenience aliases first (lower priority)
    for alias, field_name in _ASSEMBLER_ALIASES.items():
        if alias in raw and field_name not in section:
            section[field_name] = raw[alias]

    if section:
        apply_overrides(cfg, section)
    return cfg


def build_grounding_config(raw: dict[str, Any]) -> "GroundingConfig":
    """Build a GroundingConfig from the 'grounding' section of a config dict."""
    from .grounding import GroundingConfig

    cfg = GroundingConfig()
    section = raw.get("grounding", {})
    if section:
        apply_overrides(cfg, section)
    return cfg


def build_middleware_config(raw: dict[str, Any]) -> "MiddlewareConfig":
    """Build a complete MiddlewareConfig from a config dict.

    Populates nested sub-configs from their respective sections, and
    reads top-level keys for shared fields (model, budget, etc.).
    """
    from .middleware import MiddlewareConfig

    cfg = MiddlewareConfig(
        scoring=build_scoring_config(raw),
        btree=build_btree_config(raw),
        assembler=build_assembler_config(raw),
        grounding=build_grounding_config(raw),
    )

    # Apply top-level middleware fields
    top_level = {
        k: v for k, v in raw.items()
        if k in ("model", "system_prompt", "max_tokens",
                 "prune_interval", "compress_interval",
                 "auto_ingest_responses")
    }
    if top_level:
        apply_overrides(cfg, top_level)

    return cfg
