"""Typed application configuration loaded from JSON or optional YAML."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from active_memory.context import BudgetConfig
from active_memory.retrieval import FinalWeights, RetrievalConfig
from active_memory.retrieval.scoring import ScoringPolicy, StageOneWeights


@dataclass(frozen=True, slots=True)
class StorageSettings:
    backend: str = "sqlite"
    path: str = "~/.active-memory/memory.db"

    def __post_init__(self) -> None:
        if self.backend != "sqlite":
            raise ValueError("only the sqlite backend is currently supported")


@dataclass(frozen=True, slots=True)
class EmbeddingSettings:
    provider: str = "local"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32


@dataclass(frozen=True, slots=True)
class RetrievalSettings:
    candidate_limit: int = 80
    seed_limit: int = 6
    neighbor_limit: int = 30
    result_limit: int = 20
    minimum_relevance: float = 0.20


@dataclass(frozen=True, slots=True)
class ScoringSettings:
    stage_one_relevance_weight: float = 0.65
    stage_one_recency_weight: float = 0.25
    stage_one_frequency_weight: float = 0.10
    relevance_weight: float = 0.55
    recency_weight: float = 0.20
    frequency_weight: float = 0.10
    affinity_weight: float = 0.15

    def __post_init__(self) -> None:
        if not math.isclose(self.stage_one_relevance_weight + self.stage_one_recency_weight + self.stage_one_frequency_weight, 1.0, abs_tol=1e-9):
            raise ValueError("stage-one scoring weights must sum to 1.0")
        if not math.isclose(self.relevance_weight + self.recency_weight + self.frequency_weight + self.affinity_weight, 1.0, abs_tol=1e-9):
            raise ValueError("final scoring weights must sum to 1.0")


@dataclass(frozen=True, slots=True)
class BudgetSettings:
    model_context_limit: int = 200_000
    reserved_response_tokens: int = 8_000
    safety_margin_tokens: int = 2_000
    recent_turn_fraction: float = 0.35
    memory_fraction: float = 0.35
    tool_context_fraction: float = 0.20
    recent_message_limit: int = 8


@dataclass(frozen=True, slots=True)
class MemorySettings:
    minimum_segment_tokens: int = 12
    maximum_segment_tokens: int = 500
    default_namespace: str = "global"
    store_assistant_generated: bool = True
    storage_enabled: bool = True


@dataclass(frozen=True, slots=True)
class ProxySettings:
    upstream_url: str = "https://api.anthropic.com"
    host: str = "127.0.0.1"
    port: int = 8080
    strict_memory: bool = False


@dataclass(frozen=True, slots=True)
class ActiveMemoryConfig:
    storage: StorageSettings = field(default_factory=StorageSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    scoring: ScoringSettings = field(default_factory=ScoringSettings)
    budget: BudgetSettings = field(default_factory=BudgetSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    proxy: ProxySettings = field(default_factory=ProxySettings)

    def retrieval_config(self) -> RetrievalConfig:
        stage = StageOneWeights(self.scoring.stage_one_relevance_weight, self.scoring.stage_one_recency_weight, self.scoring.stage_one_frequency_weight)
        final = FinalWeights(self.scoring.relevance_weight, self.scoring.recency_weight, self.scoring.frequency_weight, self.scoring.affinity_weight)
        return RetrievalConfig(self.retrieval.candidate_limit, self.retrieval.seed_limit, self.retrieval.neighbor_limit, self.retrieval.result_limit, self.retrieval.minimum_relevance, ScoringPolicy(stage_one=stage), final)

    def budget_config(self) -> BudgetConfig:
        return BudgetConfig(**{name: getattr(self.budget, name) for name in BudgetSettings.__dataclass_fields__})


def _section(data: dict[str, Any], key: str, cls: type[Any]) -> Any:
    values = data.get(key, {})
    if not isinstance(values, dict):
        raise TypeError(f"configuration section {key!r} must be an object")
    unknown = set(values) - set(cls.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown {key} configuration fields: {', '.join(sorted(unknown))}")
    return cls(**values)


def config_from_dict(data: dict[str, Any]) -> ActiveMemoryConfig:
    known = {"storage", "embeddings", "retrieval", "scoring", "budget", "memory", "proxy"}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown configuration sections: {', '.join(sorted(unknown))}")
    return ActiveMemoryConfig(
        _section(data, "storage", StorageSettings), _section(data, "embeddings", EmbeddingSettings),
        _section(data, "retrieval", RetrievalSettings), _section(data, "scoring", ScoringSettings),
        _section(data, "budget", BudgetSettings), _section(data, "memory", MemorySettings),
        _section(data, "proxy", ProxySettings),
    )


def load_config(path: str | Path | None = None) -> ActiveMemoryConfig:
    config_path = Path(path).expanduser() if path else Path("~/.active-memory/config.json").expanduser()
    if not config_path.exists():
        return ActiveMemoryConfig()
    text = config_path.read_text(encoding="utf-8")
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required to load YAML configuration") from exc
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise TypeError("configuration root must be an object")
    return config_from_dict(data)
