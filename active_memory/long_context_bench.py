"""Long-context offline benchmark for active-memory.

This benchmark is intended to answer three questions as transcript size
grows toward and beyond 1M raw tokens:

1. Does the retrieval layer still surface early planted facts?
2. Does the assembled prompt stay materially smaller than the raw history?
3. Does assembly latency remain usable as the tree grows?

The benchmark is offline and deterministic. It does not call an LLM.
Instead it checks whether the assembled prompt contains the expected
keywords for planted facts.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import numpy as np

from .assembler import AssemblerConfig, ContextAssembler
from .btree import BTreeConfig, SemanticBTree
from .scoring import Scorer, ScoringConfig
from .types import Embedding, Embedder, KVTuple, estimate_tokens


class TopicEmbedder:
    """Deterministic semantic-ish embedder for reproducible offline benches."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim
        self._topics = {
            0: {"database", "postgres", "clickhouse", "catalog", "sql"},
            1: {"cache", "caching", "redis", "ttl", "session"},
            2: {"auth", "token", "pkce", "oauth", "refresh"},
            3: {"deploy", "docker", "release", "rollout", "build"},
            4: {"monitoring", "metrics", "grafana", "latency", "alerts"},
            5: {"budget", "cost", "finance", "spend"},
            6: {"vendor", "partner", "contact", "email"},
            7: {"frontend", "ui", "react", "design"},
            8: {"testing", "pytest", "coverage", "integration"},
            9: {"pipeline", "kafka", "ingestion", "streaming"},
        }

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[Embedding]:
        out: list[Embedding] = []
        for text in texts:
            lower = text.lower().replace("-", " ")
            vec = np.zeros(self._dim, dtype=np.float32)
            for idx, words in self._topics.items():
                if any(word in lower for word in words):
                    vec[idx] += 2.0
            for ch in lower:
                vec[(ord(ch) % (self._dim - 1)) + 1] += 0.04
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec[0] = 1.0
                norm = 1.0
            out.append((vec / norm).astype(np.float32))
        return out


@dataclass
class Probe:
    fact_id: str
    query: str
    expected_keywords: list[str]
    planted_turn: int


@dataclass
class Scenario:
    messages: list[dict]
    probes: list[Probe]
    raw_tokens: int
    turns: int


@dataclass
class StrategyMetrics:
    strategy: str
    raw_tokens: int
    avg_prompt_tokens: float
    avg_prompt_ratio: float
    recall: float
    avg_probe_latency_ms: float
    p95_probe_latency_ms: float
    max_prompt_tokens: int


@dataclass
class BenchmarkResult:
    token_target: int
    actual_raw_tokens: int
    turns: int
    strategies: list[StrategyMetrics]


def _estimate_message_tokens(messages: list[dict]) -> int:
    total = 0
    for message in messages:
        total += estimate_tokens(str(message.get("content", "")))
    return total


def _messages_to_text(messages: list[dict]) -> str:
    return "\n".join(f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in messages)


def build_long_context_scenario(
    token_target: int,
    *,
    filler_repetitions: int = 12,
) -> Scenario:
    """Create a deterministic transcript with early, middle, and late facts."""
    facts = [
        (
            "database",
            "Database decision: use PostgreSQL 16 for the metadata catalog and ClickHouse for analytics.",
            "Which database stack did we choose?",
            ["postgresql", "clickhouse"],
        ),
        (
            "auth",
            "Authentication decision: use Auth0 with PKCE, 12-minute access tokens, and rotated refresh tokens.",
            "What authentication setup did we decide on?",
            ["auth0", "pkce", "refresh"],
        ),
        (
            "budget",
            "Budget constraint: monthly infrastructure spend must stay below $12,000 without VP approval.",
            "What was the infrastructure budget limit?",
            ["12,000"],
        ),
        (
            "vendor",
            "Vendor contact: Priya Sharma at Nexus Data, priya.sharma@nexusdata.io, is the primary integration contact.",
            "Who is the vendor contact and what is their email?",
            ["priya", "nexusdata"],
        ),
        (
            "cache",
            "Caching decision: use Redis with short TTLs for sessions and expensive query fragments.",
            "What caching layer did we pick?",
            ["redis"],
        ),
    ]

    filler_blocks = [
        "Frontend review: React dashboard components, loading states, and chart interactions all need refinement before release.",
        "Testing plan: integration coverage should include Kafka ingestion, PostgreSQL writes, and Redis invalidation behavior.",
        "Monitoring plan: capture p50, p95, and p99 latency, queue lag, and cost drift in Grafana dashboards.",
        "Deployment notes: canary rollouts, Docker images, and rollback criteria should be documented in the release runbook.",
        "Data pipeline notes: Kafka topics, schema evolution, and replay workflows need explicit ownership and alerting.",
    ]

    messages: list[dict] = []
    probes: list[Probe] = []
    raw_tokens = 0
    turn = 0
    fact_positions = {1, 3, 8, 14, 22}
    fact_index = 0

    while raw_tokens < token_target:
        turn += 1

        if turn in fact_positions and fact_index < len(facts):
            fact_id, fact_text, query, keywords = facts[fact_index]
            user_content = fact_text
            assistant_content = f"Recorded. We will retain this {fact_id} decision for later reference."
            probes.append(Probe(
                fact_id=fact_id,
                query=query,
                expected_keywords=keywords,
                planted_turn=turn,
            ))
            fact_index += 1
        else:
            filler = filler_blocks[(turn - 1) % len(filler_blocks)]
            user_content = " ".join([filler] * filler_repetitions)
            assistant_content = (
                "Acknowledged. "
                + " ".join(
                    [f"Follow-up note {turn}: keep the current plan internally consistent."] * (filler_repetitions // 2)
                )
            )

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        raw_tokens = _estimate_message_tokens(messages)

    return Scenario(
        messages=messages,
        probes=probes,
        raw_tokens=raw_tokens,
        turns=len(messages) // 2,
    )


class FullContextStrategy:
    name = "full_context"

    def __init__(self, messages: list[dict]) -> None:
        self._messages = list(messages)

    def build_prompt(self, probe: Probe) -> list[dict]:
        return self._messages + [{"role": "user", "content": probe.query}]


class SlidingWindowStrategy:
    def __init__(self, messages: list[dict], window_turns: int) -> None:
        self._messages = list(messages)
        self._window_turns = window_turns
        self.name = f"sliding_window_{window_turns}"

    def build_prompt(self, probe: Probe) -> list[dict]:
        return self._messages[-self._window_turns * 2 :] + [{"role": "user", "content": probe.query}]


class ActiveMemoryStrategy:
    name = "active_memory"

    def __init__(
        self,
        messages: list[dict],
        *,
        embedder: Embedder,
        budget: int,
        recency_window: int = 6,
        max_tuples: int = 16,
    ) -> None:
        self._messages = list(messages)
        self._embedder = embedder
        self._scorer = Scorer(ScoringConfig())
        self._tree = SemanticBTree(
            embedder=embedder,
            scorer=self._scorer,
            config=BTreeConfig(max_tuples=max_tuples),
        )
        self._assembler = ContextAssembler(
            tree=self._tree,
            config=AssemblerConfig(
                total_budget=budget,
                pinned_reserve=min(8_000, max(500, budget // 8)),
                recency_window=recency_window,
                managed_top_k=200,
            ),
        )
        self._ingest_messages()

    def _ingest_messages(self) -> None:
        import re

        for message in self._messages:
            content = str(message.get("content", ""))
            role = message.get("role", "user")
            segments = re.split(r"(?<=[.!?\n])\s+", content)
            buf = ""
            for segment in segments:
                buf += (" " if buf else "") + segment
                if len(buf) >= 60:
                    self._tree.insert(f"{role}:{buf[:60]}", buf)
                    buf = ""
            if buf and len(buf.strip()) >= 10:
                self._tree.insert(f"{role}:{buf[:60]}", buf)

    def build_prompt(self, probe: Probe) -> list[dict]:
        query_emb = self._embedder.embed([probe.query])[0]
        assembled = self._assembler.assemble(self._messages, query_emb)
        messages = self._assembler.to_messages("You are a helpful assistant.", assembled)
        messages.append({"role": "user", "content": probe.query})
        return messages


def evaluate_strategy(strategy, probes: list[Probe], raw_tokens: int) -> StrategyMetrics:
    latencies_ms: list[float] = []
    prompt_tokens: list[int] = []
    passes = 0

    for probe in probes:
        start = time.perf_counter()
        prompt = strategy.build_prompt(probe)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies_ms.append(elapsed_ms)

        prompt_token_count = _estimate_message_tokens(prompt)
        prompt_tokens.append(prompt_token_count)

        prompt_text = _messages_to_text(prompt).lower()
        passed = all(keyword.lower() in prompt_text for keyword in probe.expected_keywords)
        if passed:
            passes += 1

    recall = passes / max(1, len(probes))
    avg_prompt = statistics.mean(prompt_tokens) if prompt_tokens else 0.0
    avg_latency = statistics.mean(latencies_ms) if latencies_ms else 0.0
    p95_latency = np.percentile(latencies_ms, 95) if latencies_ms else 0.0

    return StrategyMetrics(
        strategy=strategy.name,
        raw_tokens=raw_tokens,
        avg_prompt_tokens=avg_prompt,
        avg_prompt_ratio=(avg_prompt / raw_tokens) if raw_tokens else 0.0,
        recall=recall,
        avg_probe_latency_ms=avg_latency,
        p95_probe_latency_ms=float(p95_latency),
        max_prompt_tokens=max(prompt_tokens) if prompt_tokens else 0,
    )


def run_benchmark(
    token_targets: list[int],
    *,
    sliding_window_turns: int,
    active_memory_budget: int,
    filler_repetitions: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []
    embedder = TopicEmbedder()

    for token_target in token_targets:
        scenario = build_long_context_scenario(
            token_target,
            filler_repetitions=filler_repetitions,
        )
        strategies = [
            FullContextStrategy(scenario.messages),
            SlidingWindowStrategy(scenario.messages, sliding_window_turns),
            ActiveMemoryStrategy(
                scenario.messages,
                embedder=embedder,
                budget=active_memory_budget,
            ),
        ]

        metrics = [
            evaluate_strategy(strategy, scenario.probes, scenario.raw_tokens)
            for strategy in strategies
        ]
        results.append(BenchmarkResult(
            token_target=token_target,
            actual_raw_tokens=scenario.raw_tokens,
            turns=scenario.turns,
            strategies=metrics,
        ))

    return results


def print_report(results: list[BenchmarkResult]) -> None:
    for result in results:
        print("\n" + "=" * 88)
        print(
            f"Long-context benchmark | target={result.token_target:,} "
            f"| actual={result.actual_raw_tokens:,} raw tokens | turns={result.turns}"
        )
        print("=" * 88)
        print(
            f"{'Strategy':<20} {'Recall':>8} {'Avg Prompt':>12} {'Prompt/Raw':>12} "
            f"{'Avg ms':>10} {'P95 ms':>10}"
        )
        print(
            f"{'-'*20} {'-'*8} {'-'*12} {'-'*12} {'-'*10} {'-'*10}"
        )
        for metric in result.strategies:
            print(
                f"{metric.strategy:<20} {metric.recall:>7.0%} "
                f"{metric.avg_prompt_tokens:>11,.0f} "
                f"{metric.avg_prompt_ratio:>11.1%} "
                f"{metric.avg_probe_latency_ms:>9.1f} "
                f"{metric.p95_probe_latency_ms:>9.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline long-context benchmark for active-memory",
    )
    parser.add_argument(
        "--token-targets",
        default="100000,250000,500000,1000000",
        help="Comma-separated raw transcript token targets",
    )
    parser.add_argument(
        "--sliding-window-turns",
        type=int,
        default=20,
        help="Window size for the sliding-window baseline",
    )
    parser.add_argument(
        "--active-memory-budget",
        type=int,
        default=100000,
        help="Prompt budget for active-memory strategy",
    )
    parser.add_argument(
        "--filler-repetitions",
        type=int,
        default=12,
        help="How repetitive each distractor turn should be",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON after the text report",
    )
    args = parser.parse_args()

    token_targets = [int(part.strip()) for part in args.token_targets.split(",") if part.strip()]
    results = run_benchmark(
        token_targets,
        sliding_window_turns=args.sliding_window_turns,
        active_memory_budget=args.active_memory_budget,
        filler_repetitions=args.filler_repetitions,
    )

    print_report(results)

    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))


if __name__ == "__main__":
    main()
