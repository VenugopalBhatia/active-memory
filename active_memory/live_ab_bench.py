"""Live A/B benchmark for active-memory under long-context sessions.

Generates synthetic transcripts at 100K���1M raw tokens, then sends
real probe questions to Claude via the Anthropic API. Compares:

  1. full_context   — send everything (capped at model limit)
  2. sliding_window  — last N turns only
  3. active_memory   — B-tree assembled prompt within token budget

At 1M raw tokens the full-context arm literally can't fit, so it gets
truncated to the model's context window. That's the real-world scenario
active-memory is designed for: conversations that outgrow the window.

Usage:
    # Quick sanity check (100K only, 1 trial)
    python -m active_memory.live_ab_bench --quick

    # Full benchmark (100K → 1M, 3 trials per probe)
    python -m active_memory.live_ab_bench --trials 3

    # Custom scales
    python -m active_memory.live_ab_bench --scales 100000,500000,1000000

    # Rate-safe mode (smaller scales for rate-limited accounts)
    python -m active_memory.live_ab_bench --rate-safe

    # Budget sweep (vary AM budget at a fixed scale)
    python -m active_memory.live_ab_bench --sweep-budgets 10000,12000,14000,15000
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .assembler import AssemblerConfig, ContextAssembler
from .btree import BTreeConfig, SemanticBTree
from .embeddings import create_embedder
from .model_clients import ModelClient, create_model_client
from .scoring import Scorer, ScoringConfig
from .types import Embedder, Embedding, estimate_tokens


# ── Scenario generation ──────────────────────────────────────────────

@dataclass
class PlantedFact:
    fact_id: str
    content: str            # the user message containing the fact
    query: str              # probe question to test retrieval
    expected_keywords: list[str]
    planted_turn: int = 0   # filled during generation


@dataclass
class LongContextScenario:
    name: str
    messages: list[dict]
    facts: list[PlantedFact]
    raw_tokens: int
    turns: int


PLANTED_FACTS = [
    PlantedFact(
        fact_id="database",
        content="Database decision: use PostgreSQL 16 for the metadata catalog and ClickHouse for analytics.",
        query="What databases did we choose and what is each one used for?",
        expected_keywords=["PostgreSQL", "ClickHouse"],
    ),
    PlantedFact(
        fact_id="auth",
        content="Authentication decision: use Auth0 with PKCE flow, 12-minute access tokens, and rotated refresh tokens.",
        query="What authentication setup did we decide on? Include token lifetimes.",
        expected_keywords=["Auth0", "PKCE", "12"],
    ),
    PlantedFact(
        fact_id="budget",
        content="Budget constraint: monthly infrastructure spend must stay below $12,000 without VP approval from David Chen.",
        query="What is our monthly infrastructure budget cap and who approves overages?",
        expected_keywords=["12,000", "David Chen"],
    ),
    PlantedFact(
        fact_id="vendor",
        content="Vendor contact: Priya Sharma at Nexus Data, priya.sharma@nexusdata.io, is the primary integration contact.",
        query="Who is the vendor contact at Nexus Data and what is their email?",
        expected_keywords=["Priya", "nexusdata"],
    ),
    PlantedFact(
        fact_id="codename",
        content="Project codename is AURORA. All internal references must use this codename.",
        query="What is the project codename?",
        expected_keywords=["AURORA"],
    ),
]

FILLER_BLOCKS = [
    "Frontend review: React dashboard components, loading states, and chart interactions all need refinement before release.",
    "Testing plan: integration coverage should include Kafka ingestion, PostgreSQL writes, and Redis invalidation behavior.",
    "Monitoring plan: capture p50, p95, and p99 latency, queue lag, and cost drift in Grafana dashboards.",
    "Deployment notes: canary rollouts, Docker images, and rollback criteria should be documented in the release runbook.",
    "Data pipeline notes: Kafka topics, schema evolution, and replay workflows need explicit ownership and alerting.",
    "Security review: audit TLS configuration, API key rotation, dependency vulnerability scanning, and RBAC policies.",
    "Performance work: profile hot paths, optimize serialization, review connection pool sizing, and measure cold-start latency.",
    "Documentation: update architecture diagrams, add runbook entries for common failure modes, and refresh API reference.",
]


def build_scenario(
    token_target: int,
    *,
    filler_repetitions: int = 10,
) -> LongContextScenario:
    """Build a transcript at the target token count with planted facts early."""
    messages: list[dict] = []
    facts = [PlantedFact(**{k: v for k, v in asdict(f).items()}) for f in PLANTED_FACTS]

    # Plant facts in the first 25 turns
    fact_positions = {1, 4, 8, 14, 22}
    fact_iter = iter(facts)
    turn = 0

    raw_tokens = 0
    while raw_tokens < token_target:
        turn += 1

        if turn in fact_positions:
            try:
                fact = next(fact_iter)
                fact.planted_turn = turn
                user_content = fact.content
                assistant_content = f"Understood. The {fact.fact_id} decision is recorded."
            except StopIteration:
                user_content, assistant_content = _filler(turn, filler_repetitions)
        else:
            user_content, assistant_content = _filler(turn, filler_repetitions)

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": assistant_content})
        raw_tokens = sum(estimate_tokens(m["content"]) for m in messages)

    # Assign planted_turn for any facts that didn't get placed
    for f in facts:
        if f.planted_turn == 0:
            f.planted_turn = -1  # not planted

    name = f"live_{token_target // 1000}k"
    return LongContextScenario(
        name=name,
        messages=messages,
        facts=[f for f in facts if f.planted_turn > 0],
        raw_tokens=raw_tokens,
        turns=turn,
    )


def _filler(turn: int, reps: int) -> tuple[str, str]:
    block = FILLER_BLOCKS[(turn - 1) % len(FILLER_BLOCKS)]
    user_content = " ".join([block] * reps)
    assistant_content = " ".join(
        [f"Follow-up note {turn}: acknowledged, keeping plan consistent."] * (reps // 2)
    )
    return user_content, assistant_content


# ── Strategy adapters ────────────────────────────────────────────────

class FullContextStrategy:
    """Send all messages, truncated to fit the model's context window."""
    name = "full_context"

    def __init__(self, messages: list[dict], max_tokens: int = 190_000) -> None:
        self._messages = messages
        self._max_tokens = max_tokens

    def build_probe(self, fact: PlantedFact) -> list[dict]:
        # Truncate from the front to fit within model limit
        result = list(self._messages)
        while self._estimate(result) > self._max_tokens and len(result) > 2:
            result = result[2:]  # drop oldest user+assistant pair
        result.append({"role": "user", "content": fact.query})
        return result

    @staticmethod
    def _estimate(msgs: list[dict]) -> int:
        return sum(estimate_tokens(m["content"]) for m in msgs)


class SlidingWindowStrategy:
    """Keep only the last N turns."""

    def __init__(self, messages: list[dict], window_turns: int = 20) -> None:
        self._messages = messages
        self._window = window_turns
        self.name = f"sliding_window_{window_turns}"

    def build_probe(self, fact: PlantedFact) -> list[dict]:
        windowed = self._messages[-self._window * 2:]
        windowed.append({"role": "user", "content": fact.query})
        return windowed


class ActiveMemoryStrategy:
    """B-tree assembled prompt within a token budget."""
    name = "active_memory"

    def __init__(
        self,
        messages: list[dict],
        embedder: Embedder,
        budget: int = 100_000,
        recency_window: int = 6,
        max_tuples: int = 16,
    ) -> None:
        import re

        self._messages = messages
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

        # Ingest all messages
        for msg in messages:
            content = msg.get("content", "")
            role = msg.get("role", "user")
            segments = re.split(r"(?<=[.!?\n])\s+", content)
            buf = ""
            for seg in segments:
                buf += (" " if buf else "") + seg
                if len(buf) >= 60:
                    self._tree.insert(f"{role}:{buf[:60]}", buf)
                    buf = ""
            if buf and len(buf.strip()) >= 10:
                self._tree.insert(f"{role}:{buf[:60]}", buf)

        # Light maintenance — only compress, don't prune facts we just ingested.
        # Pruning immediately after batch ingest destroys old-but-relevant facts
        # before they get a chance to be retrieved by a relevant query.
        self._tree.compress_cold_subtrees()

    def build_probe(self, fact: PlantedFact) -> list[dict]:
        query_emb = self._embedder.embed([fact.query])[0]
        assembled = self._assembler.assemble(self._messages, query_emb)
        messages = self._assembler.to_messages(
            "You are a helpful assistant. Answer precisely based on our conversation history.",
            assembled,
        )
        messages.append({"role": "user", "content": fact.query})
        return messages


# ── Evaluation ───────────────────────────────────────────────────────

@dataclass
class ProbeTrialResult:
    fact_id: str
    strategy: str
    trial: int
    passed: bool
    keywords_found: dict[str, bool]
    response_text: str
    prompt_tokens: int
    latency_ms: float
    input_tokens_billed: int   # from API response usage
    output_tokens_billed: int


@dataclass
class ProbeAggResult:
    fact_id: str
    planted_turn: int
    recall_rate: float        # fraction of trials that passed
    avg_prompt_tokens: float
    avg_latency_ms: float
    trials: list[ProbeTrialResult]


@dataclass
class StrategyScaleResult:
    strategy: str
    scale_name: str
    raw_tokens: int
    turns: int
    overall_recall: float
    avg_prompt_tokens: float
    avg_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_api_calls: int
    per_fact: list[ProbeAggResult]


def _wilson_ci(rate: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z**2 / n
    centre = (rate + z**2 / (2 * n)) / denom
    spread = z * math.sqrt((rate * (1 - rate) + z**2 / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def run_live_ab(
    client: ModelClient,
    *,
    scales: list[int] = None,
    trials_per_probe: int = 3,
    model: str = "claude-sonnet-4-20250514",
    max_response_tokens: int = 512,
    embedder: Embedder | None = None,
    active_memory_budget: int = 120_000,
    sliding_window_turns: int = 20,
    filler_repetitions: int = 10,
    verbose: bool = True,
) -> list[StrategyScaleResult]:
    """Run the full live A/B benchmark.

    Returns one StrategyScaleResult per (strategy, scale) pair.
    """
    if scales is None:
        scales = [100_000, 250_000, 500_000, 1_000_000]

    if embedder is None:
        spec = create_embedder("auto", verbose=verbose, stream=sys.stderr)
        embedder = spec.embedder

    all_results: list[StrategyScaleResult] = []

    for scale in scales:
        print(f"\n{'━' * 76}", file=sys.stderr)
        print(f"  Building scenario: {scale:,} raw tokens", file=sys.stderr)
        print(f"{'━' * 76}", file=sys.stderr)

        scenario = build_scenario(scale, filler_repetitions=filler_repetitions)
        print(
            f"  Actual: {scenario.raw_tokens:,} tokens, "
            f"{scenario.turns} turns, "
            f"{len(scenario.facts)} planted facts",
            file=sys.stderr,
        )

        # Build strategies
        strategies: list[Any] = []
        print("  Building full_context strategy...", file=sys.stderr)
        strategies.append(FullContextStrategy(scenario.messages))
        print(f"  Building sliding_window_{sliding_window_turns} strategy...", file=sys.stderr)
        strategies.append(SlidingWindowStrategy(scenario.messages, sliding_window_turns))
        print("  Building active_memory strategy (ingesting + indexing)...", file=sys.stderr)
        t0 = time.perf_counter()
        strategies.append(ActiveMemoryStrategy(
            scenario.messages,
            embedder=embedder,
            budget=active_memory_budget,
        ))
        ingest_ms = (time.perf_counter() - t0) * 1000
        print(f"  Active memory ingestion: {ingest_ms:,.0f}ms", file=sys.stderr)

        # Probe each strategy
        for strategy in strategies:
            print(f"\n  ▸ {strategy.name}", file=sys.stderr)

            per_fact: list[ProbeAggResult] = []
            total_input = 0
            total_output = 0
            total_calls = 0

            for fact in scenario.facts:
                trials: list[ProbeTrialResult] = []

                for trial_idx in range(trials_per_probe):
                    probe_messages = strategy.build_probe(fact)
                    prompt_tokens = sum(
                        estimate_tokens(m["content"]) for m in probe_messages
                    )

                    # Retry with exponential backoff for rate limits
                    resp_text = ""
                    input_billed = 0
                    output_billed = 0
                    latency_ms = 0.0

                    for attempt in range(4):
                        t0 = time.perf_counter()
                        try:
                            response = client.generate(
                                model=model,
                                max_tokens=max_response_tokens,
                                messages=probe_messages,
                            )
                            latency_ms = (time.perf_counter() - t0) * 1000
                            resp_text = response.text
                            input_billed = response.input_tokens
                            output_billed = response.output_tokens
                            break
                        except Exception as e:
                            latency_ms = (time.perf_counter() - t0) * 1000
                            resp_text = f"[ERROR: {e}]"
                            if "rate" in str(e).lower() or "overloaded" in str(e).lower() or "529" in str(e) or "429" in str(e):
                                wait = 2 ** attempt * 2
                                print(
                                    f"      ⟳ Rate limited, waiting {wait}s (attempt {attempt + 1}/4)",
                                    file=sys.stderr,
                                )
                                time.sleep(wait)
                            else:
                                print(
                                    f"      ✗ API error: {e}",
                                    file=sys.stderr,
                                )
                                break

                    # Brief pause between API calls to avoid rate limits
                    time.sleep(1.0)

                    total_input += input_billed
                    total_output += output_billed
                    total_calls += 1

                    kw_found = {
                        kw: kw.lower() in resp_text.lower()
                        for kw in fact.expected_keywords
                    }
                    passed = all(kw_found.values())

                    trials.append(ProbeTrialResult(
                        fact_id=fact.fact_id,
                        strategy=strategy.name,
                        trial=trial_idx,
                        passed=passed,
                        keywords_found=kw_found,
                        response_text=resp_text,
                        prompt_tokens=prompt_tokens,
                        latency_ms=latency_ms,
                        input_tokens_billed=input_billed,
                        output_tokens_billed=output_billed,
                    ))

                    status = "PASS" if passed else "FAIL"
                    print(
                        f"    [{status}] {fact.fact_id} "
                        f"trial {trial_idx + 1}/{trials_per_probe} "
                        f"({latency_ms:,.0f}ms, {input_billed:,} in)",
                        file=sys.stderr,
                    )

                recall = sum(1 for t in trials if t.passed) / max(1, len(trials))
                per_fact.append(ProbeAggResult(
                    fact_id=fact.fact_id,
                    planted_turn=fact.planted_turn,
                    recall_rate=recall,
                    avg_prompt_tokens=statistics.mean(t.prompt_tokens for t in trials),
                    avg_latency_ms=statistics.mean(t.latency_ms for t in trials),
                    trials=trials,
                ))

            overall_recall = (
                statistics.mean(p.recall_rate for p in per_fact)
                if per_fact else 0.0
            )
            avg_prompt = (
                statistics.mean(p.avg_prompt_tokens for p in per_fact)
                if per_fact else 0.0
            )
            avg_latency = (
                statistics.mean(p.avg_latency_ms for p in per_fact)
                if per_fact else 0.0
            )

            all_results.append(StrategyScaleResult(
                strategy=strategy.name,
                scale_name=scenario.name,
                raw_tokens=scenario.raw_tokens,
                turns=scenario.turns,
                overall_recall=overall_recall,
                avg_prompt_tokens=avg_prompt,
                avg_latency_ms=avg_latency,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_api_calls=total_calls,
                per_fact=per_fact,
            ))

    return all_results


def print_report(results: list[StrategyScaleResult]) -> str:
    """Print a formatted comparison report."""
    lines: list[str] = []

    # Group by scale
    scales = sorted(set(r.scale_name for r in results))

    for scale in scales:
        scale_results = [r for r in results if r.scale_name == scale]
        raw = scale_results[0].raw_tokens
        turns = scale_results[0].turns

        lines.append(f"\n{'=' * 88}")
        lines.append(
            f"  {scale} | {raw:,} raw tokens | {turns} turns"
        )
        lines.append(f"{'=' * 88}")
        lines.append(
            f"  {'Strategy':<25} {'Recall':>8} {'95% CI':>16} "
            f"{'Avg Prompt':>12} {'Ratio':>8} {'Avg ms':>10} {'Cost (in)':>10}"
        )
        lines.append(
            f"  {'-' * 25} {'-' * 8} {'-' * 16} {'-' * 12} {'-' * 8} {'-' * 10} {'-' * 10}"
        )

        for r in scale_results:
            total_trials = sum(len(p.trials) for p in r.per_fact)
            total_passed = sum(
                sum(1 for t in p.trials if t.passed) for p in r.per_fact
            )
            rate = total_passed / max(1, total_trials)
            ci_lo, ci_hi = _wilson_ci(rate, total_trials)
            ratio = r.avg_prompt_tokens / max(1, r.raw_tokens)

            lines.append(
                f"  {r.strategy:<25} {r.overall_recall:>7.0%} "
                f"[{ci_lo:>5.0%}, {ci_hi:>5.0%}]   "
                f"{r.avg_prompt_tokens:>11,.0f} {ratio:>7.1%} "
                f"{r.avg_latency_ms:>9,.0f} {r.total_input_tokens:>9,}"
            )

        # Per-fact breakdown
        lines.append(f"\n  Per-fact recall:")
        header = f"  {'Fact':<15} {'Turn':>5}"
        for r in scale_results:
            header += f"  {r.strategy:>20}"
        lines.append(header)
        lines.append(f"  {'-' * 15} {'-' * 5}" + f"  {'-' * 20}" * len(scale_results))

        fact_ids = [p.fact_id for p in scale_results[0].per_fact]
        for i, fid in enumerate(fact_ids):
            row = f"  {fid:<15} {scale_results[0].per_fact[i].planted_turn:>5}"
            for r in scale_results:
                pr = r.per_fact[i]
                row += f"  {pr.recall_rate:>19.0%}"
            lines.append(row)

    # Summary across all scales
    lines.append(f"\n{'=' * 88}")
    lines.append("  SUMMARY ACROSS SCALES")
    lines.append(f"{'=' * 88}")

    strategy_names = sorted(set(r.strategy for r in results))
    for strat in strategy_names:
        strat_results = [r for r in results if r.strategy == strat]
        avg_recall = statistics.mean(r.overall_recall for r in strat_results)
        avg_ratio = statistics.mean(
            r.avg_prompt_tokens / max(1, r.raw_tokens) for r in strat_results
        )
        total_cost = sum(r.total_input_tokens for r in strat_results)
        lines.append(
            f"  {strat:<25} recall={avg_recall:.0%}  "
            f"prompt/raw={avg_ratio:.1%}  "
            f"total_input={total_cost:,}"
        )

    output = "\n".join(lines)
    print(output)
    return output


# ── Rate-safe comparative summary ────────────────────────────────────

@dataclass
class ComparativeRow:
    scale_name: str
    raw_tokens: int
    full_context_recall: float
    full_context_prompt: float
    full_context_latency_ms: float
    sliding_recall: float
    sliding_prompt: float
    sliding_latency_ms: float
    active_memory_recall: float
    active_memory_prompt: float
    active_memory_latency_ms: float
    active_memory_token_savings_vs_full: float
    active_memory_recall_delta_vs_sliding: float


def _summarize_comparative(results: list[StrategyScaleResult]) -> list[ComparativeRow]:
    grouped: dict[str, dict[str, StrategyScaleResult]] = {}
    for result in results:
        grouped.setdefault(result.scale_name, {})[result.strategy] = result

    rows: list[ComparativeRow] = []
    for scale_name in sorted(grouped):
        bucket = grouped[scale_name]
        full = bucket["full_context"]
        sliding = next(v for k, v in bucket.items() if k.startswith("sliding_window_"))
        active = bucket["active_memory"]
        savings = 1.0 - (active.avg_prompt_tokens / max(1.0, full.avg_prompt_tokens))
        rows.append(
            ComparativeRow(
                scale_name=scale_name,
                raw_tokens=full.raw_tokens,
                full_context_recall=full.overall_recall,
                full_context_prompt=full.avg_prompt_tokens,
                full_context_latency_ms=full.avg_latency_ms,
                sliding_recall=sliding.overall_recall,
                sliding_prompt=sliding.avg_prompt_tokens,
                sliding_latency_ms=sliding.avg_latency_ms,
                active_memory_recall=active.overall_recall,
                active_memory_prompt=active.avg_prompt_tokens,
                active_memory_latency_ms=active.avg_latency_ms,
                active_memory_token_savings_vs_full=savings,
                active_memory_recall_delta_vs_sliding=active.overall_recall - sliding.overall_recall,
            )
        )
    return rows


def _print_comparative_rows(rows: list[ComparativeRow]) -> None:
    print()
    print("=" * 104)
    print("Rate-safe live A/B | full_context vs sliding_window vs active_memory")
    print("=" * 104)
    print(
        f"{'Scale':<12} {'Raw':>8} {'Full R':>7} {'Full Tok':>10} {'Slide R':>8} "
        f"{'Slide Tok':>10} {'AM R':>6} {'AM Tok':>10} {'Save vs Full':>13} {'AM-Full ms':>11}"
    )
    print("-" * 104)
    for row in rows:
        print(
            f"{row.scale_name:<12} {row.raw_tokens:>8,} "
            f"{row.full_context_recall:>6.0%} {row.full_context_prompt:>10,.0f} "
            f"{row.sliding_recall:>7.0%} {row.sliding_prompt:>10,.0f} "
            f"{row.active_memory_recall:>5.0%} {row.active_memory_prompt:>10,.0f} "
            f"{row.active_memory_token_savings_vs_full:>12.0%} "
            f"{(row.active_memory_latency_ms - row.full_context_latency_ms):>11,.0f}"
        )


# ── Budget sweep summary ────────────────────────────────────────────

@dataclass
class SweepRow:
    budget: int
    raw_tokens: int
    full_context_recall: float
    full_context_prompt: float
    sliding_recall: float
    sliding_prompt: float
    active_memory_recall: float
    active_memory_prompt: float
    active_memory_latency_ms: float
    savings_vs_full: float
    recall_gap_vs_full: float
    recall_gain_vs_sliding: float


def _extract_triplet(results: list[StrategyScaleResult]):
    by_name = {r.strategy: r for r in results}
    full = by_name["full_context"]
    sliding = next(v for k, v in by_name.items() if k.startswith("sliding_window_"))
    active = by_name["active_memory"]
    return full, sliding, active


def _print_sweep_rows(rows: list[SweepRow]) -> None:
    print()
    print("=" * 110)
    print("Budget sweep | focus on active_memory recall vs token savings")
    print("=" * 110)
    print(
        f"{'Budget':>8} {'Raw':>8} {'Full R':>7} {'AM R':>6} {'Slide R':>8} "
        f"{'AM Tok':>10} {'Full Tok':>10} {'Save vs Full':>13} {'Gap vs Full':>12} {'AM ms':>8}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row.budget:>8,} {row.raw_tokens:>8,} {row.full_context_recall:>6.0%} "
            f"{row.active_memory_recall:>5.0%} {row.sliding_recall:>7.0%} "
            f"{row.active_memory_prompt:>10,.0f} {row.full_context_prompt:>10,.0f} "
            f"{row.savings_vs_full:>12.0%} {row.recall_gap_vs_full:>11.0%} "
            f"{row.active_memory_latency_ms:>8,.0f}"
        )


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Live A/B benchmark: active-memory vs baselines under long context",
    )
    parser.add_argument(
        "--scales",
        default="100000,250000,500000,1000000",
        help="Comma-separated raw token targets (default: 100K to 1M)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 100K only, 1 trial per probe",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=3,
        help="Trials per probe per strategy (default: 3)",
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="Model provider for live benchmark runs",
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-20250514",
        help="Model to use for probes",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100_000,
        help="Token budget for active-memory strategy (default: 100K)",
    )
    parser.add_argument(
        "--sliding-window",
        type=int,
        default=20,
        help="Window size for sliding-window baseline (default: 20 turns)",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "hash", "openai"],
        default="auto",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--filler-repetitions",
        type=int,
        default=10,
        help="Repetitions per filler turn (higher = faster scale growth)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit machine-readable JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write JSON results to this file",
    )
    parser.add_argument(
        "--rate-safe",
        action="store_true",
        help="Rate-safe mode: use smaller scales (18K-26K) for rate-limited accounts",
    )
    parser.add_argument(
        "--sweep-budgets",
        type=str,
        default=None,
        help="Budget sweep mode: comma-separated AM budgets to test at a single scale (e.g. 10000,12000,14000)",
    )
    parser.add_argument(
        "--sweep-scale",
        type=int,
        default=18000,
        help="Raw token target for budget sweep mode (default: 18000)",
    )
    parser.add_argument(
        "--embed-model",
        default="text-embedding-3-small",
        help="Embedding model when using OpenAI embeddings",
    )
    args = parser.parse_args()

    client = create_model_client(args.provider)

    embedder_spec = create_embedder(
        args.embedder,
        dim=args.embed_dim,
        openai_model=args.embed_model,
        verbose=True,
        stream=sys.stderr,
    )

    # ── Budget sweep mode ──
    if args.sweep_budgets:
        budgets = [int(b.strip()) for b in args.sweep_budgets.split(",") if b.strip()]
        rows: list[SweepRow] = []
        raw_results: list[dict] = []

        for budget in budgets:
            print(f"\n>>> Budget sweep run: {budget:,}", file=sys.stderr)
            results = run_live_ab(
                client,
                scales=[args.sweep_scale],
                trials_per_probe=args.trials,
                model=args.model,
                embedder=embedder_spec.embedder,
                active_memory_budget=budget,
                sliding_window_turns=args.sliding_window,
                filler_repetitions=args.filler_repetitions,
            )
            full, sliding, active = _extract_triplet(results)
            rows.append(
                SweepRow(
                    budget=budget,
                    raw_tokens=full.raw_tokens,
                    full_context_recall=full.overall_recall,
                    full_context_prompt=full.avg_prompt_tokens,
                    sliding_recall=sliding.overall_recall,
                    sliding_prompt=sliding.avg_prompt_tokens,
                    active_memory_recall=active.overall_recall,
                    active_memory_prompt=active.avg_prompt_tokens,
                    active_memory_latency_ms=active.avg_latency_ms,
                    savings_vs_full=1.0 - (active.avg_prompt_tokens / max(1.0, full.avg_prompt_tokens)),
                    recall_gap_vs_full=full.overall_recall - active.overall_recall,
                    recall_gain_vs_sliding=active.overall_recall - sliding.overall_recall,
                )
            )
            raw_results.append({
                "budget": budget,
                "results": [
                    {
                        "strategy": r.strategy,
                        "scale_name": r.scale_name,
                        "raw_tokens": r.raw_tokens,
                        "overall_recall": r.overall_recall,
                        "avg_prompt_tokens": r.avg_prompt_tokens,
                        "avg_latency_ms": r.avg_latency_ms,
                        "total_input_tokens": r.total_input_tokens,
                        "total_output_tokens": r.total_output_tokens,
                    }
                    for r in results
                ],
            })

        _print_sweep_rows(rows)
        if args.json:
            print(json.dumps({"rows": [asdict(r) for r in rows], "runs": raw_results}, indent=2))
        return

    # ── Determine scales ──
    if args.rate_safe:
        scales = [18_000, 22_000, 26_000]
        trials = args.trials
        if args.budget == 100_000:
            # Override default budget for rate-safe mode
            args.budget = 16_000
    elif args.quick:
        scales = [100_000]
        trials = 1
    else:
        scales = [int(s.strip()) for s in args.scales.split(",") if s.strip()]
        trials = args.trials

    mode = "rate-safe" if args.rate_safe else ("quick" if args.quick else "standard")
    print(f"""
  ╔═══════════════════════════════════════════════════════╗
  ║     active-memory LIVE A/B Benchmark                 ║
  ╚═══════════════════════════════════════════════════════╝
  Mode:        {mode}
  Provider:    {args.provider}
  Model:       {args.model}
  Scales:      {', '.join(f'{s:,}' for s in scales)} raw tokens
  Trials:      {trials} per probe per strategy
  Strategies:  full_context, sliding_window_{args.sliding_window}, active_memory
  AM Budget:   {args.budget:,} tokens
  Embedding:   {embedder_spec.description}
""", file=sys.stderr)

    results = run_live_ab(
        client,
        scales=scales,
        trials_per_probe=trials,
        model=args.model,
        embedder=embedder_spec.embedder,
        active_memory_budget=args.budget,
        sliding_window_turns=args.sliding_window,
        filler_repetitions=args.filler_repetitions,
    )

    # ── Rate-safe comparative output ──
    if args.rate_safe:
        comp_rows = _summarize_comparative(results)
        _print_comparative_rows(comp_rows)
        if args.json:
            payload = {
                "results": [
                    {
                        "strategy": r.strategy,
                        "scale_name": r.scale_name,
                        "raw_tokens": r.raw_tokens,
                        "turns": r.turns,
                        "overall_recall": r.overall_recall,
                        "avg_prompt_tokens": r.avg_prompt_tokens,
                        "avg_latency_ms": r.avg_latency_ms,
                        "total_input_tokens": r.total_input_tokens,
                        "total_output_tokens": r.total_output_tokens,
                        "total_api_calls": r.total_api_calls,
                    }
                    for r in results
                ],
                "comparative_rows": [asdict(row) for row in comp_rows],
            }
            print(json.dumps(payload, indent=2))
        return

    # ── Standard output ──
    print_report(results)

    json_data = []
    for r in results:
        json_data.append({
            "strategy": r.strategy,
            "scale": r.scale_name,
            "raw_tokens": r.raw_tokens,
            "turns": r.turns,
            "overall_recall": r.overall_recall,
            "avg_prompt_tokens": r.avg_prompt_tokens,
            "avg_latency_ms": r.avg_latency_ms,
            "total_input_tokens": r.total_input_tokens,
            "total_output_tokens": r.total_output_tokens,
            "total_api_calls": r.total_api_calls,
            "per_fact": [
                {
                    "fact_id": p.fact_id,
                    "planted_turn": p.planted_turn,
                    "recall_rate": p.recall_rate,
                    "avg_prompt_tokens": p.avg_prompt_tokens,
                    "avg_latency_ms": p.avg_latency_ms,
                }
                for p in r.per_fact
            ],
        })

    if args.json:
        print(json.dumps(json_data, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            f.write(json.dumps(json_data, indent=2))
        print(f"\n  Results written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
