"""Reproducible offline baselines, ablations, and exact-search latency benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

from active_memory.context import ApproximateTokenCounter
from active_memory.evaluation.ablations import ABLATIONS, scoring_for
from active_memory.evaluation.datasets import EvaluationCase, benchmark_cases
from active_memory.evaluation.retrieval_metrics import measure
from active_memory.ingestion.writer import MemoryWriter
from active_memory.models import Memory, MemoryEdge, MemoryFilters, content_hash, utc_now
from active_memory.retrieval import DeterministicTestEmbeddingProvider, ExactCandidateRetriever, RetrievalConfig, TwoPassRetriever, embed_normalized, score_stage_one
from active_memory.storage.sqlite_store import SQLiteMemoryStore


def _seed_case(store: SQLiteMemoryStore, provider: DeterministicTestEmbeddingProvider, case: EvaluationCase) -> None:
    writer = MemoryWriter(store, ApproximateTokenCounter())
    now = utc_now()
    for fixture in case.memories:
        created = now - timedelta(days=fixture.age_days)
        message = writer.make_message("evaluation", "user", fixture.content, created_at=created, message_id=f"msg_{case.name}_{fixture.id}")
        memory = Memory(fixture.id, "evaluation", "evaluation", message.id, fixture.memory_type, fixture.content,
                        embed_normalized(provider, [fixture.content])[0], created, created, message.token_count,
                        fixture.trust, fixture.status, created, inclusion_count=fixture.inclusions, metadata=dict(fixture.metadata))
        writer.write(message, [memory])
    edges = [MemoryEdge(source, target, edge_type, weight, now) for source, target, edge_type, weight in case.edges]
    store.add_edges(edges)


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: statistics.fmean(row[key] for row in rows) for key in rows[0]} if rows else {}


def run_retrieval_benchmark(k: int = 3) -> dict[str, dict[str, float]]:
    strategies = ["recent_only", "vector_only", "three_signal", *ABLATIONS]
    collected: dict[str, list[dict[str, float]]] = {name: [] for name in strategies}
    provider = DeterministicTestEmbeddingProvider(1024)
    for case in benchmark_cases():
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "eval.db")
            _seed_case(store, provider, case)
            now = utc_now()
            active = store.get_active_memories("evaluation")
            recent_ids = [memory.id for memory in sorted(active, key=lambda item: (item.created_at, item.id), reverse=True)]
            candidates = ExactCandidateRetriever(store, provider).retrieve(case.query, "evaluation", candidate_limit=100)
            vector_ids = [candidate.memory.id for candidate in candidates]
            three_ids = [result.memory.id for result in score_stage_one(candidates, now)]
            ranked: dict[str, list[str]] = {"recent_only": recent_ids, "vector_only": vector_ids, "three_signal": three_ids}
            for name in ABLATIONS:
                scoring, final = scoring_for(name)
                config = RetrievalConfig(candidate_limit=100, seed_limit=3, neighbor_limit=20, result_limit=20, minimum_relevance=0.0, scoring=scoring, final_weights=final)
                ranked[name] = [result.memory.id for result in TwoPassRetriever(store, provider, config).retrieve(case.query, "evaluation", now)]
            stale = {fixture.id for fixture in case.memories if fixture.status != "active"}
            for name, ids in ranked.items():
                collected[name].append(asdict(measure(ids, case.relevant_ids, k=k, stale_ids=stale, superseded_ids=stale)))
            store.close()
    return {name: _aggregate(rows) for name, rows in collected.items()}


def run_latency_benchmark(sizes: tuple[int, ...] = (100, 1000, 5000), trials: int = 5) -> list[dict[str, float | int]]:
    provider = DeterministicTestEmbeddingProvider(128)
    output: list[dict[str, float | int]] = []
    for size in sizes:
        with tempfile.TemporaryDirectory() as directory:
            store = SQLiteMemoryStore(Path(directory) / "latency.db")
            writer = MemoryWriter(store, ApproximateTokenCounter())
            now = utc_now()
            for index in range(size):
                text = f"benchmark record {index} topic {index % 97}"
                message = writer.make_message("bench", "user", text, created_at=now, message_id=f"msg_latency_{index}")
                memory = Memory(f"mem_latency_{index}", "latency", "bench", message.id, "fact", text,
                                embed_normalized(provider, [text])[0], now, now, message.token_count, "user_confirmed")
                writer.write(message, [memory])
            retriever = ExactCandidateRetriever(store, provider)
            samples: list[float] = []
            for _ in range(trials):
                started = time.perf_counter()
                retriever.retrieve("benchmark topic 42", "latency", candidate_limit=80)
                samples.append((time.perf_counter() - started) * 1000)
            output.append({"corpus_size": size, "median_ms": statistics.median(samples), "p95_ms": sorted(samples)[max(0, int(len(samples) * 0.95) - 1)]})
            store.close()
    return output


def run_all() -> dict[str, object]:
    return {"retrieval": run_retrieval_benchmark(), "exact_search_latency": run_latency_benchmark(), "embedding_cost_usd": 0.0, "dataset_cases": len(benchmark_cases())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    results = run_all()
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

