from active_memory.evaluation.datasets import benchmark_cases
from active_memory.evaluation.end_to_end import run_retrieval_benchmark
from active_memory.evaluation.retrieval_metrics import measure


def test_dataset_covers_required_categories() -> None:
    categories = {case.category for case in benchmark_cases()}
    assert len(categories) == 12
    assert "changing decisions" in categories
    assert "privacy-sensitive content" in categories


def test_retrieval_metrics_known_ranking() -> None:
    metrics = measure(["noise", "answer"], {"answer"}, k=2)
    assert metrics.recall_at_k == 1.0
    assert metrics.precision_at_k == 0.5
    assert metrics.mrr == 0.5
    assert 0.0 < metrics.ndcg_at_k < 1.0


def test_benchmark_exposes_all_baselines_and_ablations() -> None:
    results = run_retrieval_benchmark()
    assert {"recent_only", "vector_only", "three_signal", "full", "minus_affinity", "minus_frequency", "minus_recency"} <= set(results)
    assert all(0.0 <= row["recall_at_k"] <= 1.0 for row in results.values())
