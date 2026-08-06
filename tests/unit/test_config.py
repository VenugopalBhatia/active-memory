import pytest

from active_memory.config import config_from_dict


def test_typed_config_builds_runtime_policies() -> None:
    config = config_from_dict({"storage": {"path": "/tmp/test.db"}, "retrieval": {"candidate_limit": 12}, "scoring": {"relevance_weight": 0.5, "recency_weight": 0.2, "frequency_weight": 0.1, "affinity_weight": 0.2}})
    assert config.storage.path == "/tmp/test.db"
    assert config.retrieval_config().candidate_limit == 12
    assert config.retrieval_config().final_weights.affinity == 0.2


def test_config_rejects_unknown_fields_and_invalid_weights() -> None:
    with pytest.raises(ValueError, match="unknown"):
        config_from_dict({"storage": {"engine": "tree"}})
    with pytest.raises(ValueError, match="sum"):
        config_from_dict({"scoring": {"relevance_weight": 1.0}})
