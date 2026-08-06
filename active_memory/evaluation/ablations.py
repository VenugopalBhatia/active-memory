"""Retrieval baseline and ablation definitions."""

from __future__ import annotations

from active_memory.retrieval.retriever import FinalWeights
from active_memory.retrieval.scoring import ScoringPolicy, StageOneWeights


ABLATIONS = {
    "full": (StageOneWeights(), FinalWeights()),
    "minus_affinity": (StageOneWeights(), FinalWeights(0.6470588235, 0.2352941176, 0.1176470589, 0.0)),
    "minus_frequency": (StageOneWeights(0.7222222222, 0.2777777778, 0.0), FinalWeights(0.6111111111, 0.2222222222, 0.0, 0.1666666667)),
    "minus_recency": (StageOneWeights(0.8666666667, 0.0, 0.1333333333), FinalWeights(0.6875, 0.0, 0.125, 0.1875)),
}


def scoring_for(name: str) -> tuple[ScoringPolicy, FinalWeights]:
    stage_one, final = ABLATIONS[name]
    return ScoringPolicy(stage_one=stage_one), final

