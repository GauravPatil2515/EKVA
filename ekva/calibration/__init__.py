"""Per-expert signal calibration for EKVA (Phase 1).

Exposes:
  ExpertStats            — running per-expert entropy accumulator
  calibrate_expert_entropy — hook-based calibration pass over a real HF MoE model
  specialization_score   — token-type diversity per expert (used by multi-signal budget)
"""
from ekva.calibration.entropy import (
    ExpertStats,
    calibrate_expert_entropy,
    _get_moe_layers,
    _get_layer_pairs,
    _entropy_from_logits,
)
from ekva.calibration.signals import specialization_score, routing_frequency

__all__ = [
    "ExpertStats",
    "calibrate_expert_entropy",
    "_get_moe_layers",
    "_get_layer_pairs",
    "_entropy_from_logits",
    "specialization_score",
    "routing_frequency",
]
