"""Software KV-cache simulator for EKVA Phase 2.

Exposes the per-expert KV buffer + eviction strategies, the past_key_values
interception hook (Week 4), and the evaluation harness used by the benchmark
sweep (Weeks 5-6).
"""
from ekva.simulator.kv_buffer import ExpertKVBuffer
from ekva.simulator.eviction import EVICTION_REGISTRY, get_eviction
from ekva.simulator.hook import EKVACacheHook
from ekva.simulator.dynamic_recalibration import DynamicKVRecalibrationManager
from ekva.simulator.evaluate import (
    compute_perplexity,
    run_policy_eviction_grid,
    format_results_table,
)

__all__ = [
    "ExpertKVBuffer",
    "EVICTION_REGISTRY",
    "get_eviction",
    "EKVACacheHook",
    "DynamicKVRecalibrationManager",
    "compute_perplexity",
    "run_policy_eviction_grid",
    "format_results_table",
]

