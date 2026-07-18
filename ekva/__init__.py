"""EKVA — Expert-Aware KV Budget Allocation for Sparse MoE LLM inference.

Top-level package. Submodules:
  ekva.calibration  — per-expert attention-entropy & signal measurement
  ekva.budget       — KV budget derivation + allocation policies
  ekva.simulator    — software KV-cache buffer, eviction, hook, eval harness
  ekva.kernel       — Triton FlashAttention-2 variants (variable tile / fused)
  ekva.profiling    — roofline instrumentation (PyTorch Profiler / Nsight)
  ekva.models       — model registry (HF ids, expert counts, VRAM guidance)
  ekva.benchmarks   — LongBench / RULER / Needle / InfiniteBench / PPL harnesses
"""

__version__ = "0.2.0"

from ekva.calibration import calibrate_expert_entropy, ExpertStats
from ekva.budget import derive_kv_budget
from ekva.budget.policies import (
    BasePolicy,
    UniformPolicy,
    EKVAPolicy,
    EKVAMultiSignalPolicy,
    RandomPolicy,
    SnapKVStylePolicy,
    PyramidKVStylePolicy,
    DynamicKVStylePolicy,
)
from ekva.simulator import ExpertKVBuffer

__all__ = [
    "calibrate_expert_entropy",
    "ExpertStats",
    "derive_kv_budget",
    "BasePolicy",
    "UniformPolicy",
    "EKVAPolicy",
    "EKVAMultiSignalPolicy",
    "RandomPolicy",
    "SnapKVStylePolicy",
    "PyramidKVStylePolicy",
    "DynamicKVStylePolicy",
    "ExpertKVBuffer",
]
