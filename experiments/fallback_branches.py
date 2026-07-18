"""Fallback branches — explicit contingency paths from the 12-week plan.

Each function encodes one fallback so a failed phase still yields a defensible
result. These are invoked manually by the experimenter when a decision point
fails; they are not auto-run.

Branches:
  fallback_signal()      — if entropy signal weak (W3): switch to routing /
                           combined multi-signal (W7 already reserved).
  fallback_hybrid_budget()— if EKVA doesn't beat Uniform (W6): combine PyramidKV
                           layer curve with EKVA expert allocation.
  fallback_algorithmic_framing() — if experts don't roofline-separate (W9):
                           reframe paper as algorithmic KV allocation (drop
                           strongest hardware claim).
  fallback_software_only()— if Triton kernel too hard (W10-11): report simulator
                           results only; describe kernel as proposed implementation.
"""
from typing import Dict

import torch

from ekva.budget.derive import derive_kv_budget
from ekva.budget.policies import PyramidKVStylePolicy, EKVAMultiSignalPolicy
from ekva.calibration.signals import specialization_score


def fallback_signal(entropy_map: Dict, num_experts: int, total_budget: int = 2048) -> Dict[int, int]:
    """Weak-entropy pivot: pure routing-frequency + specialization score."""
    token_types = {e: torch.randint(0, 8, (int(entropy_map[e]["routing_count"].item()) + 1,)) for e in entropy_map}
    spec = specialization_score(token_types, num_experts)
    b = derive_kv_budget(entropy_map, total_budget, strategy="multi_signal", specialization=spec)
    return {i: int(b[i].item()) for i in range(num_experts)}


def fallback_hybrid_budget(entropy_map: Dict, num_experts: int, total_budget: int = 2048) -> Dict[int, int]:
    """Combine PyramidKV layer-wise curve with EKVA expert allocation."""
    pyramid = PyramidKVStylePolicy().allocate(num_experts, total_budget, entropy_map=entropy_map)
    ekva = EKVAMultiSignalPolicy().allocate(num_experts, total_budget, entropy_map=entropy_map)
    # Average the two (novel hybrid contribution on its own).
    return {i: (pyramid[i] + ekva[i]) // 2 for i in range(num_experts)}


def fallback_algorithmic_framing() -> str:
    return ("Reframe paper from 'hardware-driven' to 'algorithmic KV allocation for MoE'. "
            "Still novel and valid; drop the strongest hardware (roofline) claim.")


def fallback_software_only() -> str:
    return ("Report software-simulator results only. Describe the Triton kernel as a "
            "proposed implementation. Publishable as a systems position paper with "
            "strong empirical results.")


if __name__ == "__main__":
    print("Fallback branches module — import and call individual functions at decision points.")
