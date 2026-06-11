"""Budget allocation policies for Phase 2 simulator.

Each policy takes per-expert calibration statistics and returns a
Dict[expert_id, int] mapping expert ids to KV token budgets.
"""
from typing import Dict, Optional

import torch

from ekva.budget import derive_kv_budget


class BasePolicy:
    """Base class for KV budget allocation policies."""

    name: str = "base"

    def allocate(
        self,
        num_experts: int,
        total_budget: int,
        entropy_map: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        min_per_expert: int = 64,
    ) -> Dict[int, int]:
        raise NotImplementedError


class UniformPolicy(BasePolicy):
    """Baseline: every expert receives the same KV token budget."""

    name = "uniform"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64):
        per_expert = max(min_per_expert, total_budget // num_experts)
        return {i: per_expert for i in range(num_experts)}


class EKVAPolicy(BasePolicy):
    """EKVA: proportional allocation based on per-expert attention entropy
    and routing frequency (the Phase 1 calibration heuristic)."""

    name = "ekva"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64):
        if entropy_map is None:
            raise ValueError("EKVAPolicy requires entropy_map from Phase 1 calibration.")
        budget_tensor = derive_kv_budget(
            entropy_map=entropy_map,
            total_budget=total_budget,
            min_per_expert=min_per_expert,
            strategy="proportional",
        )
        return {i: int(budget_tensor[i].item()) for i in range(num_experts)}


class RandomPolicy(BasePolicy):
    """Random budget allocation (ablation / sanity check baseline)."""

    name = "random"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64):
        weights = torch.rand(num_experts)
        weights = weights / weights.sum()
        budgets = (weights * total_budget).long()
        budgets = budgets.clamp(min=min_per_expert)
        # Greedy correction to hit total_budget exactly
        diff = total_budget - int(budgets.sum().item())
        sign = 1 if diff > 0 else -1
        idx = 0
        while diff != 0:
            budgets[idx % num_experts] += sign
            diff -= sign
            idx += 1
        return {i: int(budgets[i].item()) for i in range(num_experts)}


class SnapKVStylePolicy(BasePolicy):
    """Uniform budget but select top-B tokens by accumulated attention (SnapKV-style).

    This is a software reimplementation of SnapKV's key idea (select important
    tokens per head) applied at the expert level as a comparison point.
    Unlike full SnapKV this does not do per-head token selection; it applies a
    uniform token importance selection across heads for simplicity.
    """

    name = "snapkv_style"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64):
        # SnapKV-style uses uniform budget but smarter eviction (handled in ExpertKVBuffer)
        per_expert = max(min_per_expert, total_budget // num_experts)
        return {i: per_expert for i in range(num_experts)}
