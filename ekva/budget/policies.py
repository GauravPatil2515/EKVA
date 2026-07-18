"""KV budget allocation policies — the full policy axis of the experiment matrix.

Policies (all return Dict[expert_id -> int] KV token budgets):
  UniformPolicy        — every expert gets total_budget // num_experts
  EKVAPolicy           — entropy-only proportional (Phase 1-6 baseline)
  EKVAMultiSignalPolicy— entropy * routing * specialization (Week 7 upgrade)
  RandomPolicy         — random weights (ablation / sanity baseline)
  SnapKVStylePolicy    — uniform budget + attention-score eviction downstream
  PyramidKVStylePolicy — layer-wise pyramid curve (shallow layers larger)
  DynamicKVStylePolicy — per-head adaptive budget (placeholder for reference impl)
"""
from typing import Dict, Optional

import torch

from ekva.budget.derive import derive_kv_budget


class BasePolicy:
    name: str = "base"

    def allocate(
        self,
        num_experts: int,
        total_budget: int,
        entropy_map: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
        min_per_expert: int = 64,
        **kwargs,
    ) -> Dict[int, int]:
        raise NotImplementedError


class UniformPolicy(BasePolicy):
    """Baseline: every expert receives the same KV token budget."""
    name = "uniform"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, **kwargs):
        per_expert = max(min_per_expert, total_budget // num_experts)
        return {i: per_expert for i in range(num_experts)}


class EKVAPolicy(BasePolicy):
    """EKVA: proportional allocation based on per-expert attention entropy + routing."""
    name = "ekva"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, **kwargs):
        if entropy_map is None:
            raise ValueError("EKVAPolicy requires entropy_map from Phase 1 calibration.")
        budget_tensor = derive_kv_budget(
            entropy_map=entropy_map, total_budget=total_budget,
            min_per_expert=min_per_expert, strategy="proportional",
        )
        return {i: int(budget_tensor[i].item()) for i in range(num_experts)}


class EKVAMultiSignalPolicy(BasePolicy):
    """EKVA multi-signal: entropy + routing frequency + specialization (Week 7)."""
    name = "ekva_multi_signal"

    def allocate(
        self, num_experts, total_budget, entropy_map=None, min_per_expert=64,
        specialization: Optional[torch.Tensor] = None, **kwargs,
    ):
        if entropy_map is None:
            raise ValueError("EKVAMultiSignalPolicy requires entropy_map.")
        budget_tensor = derive_kv_budget(
            entropy_map=entropy_map, total_budget=total_budget,
            min_per_expert=min_per_expert, strategy="multi_signal",
            specialization=specialization,
        )
        return {i: int(budget_tensor[i].item()) for i in range(num_experts)}


class RandomPolicy(BasePolicy):
    """Random budget allocation (ablation / sanity-check baseline)."""
    name = "random"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, seed: int = 0, **kwargs):
        g = torch.Generator().manual_seed(seed)
        weights = torch.rand(num_experts, generator=g)
        weights = weights / weights.sum()
        budgets = (weights * total_budget).long().clamp(min=min_per_expert)

        diff = total_budget - int(budgets.sum().item())
        sign = 1 if diff > 0 else -1
        max_iters = num_experts * abs(diff) + 1
        iters, idx = 0, 0
        while diff != 0 and iters < max_iters:
            candidate = budgets[idx % num_experts] + sign
            if candidate >= min_per_expert:
                budgets[idx % num_experts] = candidate
                diff -= sign
            idx += 1
            iters += 1
        return {i: int(budgets[i].item()) for i in range(num_experts)}


class SnapKVStylePolicy(BasePolicy):
    """Uniform budget but select top-B tokens by accumulated attention downstream.

    Software reimplementation of SnapKV's key idea (important-token selection)
    applied at the expert level. Eviction (not budget shape) is where it differs.
    """
    name = "snapkv_style"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, **kwargs):
        per_expert = max(min_per_expert, total_budget // num_experts)
        return {i: per_expert for i in range(num_experts)}


class PyramidKVStylePolicy(BasePolicy):
    """Layer-wise pyramid: shallow layers get larger budgets, deep layers smaller.

    Approximates PyramidKV's "pyramidal information funneling" at the expert
    budget level (one budget per expert, shared across its layer position).
    """
    name = "pyramidkv_style"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, **kwargs):
        # Entropy map keys encode experts; layer position is derived from
        # avg_entropy length when available, else a flat pyramid over experts.
        num_layers = 1
        if entropy_map:
            num_layers = max(e["avg_entropy"].shape[0] for e in entropy_map.values())
        # Pyramid weight: peak at layer 0, decaying toward deeper layers.
        if num_layers > 1:
            w = torch.linspace(1.0, 0.4, num_layers)
        else:
            w = torch.ones(num_experts)
        # Tile across experts (one weight per layer position).
        layer_w = w.repeat_interleave(max(1, num_experts // num_layers))[:num_experts]
        if layer_w.sum() == 0:
            layer_w = torch.ones(num_experts)
        budgets = (layer_w / layer_w.sum() * total_budget).round().long().clamp(min=min_per_expert)
        diff = total_budget - int(budgets.sum().item())
        sign = 1 if diff > 0 else -1
        idx = 0
        while diff != 0 and 0 <= idx < num_experts:
            budgets[idx] = max(min_per_expert, budgets[idx] + sign)
            diff -= sign
            idx = (idx + 1) % num_experts
        return {i: int(budgets[i].item()) for i in range(num_experts)}


class DynamicKVStylePolicy(BasePolicy):
    """Per-head adaptive budget placeholder (reference impl to be wired in Week 5-6).

    Falls back to uniform until a reference DynamicKV implementation is attached.
    """
    name = "dynamickv_style"

    def allocate(self, num_experts, total_budget, entropy_map=None, min_per_expert=64, **kwargs):
        per_expert = max(min_per_expert, total_budget // num_experts)
        return {i: per_expert for i in range(num_experts)}


POLICY_REGISTRY: Dict[str, type] = {
    p.name: p
    for p in (
        UniformPolicy,
        EKVAPolicy,
        EKVAMultiSignalPolicy,
        RandomPolicy,
        SnapKVStylePolicy,
        PyramidKVStylePolicy,
        DynamicKVStylePolicy,
    )
}


def get_policy(name: str) -> BasePolicy:
    if name not in POLICY_REGISTRY:
        raise KeyError(f"Unknown policy '{name}'. Available: {list(POLICY_REGISTRY)}")
    return POLICY_REGISTRY[name]()
