"""Derive a per-expert KV budget tensor from calibration statistics.

Two strategies are supported:
  "proportional"     — entropy-only importance (Weeks 1-6 baseline)
  "multi_signal"     — entropy * (routing weight) * (specialization weight)
                       (Week 7 upgrade; robust fallback if entropy is weak)

The budget tensor always (a) sums to `total_budget` and (b) respects
`min_per_expert` so no expert is starved.
"""
from typing import Dict, Optional

import torch


def _importance_scores(
    entropy_map: Dict[int, Dict[str, torch.Tensor]],
    strategy: str,
    specialization: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    expert_ids = sorted(entropy_map.keys())
    importance = []
    for eid in expert_ids:
        stats = entropy_map[eid]
        avg_entropy = stats["avg_entropy"]  # [L]
        routing = stats["routing_count"].float().clamp_min(1.0)
        ent_mean = avg_entropy.mean().clamp_min(1e-6)
        route_log = routing.log().clamp_min(1e-6)
        spec_val = specialization[eid] if (specialization is not None and eid < len(specialization)) else torch.tensor(0.0)

        if strategy == "entropy_only":
            score = ent_mean
        elif strategy == "routing_only":
            score = route_log
        elif strategy == "specialization_only":
            score = 1.0 + spec_val
        elif strategy == "proportional":
            score = ent_mean * route_log
        elif strategy == "multi_signal":
            score = ent_mean * route_log * (1.0 + spec_val)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        importance.append(score)
    return torch.stack(importance).clamp_min(0.0)


def _distribute(total_budget: int, weights: torch.Tensor, num_experts: int, min_per_expert: int) -> torch.Tensor:
    if weights.sum() == 0:
        base = total_budget // num_experts
        budgets = torch.full((num_experts,), base, dtype=torch.int64)
    else:
        norm = weights / weights.sum()
        budgets = (norm * float(total_budget)).round().to(torch.int64)
    budgets = budgets.clamp(min=min_per_expert)

    # Greedy correction so the sum lands exactly on total_budget.
    diff = int(total_budget - budgets.sum())
    sign = 1 if diff > 0 else -1
    idx = 0
    while diff != 0 and 0 <= idx < num_experts:
        budgets[idx] = max(min_per_expert, budgets[idx] + sign)
        diff -= sign
        idx = (idx + 1) % num_experts
    return budgets


def derive_kv_budget(
    entropy_map: Dict[int, Dict[str, torch.Tensor]],
    total_budget: int,
    min_per_expert: int = 64,
    strategy: str = "proportional",
    specialization: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Derive a per-expert KV budget tensor from calibration statistics.

    Args:
        entropy_map: dict[expert_id] -> {"avg_entropy": Tensor[L], "routing_count": Tensor[]}
        total_budget: total KV tokens budget across all experts.
        min_per_expert: minimum KV tokens per expert to avoid starvation.
        strategy: "proportional" (entropy-only) or "multi_signal" (Week 7).
        specialization: [num_experts] tensor from calibration.specialization_score
            (required when strategy=="multi_signal").

    Returns:
        Tensor[num_experts] integer KV budgets that sum to total_budget.
    """
    num_experts = len(entropy_map)
    if total_budget < num_experts * min_per_expert:
        raise ValueError(
            f"total_budget ({total_budget}) must be >= num_experts*min_per_expert "
            f"({num_experts * min_per_expert})"
        )
    if strategy == "multi_signal" and specialization is None:
        raise ValueError("strategy='multi_signal' requires the `specialization` tensor.")

    weights = _importance_scores(entropy_map, strategy, specialization)
    return _distribute(total_budget, weights, num_experts, min_per_expert)
