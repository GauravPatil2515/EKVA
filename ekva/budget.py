from typing import Dict

import torch


def derive_kv_budget(
    entropy_map: Dict[int, Dict[str, torch.Tensor]],
    total_budget: int,
    min_per_expert: int = 64,
    strategy: str = "proportional",
) -> torch.Tensor:
    """Derive a per-expert KV budget tensor from calibration statistics.

    Args:
        entropy_map: dict[expert_id] -> {"avg_entropy": Tensor[L], "routing_count": Tensor[]}
        total_budget: total KV tokens budget across all experts.
        min_per_expert: minimum KV tokens per expert to avoid starvation.
        strategy: currently only "proportional" is implemented.

    Returns:
        Tensor[num_experts] containing integer KV budgets per expert that sum
        approximately to total_budget.
    """
    num_experts = len(entropy_map)
    expert_ids = sorted(entropy_map.keys())

    # Aggregate a scalar importance score per expert
    importance = []
    for eid in expert_ids:
        stats = entropy_map[eid]
        avg_entropy = stats["avg_entropy"]  # shape [L]
        routing_count = stats["routing_count"].float().clamp_min(1.0)

        # Simple heuristic: entropy * log(routing_count)
        # You can refine this later with roofline or specialization scores.
        score = avg_entropy.mean() * routing_count.log()
        importance.append(score)

    importance_tensor = torch.stack(importance)
    importance_tensor = importance_tensor.clamp_min(0.0)

    if strategy == "proportional":
        if importance_tensor.sum() == 0:
            # Fall back to uniform if all scores are zero
            base = total_budget // num_experts
            budgets = torch.full((num_experts,), base, dtype=torch.int64)
        else:
            weights = importance_tensor / importance_tensor.sum()
            budgets = (weights * float(total_budget)).round().to(torch.int64)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Enforce minimum per expert and adjust to match total_budget
    budgets = torch.clamp(budgets, min=min_per_expert)
    diff = int(total_budget - budgets.sum())
    if diff != 0:
        # Distribute the difference across experts (greedy correction)
        sign = 1 if diff > 0 else -1
        idx = 0
        while diff != 0 and 0 <= idx < num_experts:
            budgets[idx] += sign
            diff -= sign
            idx = (idx + 1) % num_experts

    return budgets
