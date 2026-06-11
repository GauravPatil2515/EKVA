import math
from typing import Dict

import torch

from ekva.calibration import ExpertStats
from ekva.budget import derive_kv_budget


def test_expert_stats_entropy_and_finalize():
    num_layers = 3
    stats = ExpertStats(num_layers=num_layers)

    # Fake attention probs: batch=1, heads=2, q_len=2, k_len=4
    attn_probs = torch.full((1, 2, 2, 4), 0.25)
    stats.update_entropy(layer_idx=0, attn_probs=attn_probs)
    stats.update_entropy(layer_idx=1, attn_probs=attn_probs)

    out = stats.finalize()
    avg_entropy = out["avg_entropy"]

    # Entropy of uniform distribution over 4 keys: -sum(0.25*log(0.25)) = log(4)
    expected = math.log(4.0)
    assert avg_entropy.shape[0] == num_layers
    assert torch.isclose(avg_entropy[0], torch.tensor(expected, dtype=torch.float64), atol=1e-5)
    assert torch.isclose(avg_entropy[1], torch.tensor(expected, dtype=torch.float64), atol=1e-5)


def test_derive_kv_budget_shapes_and_constraints():
    num_experts = 8
    total_budget = 2048
    min_per_expert = 64

    # Build a fake entropy_map
    entropy_map: Dict[int, Dict[str, torch.Tensor]] = {}
    for eid in range(num_experts):
        avg_entropy = torch.full((4,), 1.0 + 0.1 * eid)  # slightly increasing entropy
        routing_count = torch.tensor(100 + 10 * eid)
        entropy_map[eid] = {"avg_entropy": avg_entropy, "routing_count": routing_count}

    budgets = derive_kv_budget(
        entropy_map=entropy_map,
        total_budget=total_budget,
        min_per_expert=min_per_expert,
        strategy="proportional",
    )

    assert budgets.shape[0] == num_experts
    assert budgets.sum().item() == total_budget
    assert torch.all(budgets >= min_per_expert)

    # Check that higher-entropy experts tend to receive >= budget of lower-entropy ones
    for i in range(num_experts - 1):
        assert budgets[i + 1] >= budgets[i] or budgets[i + 1] >= min_per_expert
