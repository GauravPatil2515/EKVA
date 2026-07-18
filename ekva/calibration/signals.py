"""Secondary calibration signals for the multi-signal EKVA budget (Weeks 3, 7).

These are computed from the same calibration pass (or a small extra pass) and
feed `ekva.budget.derive.derive_kv_budget(..., strategy="multi_signal")`.

  routing_frequency  — how often each expert is selected by the router
                       (proxy for how much text the expert "touches")
  specialization_score — token-type diversity *inverse* — experts that attend
                       to a narrow band of token types get a higher score, i.e.
                       they are specialized and may tolerate tighter budgets.
"""
from typing import Dict, List

import torch


def routing_frequency(entropy_map: Dict[int, Dict[str, torch.Tensor]]) -> torch.Tensor:
    """Return a [num_experts] tensor of routing counts (tokens routed per expert)."""
    counts = [entropy_map[eid]["routing_count"].float() for eid in sorted(entropy_map.keys())]
    return torch.stack(counts)


def specialization_score(
    token_type_assignments: Dict[int, torch.Tensor],
    num_experts: int,
) -> torch.Tensor:
    """Compute specialization score per expert from assigned token-type ids.

    Args:
        token_type_assignments: expert_id -> 1D tensor of token-type ids that
            were routed to that expert during calibration. Token types can be
            coarse buckets (e.g. punctuation / number / alpha / whitespace) or
            PoS tags — any discrete vocabulary works.
        num_experts: total number of experts.

    Returns:
        [num_experts] tensor in [0, 1]; higher = more specialized (lower
        diversity of token types handled).
    """
    scores = torch.zeros(num_experts)
    for eid in range(num_experts):
        ids = token_type_assignments.get(eid)
        if ids is None or ids.numel() == 0:
            scores[eid] = 0.0
            continue
        # Shannon evenness of the token-type distribution for this expert.
        counts = torch.bincount(ids.long(), minlength=1).float()
        p = counts / counts.sum()
        p = p[p > 0]
        ent = -(p * p.log()).sum()
        max_ent = torch.log(torch.tensor(p.numel(), dtype=torch.float32))
        evenness = (ent / max_ent) if max_ent > 0 else torch.tensor(0.0)
        # specialization = 1 - evenness (handles one narrow type -> ~1)
        scores[eid] = 1.0 - evenness.item()
    return scores
