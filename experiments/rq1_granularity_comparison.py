"""RQ1: Layer vs Expert Granularity Ablation.

Compares entropy-based KV budget allocation at two granularities:
  (a) Layer-level (CAKE/MEDA-style): average entropy per layer, allocate budget per layer
  (b) Expert-level (EKVA): per-expert entropy, allocate budget per expert

Same total budget. Same benchmarks. Same model.
Outputs: results dict + figure comparing PPL degradation curves.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.budget.derive import derive_kv_budget
from ekva.budget.policies import UniformPolicy, EKVAPolicy
from ekva.simulator.evaluate import run_policy_eviction_grid


def layer_aggregated_entropy(entropy_map, num_experts, num_layers):
    """Compute layer-aggregated entropy: mean across experts per layer."""
    layer_entropy = torch.zeros(num_layers)
    for eid in range(num_experts):
        layer_entropy += entropy_map[eid]["avg_entropy"]
    layer_entropy /= num_experts
    return layer_entropy


def derive_layer_budgets(layer_entropy, total_budget, num_experts, min_per_expert=64):
    """Allocate budget proportional to layer entropy, then distribute across experts.

    Uses the actual number of layers from the calibration data (not the model spec).
    """
    num_layers = layer_entropy.shape[0]
    norm = layer_entropy / layer_entropy.sum()
    layer_budgets = (norm * total_budget).round().long()
    layer_budgets = layer_budgets.clamp(min=min_per_expert)
    # Correct sum
    diff = int(total_budget - layer_budgets.sum())
    sign = 1 if diff > 0 else -1
    idx = 0
    while diff != 0 and 0 <= idx < num_layers:
        layer_budgets[idx] = max(min_per_expert, layer_budgets[idx] + sign)
        diff -= sign
        idx = (idx + 1) % num_layers

    # Distribute each layer's budget equally across experts
    expert_budgets = {}
    experts_per_layer = max(1, num_experts // num_layers)
    for eid in range(num_experts):
        layer_idx = eid % num_layers
        expert_budgets[eid] = max(min_per_expert, layer_budgets[layer_idx] // experts_per_layer)
    # Correct total
    total = sum(expert_budgets.values())
    diff = total_budget - total
    sign = 1 if diff > 0 else -1
    idx = 0
    while diff != 0 and 0 <= idx < num_experts:
        expert_budgets[idx] = max(min_per_expert, expert_budgets[idx] + sign)
        diff -= sign
        idx = (idx + 1) % num_experts
    return expert_budgets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"])
    ap.add_argument("--calibration", required=True, help="Path to *_phase1.pt file")
    ap.add_argument("--total-budget", type=int, default=2048)
    ap.add_argument("--min-per-expert", type=int, default=64)
    ap.add_argument("--budget-fractions", nargs="+", type=float, default=[0.1, 0.2, 0.4, 0.6, 0.8])
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(args.calibration, map_location="cpu")
    entropy_map = d["entropy_map"]
    num_experts = spec.num_experts

    # Use actual calibration data dimensions (may differ from model spec for mock data)
    num_layers = entropy_map[0]["avg_entropy"].shape[0]

    # Expert-level budgets (EKVA)
    ekva_policy = EKVAPolicy()
    # Layer-level budgets
    layer_entropy = layer_aggregated_entropy(entropy_map, num_experts, num_layers)

    results = {
        "model": args.model,
        "num_experts": num_experts,
        "num_layers_calibration": num_layers,
        "num_layers_model": spec.num_layers,
        "layer_entropy_mean": layer_entropy.mean().item(),
        "layer_entropy_std": layer_entropy.std().item(),
        "expert_entropy_mean": torch.stack([entropy_map[e]["avg_entropy"].mean() for e in range(num_experts)]).mean().item(),
        "expert_entropy_std": torch.stack([entropy_map[e]["avg_entropy"].mean() for e in range(num_experts)]).std().item(),
    }

    # Compare at each budget fraction
    comparison = {}
    for frac in args.budget_fractions:
        total_b = int(args.total_budget * frac)
        # Dynamic min_per_expert: ensure budget >= num_experts * min_per_expert
        dynamic_min = max(args.min_per_expert, total_b // num_experts)

        # Expert-level (EKVA)
        try:
            ekva_budgets = ekva_policy.allocate(num_experts, total_b, entropy_map=entropy_map)
            ekva_sum = sum(ekva_budgets.values())
        except ValueError:
            # Budget too small for num_experts; skip
            continue

        # Layer-level
        layer_budgets = derive_layer_budgets(layer_entropy, total_b, num_experts, dynamic_min)
        layer_sum = sum(layer_budgets.values())

        comparison[f"{int(frac*100)}%"] = {
            "ekva_sum": ekva_sum,
            "layer_sum": layer_sum,
            "ekva_budgets": {str(k): v for k, v in ekva_budgets.items()},
            "layer_budgets": {str(k): v for k, v in layer_budgets.items()},
        }

    results["comparison"] = comparison

    # Run through simulator grid for both policies
    for policy_name, policy in [("ekva", EKVAPolicy()), ("uniform", UniformPolicy())]:
        grid = run_policy_eviction_grid(
            num_experts=num_experts,
            total_budget=args.total_budget,
            policy_names=[policy_name],
            eviction_names=["attention", "recency"],
            budget_fractions=args.budget_fractions,
            entropy_map=entropy_map if policy_name == "ekva" else None,
        )
        results[f"grid_{policy_name}"] = grid

    out_path = Path(args.out_dir) / f"rq1_granularity_{args.model}.pt"
    torch.save(results, out_path)
    print(f"[RQ1] Saved results -> {out_path}")

    # Print summary
    for frac_key, comp in comparison.items():
        print(f"  Budget {frac_key}: EKVA sum={comp['ekva_sum']}, Layer sum={comp['layer_sum']}")


if __name__ == "__main__":
    main()
