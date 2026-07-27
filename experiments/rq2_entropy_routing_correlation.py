"""RQ2: Entropy-Routing Frequency Correlation Analysis.

Computes Pearson and Spearman correlations between per-expert attention entropy
and routing frequency across all 3 MoE models (Qwen: 60 experts, Mixtral: 8,
DeepSeek: 64). Also computes per-layer correlations.

Outputs: correlation table + multi-panel figure.
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from scipy.stats import pearsonr, spearmanr


def load_calibration(path):
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        d = torch.load(path, map_location="cpu")
    return d["entropy_map"]


def compute_correlations(entropy_map, num_experts):
    """Compute per-expert and per-layer correlations."""
    entropies = []
    routings = []
    for eid in range(num_experts):
        avg_ent = entropy_map[eid]["avg_entropy"].mean().item()
        route_count = entropy_map[eid]["routing_count"].item()
        entropies.append(avg_ent)
        routings.append(route_count)

    ent_tensor = torch.tensor(entropies)
    route_tensor = torch.tensor(routings)

    # Per-expert correlations
    pearson_r, pearson_p = pearsonr(entropies, routings)
    spearman_r, spearman_p = spearmanr(entropies, routings)

    # Per-layer correlations
    num_layers = entropy_map[0]["avg_entropy"].shape[0]
    layer_pearson = []
    layer_spearman = []
    for layer_idx in range(num_layers):
        layer_ent = [entropy_map[eid]["avg_entropy"][layer_idx].item() for eid in range(num_experts)]
        layer_route = [entropy_map[eid]["routing_count"].item() for eid in range(num_experts)]
        if len(set(layer_ent)) > 1 and len(set(layer_route)) > 1:
            pr, _ = pearsonr(layer_ent, layer_route)
            sr, _ = spearmanr(layer_ent, layer_route)
        else:
            pr, sr = float("nan"), float("nan")
        layer_pearson.append(pr)
        layer_spearman.append(sr)

    return {
        "per_expert": {
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
            "entropies": entropies,
            "routings": routings,
        },
        "per_layer": {
            "pearson": layer_pearson,
            "spearman": layer_spearman,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    choices=["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"])
    ap.add_argument("--calibration-dir", default="output",
                    help="Directory containing *_phase1.pt files")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    all_results = {}
    for model_name in args.models:
        spec = get_model_spec(model_name)
        cal_path = Path(args.calibration_dir) / f"{model_name}_phase1.pt"
        if not cal_path.exists():
            cal_path = Path(args.calibration_dir) / f"{model_name}_general_phase1.pt"
        if not cal_path.exists():
            print(f"[RQ2] WARNING: {cal_path} not found, skipping {model_name}")
            continue

        entropy_map = load_calibration(str(cal_path))
        correlations = compute_correlations(entropy_map, spec.num_experts)
        all_results[model_name] = correlations

        print(f"\n[RQ2] {model_name} ({spec.num_experts} experts, {spec.num_layers} layers)")
        print(f"  Per-expert Pearson r={correlations['per_expert']['pearson_r']:.4f} (p={correlations['per_expert']['pearson_p']:.4e})")
        print(f"  Per-expert Spearman r={correlations['per_expert']['spearman_r']:.4f} (p={correlations['per_expert']['spearman_p']:.4e})")
        print(f"  Entropy range: [{min(correlations['per_expert']['entropies']):.4f}, {max(correlations['per_expert']['entropies']):.4f}]")
        print(f"  Routing range: [{min(correlations['per_expert']['routings']):.0f}, {max(correlations['per_expert']['routings']):.0f}]")

        # Per-layer summary
        valid_layers = [(i, correlations['per_layer']['pearson'][i])
                        for i in range(spec.num_layers)
                        if not torch.isnan(torch.tensor(correlations['per_layer']['pearson'][i]))]
        if valid_layers:
            avg_pearson = sum(v for _, v in valid_layers) / len(valid_layers)
            print(f"  Avg per-layer Pearson r={avg_pearson:.4f} (over {len(valid_layers)}/{spec.num_layers} layers)")

    # Save results
    out_path = Path(args.out_dir) / "rq2_correlation.pt"
    torch.save(all_results, out_path)
    print(f"\n[RQ2] Saved correlation results -> {out_path}")


if __name__ == "__main__":
    main()
