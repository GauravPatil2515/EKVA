"""Visualization scripts for RQ1 and RQ2 results."""
import argparse
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def plot_rq1_granularity(results_path, out_dir="output"):
    """Plot RQ1: layer vs expert granularity comparison."""
    results = torch.load(results_path, map_location="cpu")
    comparison = results.get("comparison", {})

    if not comparison:
        print("[RQ1 Plot] No comparison data found in results.")
        return

    fractions = sorted(comparison.keys(), key=lambda x: int(x.strip("%")))
    ekva_sums = [comparison[f]["ekva_sum"] for f in fractions]
    layer_sums = [comparison[f]["layer_sum"] for f in fractions]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Budget allocation comparison
    ax1 = axes[0]
    x = np.arange(len(fractions))
    width = 0.35
    ax1.bar(x - width / 2, ekva_sums, width, label="EKVA (expert-level)", color="steelblue")
    ax1.bar(x + width / 2, layer_sums, width, label="Layer-aggregated", color="coral")
    ax1.set_xlabel("Budget Fraction")
    ax1.set_ylabel("Total KV Budget (tokens)")
    ax1.set_title(f"RQ1: Budget Allocation — {results['model']}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(fractions)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    # Per-expert budget distribution (at 40% budget)
    ax2 = axes[1]
    mid_frac = "40%"
    if mid_frac in comparison:
        ekva_b = comparison[mid_frac]["ekva_budgets"]
        layer_b = comparison[mid_frac]["layer_budgets"]
        experts = sorted(ekva_b.keys(), key=lambda x: int(x))
        ekva_vals = [ekva_b[e] for e in experts]
        layer_vals = [layer_b.get(e, 0) for e in experts]
        ax2.scatter(range(len(experts)), ekva_vals, alpha=0.6, label="EKVA", color="steelblue", s=20)
        ax2.scatter(range(len(experts)), layer_vals, alpha=0.6, label="Layer", color="coral", s=20)
        ax2.set_xlabel("Expert ID")
        ax2.set_ylabel("KV Budget (tokens)")
        ax2.set_title(f"RQ1: Per-Expert Budgets at {mid_frac}")
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(out_dir) / "rq1_granularity_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ1 Plot] Saved -> {out_path}")


def plot_rq2_correlation(results_path, out_dir="output"):
    """Plot RQ2: entropy-routing correlation per model."""
    all_results = torch.load(results_path, map_location="cpu")

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, data) in zip(axes, all_results.items()):
        per_expert = data["per_expert"]
        entropies = per_expert["entropies"]
        routings = per_expert["routings"]

        ax.scatter(entropies, routings, alpha=0.7, s=30, color="steelblue")
        ax.set_xlabel("Mean Attention Entropy")
        ax.set_ylabel("Routing Count (tokens)")
        ax.set_title(f"{model_name}\nPearson r={per_expert['pearson_r']:.3f}, Spearman r={per_expert['spearman_r']:.3f}")
        ax.grid(alpha=0.3)

        # Add trend line
        if len(entropies) > 1:
            z = np.polyfit(entropies, routings, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(entropies), max(entropies), 100)
            ax.plot(x_line, p(x_line), "--", color="red", alpha=0.7, linewidth=1)

    plt.tight_layout()
    out_path = Path(out_dir) / "rq2_correlation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ2 Plot] Saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq1-results", required=True, help="Path to RQ1 results .pt file")
    ap.add_argument("--rq2-results", required=True, help="Path to RQ2 results .pt file")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    plot_rq1_granularity(args.rq1_results, args.out_dir)
    plot_rq2_correlation(args.rq2_results, args.out_dir)


if __name__ == "__main__":
    main()
