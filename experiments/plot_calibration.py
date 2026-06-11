import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize EKVA expert attention entropy and derived KV budgets.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the calibration output .pt file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save the plots. Defaults to the directory of the input file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"[EKVA] Loading calibration data from {input_path} ...")
    d = torch.load(input_path, map_location="cpu")

    entropy_map = d["entropy_map"]
    budget_tensor = d["budget_tensor"]
    meta = d.get("meta", {})

    print("\n" + "=" * 50)
    print(" EXPERT ENTROPY & ROUTING STATISTICS")
    print("=" * 50)
    for eid, stats in sorted(entropy_map.items()):
        mean_entropy = stats["avg_entropy"].mean().item()
        routing_count = stats["routing_count"].item()
        print(f"Expert {eid:2d}: mean_entropy = {mean_entropy:.4f}, routed_tokens = {routing_count}")
    print("-" * 50)
    print("Derived Budget Tensor:", budget_tensor.tolist())
    print("=" * 50 + "\n")

    # Determine output directory
    out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Plot entropy heatmap (Figure 1)
    # Build matrix [num_experts x num_layers]
    mat = torch.stack([entropy_map[eid]["avg_entropy"] for eid in sorted(entropy_map.keys())])
    
    plt.figure(figsize=(12, 5))
    im = plt.imshow(mat.numpy(), aspect="auto", cmap="plasma")
    plt.colorbar(im, label="Avg Attention Entropy")
    plt.xlabel("Layer Index")
    plt.ylabel("Expert ID")
    plt.title(f"Per-Expert Attention Entropy Heatmap ({meta.get('model', 'Model')})")
    
    heatmap_path = out_dir / "entropy_heatmap.png"
    plt.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EKVA] Saved Entropy Heatmap to {heatmap_path}")

    # 2. Plot budget vs routing / entropy scatter (Figure 2)
    budgets = budget_tensor.tolist()
    entropies = [entropy_map[eid]["avg_entropy"].mean().item() for eid in sorted(entropy_map.keys())]

    plt.figure(figsize=(8, 6))
    plt.scatter(entropies, budgets, color="darkorange", edgecolors="black", s=100, zorder=3)
    
    for i, (x, y) in enumerate(zip(entropies, budgets)):
        plt.annotate(
            f"E{i}",
            (x, y),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontweight="bold",
        )
        
    plt.grid(True, linestyle="--", alpha=0.5, zorder=0)
    plt.xlabel("Mean Attention Entropy")
    plt.ylabel("Derived KV Budget (Tokens)")
    plt.title(f"EKVA: Expert Entropy vs Allocated Budget ({meta.get('model', 'Model')})")
    
    scatter_path = out_dir / "entropy_vs_budget.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EKVA] Saved Entropy vs Budget Scatter plot to {scatter_path}")


if __name__ == "__main__":
    main()
