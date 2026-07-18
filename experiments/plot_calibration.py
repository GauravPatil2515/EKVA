"""Plot per-expert entropy heatmap + entropy-vs-budget scatter from a .pt file.

Usage:
  python experiments/plot_calibration.py --input output/mixtral-8x7b_general_phase1.pt
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    p = Path(args.input)
    if not p.exists():
        print(f"Error: {args.input} not found"); sys.exit(1)
    d = torch.load(p, map_location="cpu")
    emap, budget, meta = d["entropy_map"], d["budget_tensor"], d.get("meta", {})

    out_dir = Path(args.out_dir) if args.out_dir else p.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = torch.stack([emap[e]["avg_entropy"] for e in sorted(emap.keys())])
    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(mat.numpy(), aspect="auto", cmap="plasma")
    fig.colorbar(im, ax=ax, label="Avg Attention Entropy")
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Expert ID")
    ax.set_title(f"Per-Expert Attention Entropy Heatmap ({meta.get('model','Model')})")
    fig.savefig(out_dir / "entropy_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    ent = [emap[e]["avg_entropy"].mean().item() for e in sorted(emap.keys())]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(ent, budget.tolist(), color="darkorange", edgecolors="black", s=100, zorder=3)
    for i, (x, y) in enumerate(zip(ent, budget.tolist())):
        ax.annotate(f"E{i}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_xlabel("Mean Attention Entropy"); ax.set_ylabel("Derived KV Budget (Tokens)")
    ax.set_title(f"EKVA: Expert Entropy vs Allocated Budget ({meta.get('model','Model')})")
    fig.savefig(out_dir / "entropy_vs_budget.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved entropy_heatmap.png, entropy_vs_budget.png in {out_dir}")


if __name__ == "__main__":
    main()
