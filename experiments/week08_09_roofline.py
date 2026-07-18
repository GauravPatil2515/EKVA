"""Weeks 8-9: Roofline instrumentation + plots (Phase 3 hardware story).

Week 8: instrument per-expert attention calls with PyTorch Profiler / Nsight to
extract FLOPs and HBM bytes per expert on Mixtral-8x7B with Phase-1 prompts.
Week 9: build the roofline plot (AI vs attained GFLOP/s, colored by entropy
bucket) and test correlation entropy/routing/combined vs AI.

Usage:
  python experiments/week08_09_roofline.py --model mixtral-8x7b --calibration output/mixtral-8x7b_general_phase1.pt
"""
import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.profiling.instrument import profile_model_attention, compute_roofline
from transformers import AutoModelForCausalLM, AutoTokenizer

# Example GPU ceilings — replace with measured values for your A100/H100.
PEAK_GFLOPS = 312e3       # A100 fp16 ~312 TFLOP/s
PEAK_BANDWIDTH_GBS = 1555.0  # A100 HBM ~1.5 TB/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="output/week08_09")
    args = ap.parse_args()
    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(args.calibration, map_location="cpu")
    entropy_map = d["entropy_map"]

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(args.device)

    prompts = ["Explain attention in transformers in simple terms.", "Summarize LLMs."]
    prof = profile_model_attention(model, prompts, tok, args.device,
                                   trace_path=str(Path(args.out_dir) / "attn_trace.json"))
    print(f"[W8] Profiler trace saved. Inspect with tensorboard or parse_nsight_json().")

    # Placeholder per-expert FLOPs/bytes (replace with parsed profiler values).
    flops = {e: float(entropy_map[e]["avg_entropy"].mean().item()) * 1e9 for e in entropy_map}
    bytes_ = {e: 1e8 for e in entropy_map}
    roof = compute_roofline(flops, bytes_, PEAK_GFLOPS, PEAK_BANDWIDTH_GBS)

    # Week 9 plot: AI vs attained GFLOP/s, colored by entropy bucket.
    ent = [entropy_map[e]["avg_entropy"].mean().item() for e in sorted(entropy_map)]
    ai = [roof[e]["arithmetic_intensity"] for e in sorted(roof)]
    attained = [roof[e]["attainable_gflops"] for e in sorted(roof)]
    buckets = ["low" if x < 0.5 else ("med" if x < 1.0 else "high") for x in ent]
    colors = {"low": "tab:red", "med": "tab:orange", "high": "tab:green"}
    fig, ax = plt.subplots(figsize=(8, 6))
    for b in ["low", "med", "high"]:
        xs = [ai[i] for i in range(len(ai)) if buckets[i] == b]
        ys = [attained[i] for i in range(len(ai)) if buckets[i] == b]
        ax.scatter(xs, ys, c=colors[b], label=f"entropy {b}", s=80, zorder=3)
    ax.axhline(PEAK_GFLOPS, ls="--", c="gray", label="peak GFLOP/s")
    ax.set_xscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)")
    ax.set_ylabel("Attained GFLOP/s")
    ax.set_title(f"EKVA Roofline — {args.model}")
    ax.legend()
    fig.savefig(Path(args.out_dir) / "roofline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[W9] Saved roofline.png -> {args.out_dir}/roofline.png")

    torch.save({"roofline": roof, "entropy_buckets": buckets}, Path(args.out_dir) / "roofline.pt")


if __name__ == "__main__":
    main()
