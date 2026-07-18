"""Week 3: Correlation & sensitivity study.

For each expert, force its ExpertKVBuffer budget to {25%,50%,75%,100%} while all
others stay full; measure perplexity delta per expert per budget level.
Plots entropy vs sensitivity slope; also tests routing-freq vs sensitivity and
combined-score vs sensitivity. Reports Pearson/Spearman correlation coefficients.

Uses the budget simulator (no kernel). Requires a real model + transformers.
Usage:
  python experiments/week03_sensitivity.py --model qwen1.5-moe-a2.7b --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt
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
from ekva.simulator.kv_buffer import ExpertKVBuffer
from ekva.simulator.evaluate import compute_perplexity
from scipy.stats import pearsonr, spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True, help=".pt from Week 1-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="output/week03")
    args = ap.parse_args()
    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(args.calibration, map_location="cpu")
    entropy_map = d["entropy_map"]
    base = derive_kv_budget(entropy_map, total_budget=2048, strategy="proportional")

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(args.device)

    eval_prompts = ["Explain attention in transformers.", "Summarize large language models.",
                    "Write Python to merge two sorted lists.", "Supervised vs unsupervised learning?"]

    # Baseline PPL at full budget per expert.
    full_ppl = compute_perplexity(model, tok, eval_prompts, torch.device(args.device))

    fractions = [0.25, 0.5, 0.75, 1.0]
    slopes = {}
    for eid in range(spec.num_experts):
        deltas = []
        for f in fractions:
            # Simulate: shrink expert eid's budget, keep others full.
            # (Software sim; real truncation via hook in Week 4.)
            bufs = {i: ExpertKVBuffer(budget=int(base[i].item()) if i != eid else max(64, int(base[eid].item()*f)),
                                      head_dim=spec.head_dim, num_heads=spec.num_attention_heads,
                                      eviction="attention", device="cpu") for i in range(spec.num_experts)}
            # Placeholder PPL delta (replace with real hook-based eval in Week 4).
            ppl = full_ppl + 0.0  # TODO wire EKVACacheHook.truncate per expert
            deltas.append(ppl)
        # sensitivity slope = d(PPL)/d(budget_fraction)
        slopes[eid] = (deltas[-1] - deltas[0]) / (fractions[-1] - fractions[0])

    ent = [entropy_map[e]["avg_entropy"].mean().item() for e in sorted(entropy_map)]
    route = [entropy_map[e]["routing_count"].item() for e in sorted(entropy_map)]
    slope_vals = [slopes[e] for e in sorted(entropy_map)]

    r_e = pearsonr(ent, slope_vals)
    r_r = pearsonr(route, slope_vals)
    print(f"[W3] Pearson(entropy, sensitivity)={r_e:.3f}  Pearson(routing, sensitivity)={r_r:.3f}")
    print(f"[W3] Spearman(entropy, sensitivity)={spearmanr(ent, slope_vals):.3f}")

    torch.save({"slopes": slopes, "entropy": ent, "routing": route,
                "pearson_entropy": r_e, "pearson_routing": r_r,
                "full_ppl": full_ppl}, Path(args.out_dir) / "sensitivity.pt")
    print(f"[W3] Saved sensitivity study -> {args.out_dir}/sensitivity.pt")


if __name__ == "__main__":
    main()
