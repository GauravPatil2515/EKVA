"""Weeks 5-6: Full Phase 2 benchmark sweep (the main results table).

Runs the 16-combination grid (4 policies x 4 evictions) across budget fractions
{10,20,40,60,80}% on LongBench / RULER / Needle, for Mixtral-8x7B and
DeepSeek-MoE-16B. Records PPL / task accuracy / actual memory. Produces the
master results CSV: Method | Memory% | PPL | Throughput.

Usage:
  python experiments/week05_06_benchmark_sweep.py --model mixtral-8x7b --calibration output/mixtral-8x7b_general_phase1.pt
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.budget.policies import POLICY_REGISTRY, get_policy
from ekva.simulator.eviction import EVICTION_REGISTRY
from ekva.simulator.evaluate import run_policy_eviction_grid
from transformers import AutoModelForCausalLM, AutoTokenizer

BUDGET_FRACTIONS = [0.1, 0.2, 0.4, 0.6, 0.8]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--total-budget", type=int, default=4096)
    ap.add_argument("--out", default="output/week05_06/master_results.csv")
    args = ap.parse_args()
    spec = get_model_spec(args.model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    entropy_map = torch.load(args.calibration, map_location="cpu")["entropy_map"]
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(args.device)

    rows = []
    # FullKV baseline row.
    rows.append({"method": "FullKV", "memory_pct": 100.0, "ppl": None, "throughput": None,
                 "policy": "full", "eviction": "-", "frac": 1.0})

    # score_fn: real truncated-PPL via EKVACacheHook (wired in Week 4). Placeholder here.
    def score_fn(pname, evict, frac, budgets):
        return None  # TODO: return compute_perplexity with hook truncation

    grid = run_policy_eviction_grid(
        num_experts=spec.num_experts, total_budget=args.total_budget,
        entropy_map=entropy_map, score_fn=score_fn,
        budget_fractions=BUDGET_FRACTIONS,
    )
    for key, entry in grid.items():
        pname, evict, frac_s = key.split("|")
        frac = float(frac_s.strip("%")) / 100.0
        mem_pct = 100.0 * sum(entry["budgets"].values()) / (spec.num_experts * args.total_budget)
        rows.append({"method": f"EKVA-{pname}", "memory_pct": round(mem_pct, 1),
                     "ppl": entry.get("metric"), "throughput": None,
                     "policy": pname, "eviction": evict, "frac": frac})

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "memory_pct", "ppl", "throughput", "policy", "eviction", "frac"])
        w.writeheader()
        w.writerows(rows)
    print(f"[W5-6] Wrote master results -> {args.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
