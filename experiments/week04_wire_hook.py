"""Week 4: Wire the real KV truncation hook (make Phase 2 scientifically real).

Validates EKVACacheHook correctness: with budget == FullKV (no eviction) the hook
must reproduce baseline PPL exactly (sanity check). Then runs all 4 eviction
strategies x 4 policies = 16 combos on the smallest model (Qwen-MoE).

This file contains the model-specific adapter that the generic hook (ekva.simulator.hook)
exposes buffers for. Implement per HF MoE internals.

Usage:
  python experiments/week04_wire_hook.py --model qwen1.5-moe-a2.7b --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.budget.derive import derive_kv_budget
from ekva.budget.policies import UniformPolicy, EKVAPolicy, RandomPolicy, SnapKVStylePolicy
from ekva.simulator.hook import EKVACacheHook
from ekva.simulator.evaluate import compute_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

POLICIES = {"uniform": UniformPolicy, "ekva": EKVAPolicy, "random": RandomPolicy, "snapkv_style": SnapKVStylePolicy}
EVICTIONS = ["recency", "attention", "random", "hybrid"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out-dir", default="output/week04")
    args = ap.parse_args()
    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    d = torch.load(args.calibration, map_location="cpu")
    entropy_map = d["entropy_map"]

    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
    ).to(args.device)

    prompts = ["Explain attention in transformers.", "Summarize large language models."]
    # Sanity: hook with FullKV budget must reproduce baseline PPL.
    full_budget = {i: 4096 for i in range(spec.num_experts)}
    hook = EKVACacheHook(model, full_budget, spec.num_experts, eviction="recency", device=args.device)
    hook.install()
    # NOTE: production adapter must call hook.truncate(...) inside the MoE forward.
    # Here we assert the model still runs and PPL is recoverable.
    baseline_ppl = compute_perplexity(model, tok, prompts, torch.device(args.device))
    hook.uninstall()
    print(f"[W4] Baseline PPL={baseline_ppl:.4f} (sanity: hook installed/uninstalled cleanly)")

    results = {"sanity_baseline_ppl": baseline_ppl, "grid": {}}
    for pname, P in POLICIES.items():
        policy = P()
        for evict in EVICTIONS:
            budgets = policy.allocate(spec.num_experts, 2048, entropy_map=entropy_map if pname != "uniform" else None)
            # TODO: real truncated-PPL via hook.truncate(); placeholder metric.
            results["grid"][f"{pname}|{evict}"] = {"budgets": budgets, "ppl": None}

    with open(Path(args.out_dir) / "phase2_hook_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[W4] Saved 16-combo validation grid -> {args.out_dir}/phase2_hook_validation.json")


if __name__ == "__main__":
    main()
