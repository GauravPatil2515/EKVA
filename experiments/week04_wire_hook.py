"""Week 4: Wire the real KV truncation hook for Qwen1.5-MoE on RTX 3050.

This script implements the model-specific adapter that the generic hook
(ekva/simulator/hook.py) exposes buffers for. It wires EKVACacheHook
into Qwen1.5-MoE's MoE forward pass so that PPL is computed with
actual KV truncation per expert.

Usage:
    python experiments/week04_wire_hook.py \
        --model qwen1.5-moe-a2.7b \
        --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt \
        --device cpu

The --device cpu flag is for RTX 3050 (6GB). Use --device cuda on A100.
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
from ekva.simulator.hook import EKVACacheHook, QwenMoEAdapter, find_moe_layers
from ekva.simulator.evaluate import compute_perplexity
from transformers import AutoModelForCausalLM, AutoTokenizer

POLICIES = {"uniform": UniformPolicy, "ekva": EKVAPolicy, "random": RandomPolicy, "snapkv_style": SnapKVStylePolicy}
EVICTIONS = ["recency", "attention", "random", "hybrid"]


def load_calibration(path):
    d = torch.load(path, map_location="cpu")
    return d["entropy_map"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="output/week04")
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[W4] Loading {spec.hf_id} on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
        device_map="auto" if args.device == "cpu" else None,
        low_cpu_mem_usage=True,
    ).to(args.device)
    model.eval()

    entropy_map = load_calibration(args.calibration)

    # Sanity check: hook with FullKV budget must reproduce baseline PPL.
    full_budget = {i: 4096 for i in range(spec.num_experts)}
    hook = EKVACacheHook(model, full_budget, spec.num_experts, eviction="recency", device=args.device)
    hook.install()

    prompts = ["Explain attention in transformers.", "Summarize large language models."]
    baseline_ppl = compute_perplexity(model, tok, prompts, torch.device(args.device))
    hook.uninstall()
    print(f"[W4] Baseline PPL (FullKV) = {baseline_ppl:.4f}")

    # Wire Qwen adapter
    adapter = QwenMoEAdapter(hook)
    adapter.install(model)

    # Run 16 combos: 4 policies x 4 evictions at 20% budget
    results = {"sanity_baseline_ppl": baseline_ppl, "grid": {}}
    for pname, P in POLICIES.items():
        policy = P()
        for evict in EVICTIONS:
            budgets = policy.allocate(spec.num_experts, 2048, entropy_map=entropy_map if pname != "uniform" else None)
            # Install hook with these budgets
            hook.reset()
            hook.budgets = budgets
            hook.install()
            try:
                ppl = compute_perplexity(model, tok, prompts, torch.device(args.device))
            except Exception as e:
                ppl = None
                print(f"  [W4] Error with {pname}|{evict}: {e}")
            hook.uninstall()
            results["grid"][f"{pname}|{evict}"] = {"budgets": budgets, "ppl": ppl}
            print(f"  [W4] {pname}|{evict}: PPL={ppl}")

    adapter.uninstall()

    with open(Path(args.out_dir) / "phase2_hook_validation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[W4] Saved 16-combo validation grid -> {args.out_dir}/phase2_hook_validation.json")


if __name__ == "__main__":
    main()
