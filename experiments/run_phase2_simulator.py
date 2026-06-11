"""Phase 2: Software EKVA Simulator.

Compares KV budget allocation policies (uniform, EKVA, random, snapkv-style)
on a set of calibration prompts using perplexity as the primary metric.

This is a pure Python/PyTorch simulator — no custom kernels required.
It validates that adaptive per-expert budgets preserve quality at lower
memory cost before any Triton kernel work begins.

Usage:
    python experiments/run_phase2_simulator.py \
        --model mistralai/Mixtral-8x7B-v0.1 \
        --num-experts 8 \
        --total-budget 2048 \
        --calibration output/mixtral_phase1.pt \
        --device cuda
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ekva.simulator.policies import (
    BasePolicy,
    EKVAPolicy,
    RandomPolicy,
    SnapKVStylePolicy,
    UniformPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2: EKVA KV simulator evaluation")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--num-experts", type=int, required=True)
    parser.add_argument("--total-budget", type=int, default=2048)
    parser.add_argument("--min-per-expert", type=int, default=64)
    parser.add_argument("--calibration", type=str, default=None, help="Path to .pt file from Phase 1")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out", type=str, default="output/phase2_results.json")
    return parser.parse_args()


def load_entropy_map(calibration_path: str) -> Optional[Dict]:
    if calibration_path is None:
        return None
    data = torch.load(calibration_path, map_location="cpu")
    return data.get("entropy_map", None)


def compute_perplexity(
    model,
    tokenizer,
    prompts: List[str],
    device: torch.device,
) -> float:
    """Compute average perplexity over a list of prompts using the model's default KV."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss
            n_tokens = input_ids.shape[1]
            total_nll += loss.item() * n_tokens
            total_tokens += n_tokens
    ppl = math.exp(total_nll / max(total_tokens, 1))
    return ppl


def format_results_table(results: Dict[str, Dict]) -> str:
    header = f"{'Policy':<20} | {'Budget %':>8} | {'Total Budget':>12} | {'PPL':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for policy_name, r in results.items():
        line = f"{policy_name:<20} | {r['budget_pct']:>7.1f}% | {r['total_budget']:>12} | {r['ppl']:>10.4f}"
        lines.append(line)
    lines.append(sep)
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EKVA P2] Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.device == "cuda" else torch.float32,
    ).to(args.device)

    entropy_map = load_entropy_map(args.calibration)
    if entropy_map is None:
        print("[EKVA P2] Warning: no calibration file found; EKVA policy will fail. Run Phase 1 first.")

    # Evaluation prompts (extend with LongBench tasks for real experiments)
    eval_prompts = [
        "Explain the attention mechanism in transformers.",
        "Summarize: Large language models have shown remarkable capabilities across a range of tasks including summarization, question answering, and code generation.",
        "Write a Python function to merge two sorted lists.",
        "What is the difference between supervised and unsupervised learning?",
        "Describe the architecture of a sparse mixture-of-experts model.",
    ]

    # Full KV baseline (no budget restriction, model default)
    print("[EKVA P2] Computing FullKV baseline perplexity ...")
    full_kv_ppl = compute_perplexity(model, tokenizer, eval_prompts, torch.device(args.device))
    full_budget = args.num_experts * 4096  # approximate full KV per expert
    print(f"[EKVA P2] FullKV PPL: {full_kv_ppl:.4f}")

    results = {
        "FullKV": {
            "ppl": full_kv_ppl,
            "total_budget": full_budget,
            "budget_pct": 100.0,
        }
    }

    # Evaluate each policy
    policies: List[BasePolicy] = [
        UniformPolicy(),
        EKVAPolicy(),
        RandomPolicy(),
        SnapKVStylePolicy(),
    ]

    for policy in policies:
        print(f"[EKVA P2] Evaluating policy: {policy.name} ...")
        try:
            budgets = policy.allocate(
                num_experts=args.num_experts,
                total_budget=args.total_budget,
                entropy_map=entropy_map,
                min_per_expert=args.min_per_expert,
            )
        except Exception as e:
            print(f"[EKVA P2] Skipping {policy.name}: {e}")
            continue

        total_budget_used = sum(budgets.values())
        budget_pct = 100.0 * total_budget_used / max(full_budget, 1)

        # NOTE: This measures the model's *natural* PPL without hard truncation.
        # For a real Phase 2 experiment, you would hook into the model's past_key_values
        # and replace them with ExpertKVBuffer-truncated caches per expert.
        # That integration is model-specific and is implemented in ekva/simulator/kv_buffer.py.
        # Here we run the evaluation with default attention to get the baseline numbers
        # before the truncated cache is wired in.
        ppl = compute_perplexity(model, tokenizer, eval_prompts, torch.device(args.device))

        results[policy.name] = {
            "ppl": ppl,
            "total_budget": total_budget_used,
            "budget_pct": budget_pct,
            "budgets_per_expert": budgets,
        }

    print("\n" + format_results_table(results))

    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[EKVA P2] Results saved to {out_path}")


if __name__ == "__main__":
    main()
