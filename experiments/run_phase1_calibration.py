import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ekva.calibration import calibrate_expert_entropy
from ekva.budget import derive_kv_budget


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Run EKVA expert entropy calibration and derive KV budgets.",
    )
    parser.add_argument("--model", type=str, required=True, help="HF model id, e.g. mistralai/Mixtral-8x7B-v0.1")
    parser.add_argument("--num-experts", type=int, required=True, help="Number of experts per MoE layer")
    parser.add_argument("--total-budget", type=int, default=2048, help="Total KV budget across all experts")
    parser.add_argument("--min-per-expert", type=int, default=64, help="Minimum KV tokens per expert")
    parser.add_argument("--max-new-tokens", type=int, default=0, help="Optional decode length during calibration")
    parser.add_argument("--out", type=str, required=True, help="Output path (.pt or .json) for stats and budgets")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda or cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EKVA] Loading model {args.model} on {args.device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32)
    model.to(args.device)

    # Simple default calibration prompts; replace or extend with task-specific ones.
    calibration_prompts = [
        "Explain the concept of attention in transformers in simple terms.",
        "Summarize the following article about large language models.",
        "Write a Python function to compute the factorial of a number.",
        "Prove that the sum of two even integers is even.",
    ]

    print("[EKVA] Running calibration to collect per-expert entropy and routing stats ...")
    entropy_map = calibrate_expert_entropy(
        model=model,
        tokenizer=tokenizer,
        calibration_prompts=calibration_prompts,
        num_experts=args.num_experts,
        max_new_tokens=args.max_new_tokens,
    )

    print("[EKVA] Deriving KV budgets from calibration statistics ...")
    budget_tensor = derive_kv_budget(
        entropy_map=entropy_map,
        total_budget=args.total_budget,
        min_per_expert=args.min_per_expert,
        strategy="proportional",
    )

    print("[EKVA] Resulting per-expert KV budget:")
    print(budget_tensor)
    print(f"[EKVA] Sum of budgets: {int(budget_tensor.sum())}")

    payload = {
        "model": args.model,
        "num_experts": args.num_experts,
        "total_budget": args.total_budget,
        "min_per_expert": args.min_per_expert,
        "max_new_tokens": args.max_new_tokens,
        "entropy_map_keys": sorted(list(entropy_map.keys())),
        "budget": budget_tensor.cpu().tolist(),
    }

    if out_path.suffix == ".pt":
        torch.save({"entropy_map": entropy_map, "budget_tensor": budget_tensor, "meta": payload}, out_path)
    else:
        # Save a lightweight JSON summary if not using .pt
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    print(f"[EKVA] Saved calibration summary to {out_path}")


if __name__ == "__main__":
    main()
