"""Week 1-2: Real-model Phase 1 calibration (entropy + routing signal).

Replaces the mock calibration with real signal across all 3 candidate models.
Run order per the plan: Qwen1.5-MoE first (fits the 3050 / Colab T4), then
Mixtral-8x7B and DeepSeek-MoE-16B on Colab A100. Varies prompt sets
(general / long_context / code / math) to test prompt-dependence of entropy.

For each model + prompt set, produces output/{model}_{promptset}_phase1.pt and
entropy heatmap + budget scatter (via ekva calibration plotting).

Usage:
  python experiments/week01_02_calibration.py \
      --model mixtral-8x7b --device cuda --prompt-sets general code math
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.calibration.entropy import calibrate_expert_entropy
from ekva.budget.derive import derive_kv_budget
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_prompt_sets(path: str, names):
    cfg = yaml.safe_load(open(path))
    sets = cfg.get("default_prompt_sets", {})
    return {n: sets[n] for n in names if n in sets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--prompt-sets", nargs="+", default=["general", "code", "math"])
    ap.add_argument("--total-budget", type=int, default=2048)
    ap.add_argument("--min-per-expert", type=int, default=64)
    ap.add_argument("--config", default="configs/models.yaml")
    ap.add_argument("--quantize", choices=["4bit", "8bit"], help="Quantization (e.g., 4bit) to prevent OOM")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)
    prompt_sets = load_prompt_sets(args.config, args.prompt_sets)

    print(f"[W1-2] Loading {spec.hf_id} ({args.model}) on {args.device} ...")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)
    
    kwargs = {"torch_dtype": torch.float16 if args.device == "cuda" else torch.float32}
    if args.device == "cuda":
        kwargs["device_map"] = "auto"
        if args.quantize:
            from transformers import BitsAndBytesConfig
            if args.quantize == "4bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif args.quantize == "8bit":
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        
    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **kwargs)
    if args.device != "cuda":
        model = model.to(args.device)

    for pset_name, prompts in prompt_sets.items():
        print(f"[W1-2] Calibrating prompt set '{pset_name}' ({len(prompts)} prompts) ...")
        entropy_map = calibrate_expert_entropy(
            model=model, tokenizer=tok, calibration_prompts=prompts,
            num_experts=spec.num_experts,
        )
        budget = derive_kv_budget(
            entropy_map=entropy_map, total_budget=args.total_budget,
            min_per_expert=args.min_per_expert, strategy="proportional",
        )
        out_path = Path(args.out_dir) / f"{args.model}_{pset_name}_phase1.pt"
        torch.save({"entropy_map": entropy_map, "budget_tensor": budget,
                    "meta": {"model": args.model, "prompt_set": pset_name,
                             "total_budget": args.total_budget}}, out_path)
        print(f"[W1-2] Saved {out_path}  budget_sum={int(budget.sum())}")

    # Decision point (printed, not enforced): does entropy vary across experts?
    print("[W1-2] Decision point: inspect entropy variance across experts in the "
          "saved .pt files. If >=2/3 models show meaningful variance -> proceed.")


if __name__ == "__main__":
    main()
