"""Download model weights for EKVA experiments.

Usage:
    python scripts/download_weights.py --model qwen1.5-moe-a2.7b
    python scripts/download_weights.py --model mixtral-8x7b
    python scripts/download_weights.py --model deepseek-moe-16b

Weights are cached in ~/.cache/huggingface by default.
Use --offload to save to ./models/ instead.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ekva.models import get_model_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"])
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--offload", action="store_true",
                    help="Save weights to ./models/ instead of HF cache")
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    args = ap.parse_args()

    spec = get_model_spec(args.model)
    print(f"Downloading {spec.hf_id} ({args.model})...")
    print(f"  VRAM estimate (fp16): ~{spec.approx_vram_gb_fp16} GB")
    print(f"  Recommended HW: {spec.recommended_hw}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    kwargs = {"torch_dtype": getattr(__import__("torch"), args.dtype)}
    if args.offload:
        kwargs["device_map"] = "cpu"
        kwargs["offload_folder"] = f"./models/{args.model}"

    print(f"  Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(spec.hf_id)

    print(f"  Loading model weights (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        spec.hf_id,
        **kwargs,
    )

    print(f"  Done! Model loaded on device: {next(model.parameters()).device}")
    print(f"  To use in experiments:")
    print(f"    python experiments/week01_02_calibration.py --model {args.model} --device {args.device}")


if __name__ == "__main__":
    main()
