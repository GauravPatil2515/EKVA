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

    print(f"  Loading tokenizer...")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec.hf_id)

    print(f"  Downloading model weights (this may take a while)...")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=spec.hf_id, local_dir=f"./models/{args.model}" if args.offload else None)

    print(f"  Done! Model downloaded.")
    print(f"  To use in experiments:")
    print(f"    python experiments/week01_02_calibration.py --model {args.model} --device {args.device}")


if __name__ == "__main__":
    main()
