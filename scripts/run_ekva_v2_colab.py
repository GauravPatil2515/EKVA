"""Standalone Colab / Cloud GPU Execution Script for EKVA v2.

Run with:
    python3 scripts/run_ekva_v2_colab.py --model qwen1.5-moe-a2.7b
    python3 scripts/run_ekva_v2_colab.py --all-models
"""
import argparse
import os
import sys
import subprocess

# Ensure repo root is always first in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

def main():
    parser = argparse.ArgumentParser(description="EKVA v2 Colab / Cloud GPU Runner")
    parser.add_argument("--model", default="qwen1.5-moe-a2.7b", help="Target MoE model")
    parser.add_argument("--all-models", action="store_true", help="Run all 3 models")
    parser.add_argument("--out-dir", default="output", help="Output directory")
    args = parser.parse_args()

    print("=" * 70)
    print("🚀 EKVA v2: EXPERT-CONDITIONED KV CACHE COMPRESSION BENCHMARK")
    print("=" * 70)

    # 1. Run unit test suite
    print("\n[Step 1/3] Running test suite...")
    res = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-v"], cwd=REPO_ROOT, capture_output=False)
    if res.returncode != 0:
        print("❌ Test suite failed. Fix errors before running benchmark.")
        sys.exit(1)
    print("✅ All unit tests passed!")

    # 2. Run master evaluation harness
    print("\n[Step 2/3] Running master evaluation suite...")
    models = ["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"] if args.all_models else [args.model]
    try:
        from scripts.run_ekva_v2_experiments import run_full_evaluation_pipeline
    except ImportError:
        from run_ekva_v2_experiments import run_full_evaluation_pipeline
    results = run_full_evaluation_pipeline(models=models, out_dir=args.out_dir)

    # 3. Run systems latency & roofline model
    print("\n[Step 3/3] Running analytical systems roofline model...")
    try:
        from experiments.analytical_roofline_model import run_roofline_experiment
    except ImportError:
        from analytical_roofline_model import run_roofline_experiment
    run_roofline_experiment(models=models, out_dir=args.out_dir)

    print("\n" + "=" * 70)
    print("🎉 ALL EKVA v2 EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print(f"📁 Output artifacts and figures saved to: {os.path.abspath(args.out_dir)}")
    print("=" * 70)

if __name__ == "__main__":
    main()
