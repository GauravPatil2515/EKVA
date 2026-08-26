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
    parser.add_argument(
        "--synthetic-only", action="store_true",
        help="DEBUG ONLY: run the old formula-generated pipeline (no real model weights, "
             "no real generations). Numbers from this mode must never be used in the paper "
             "-- see EKVA_v3_Data_Audit_and_Mechanism.md Task 0. Use this only to smoke-test "
             "plotting/wiring without a GPU.",
    )
    parser.add_argument("--gsm8k-samples", type=int, default=200)
    parser.add_argument("--humaneval-samples", type=int, default=80)
    parser.add_argument("--pg19-docs", type=int, default=30)
    parser.add_argument("--niah-samples", type=int, default=40)
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

    models = ["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"] if args.all_models else [args.model]

    if args.synthetic_only:
        print("\n⚠️  --synthetic-only: results are formula-generated, NOT measured. Do not use for the paper.")
        print("\n[Step 2/3] Running formula-generated smoke-test suite...")
        try:
            from scripts.run_ekva_v2_experiments import run_full_evaluation_pipeline
        except ImportError:
            from run_ekva_v2_experiments import run_full_evaluation_pipeline
        run_full_evaluation_pipeline(models=models, out_dir=args.out_dir)
    else:
        print("\n[Step 2/3] Running REAL evaluation suite (real weights, real generations, real scoring)...")
        try:
            from scripts.run_real_evaluation_suite import main as run_real_main
        except ImportError:
            from run_real_evaluation_suite import main as run_real_main
        for m in models:
            sys.argv = [
                "run_real_evaluation_suite.py", "--model", m, "--out-dir", args.out_dir,
                "--gsm8k-samples", str(args.gsm8k_samples),
                "--humaneval-samples", str(args.humaneval_samples),
                "--pg19-docs", str(args.pg19_docs),
                "--niah-samples", str(args.niah_samples),
            ]
            run_real_main()

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
