"""Free-tier Colab script for Weeks 1-4: Real calibration + hook validation.

Runs on FREE Colab T4 (15GB VRAM) with 4-bit quantization. Covers:
  - Week 1: Real entropy + routing calibration on Qwen1.5-MoE-A2.7B
  - Week 2: Cross-model correlation data (uses mock for Mixtral/DeepSeek if too heavy)
  - Week 4: Phase 2 hook validation (FullKV baseline + 16 policy×eviction combos)

Usage in Colab notebook:
    %run experiments/colab/free_tier_w1_w4.py

Requirements:
  - Free Colab T4 GPU (Runtime -> Change runtime type -> T4 GPU)
  - ~15GB disk for model weights
"""
import os
import sys
import subprocess
import torch


def setup():
    """Install deps and ensure repo is present."""
    print("=== Free-Tier EKVA Weeks 1-4 Setup ===")
    
    # Detect if we're in Colab or local
    if os.path.exists("/content"):
        os.chdir("/content")
        REPO_DIR = os.environ.get("EKVA_REPO_DIR", "/content/EKVA")
        if not os.path.exists(REPO_DIR):
            subprocess.run(["git", "clone", "https://github.com/your-org/EKVA.git", REPO_DIR], check=False)
        repo_path = REPO_DIR
    else:
        # Local: find the EKVA repo from current working directory
        current = os.getcwd()
        while current != "/" and not os.path.exists(os.path.join(current, "ekva")):
            current = os.path.dirname(current)
        if os.path.exists(os.path.join(current, "ekva")):
            repo_path = current
        else:
            repo_path = "/home/gaurav/Desktop/gaurav code /Paper/EKVA/EKVA"
    
    os.chdir(repo_path)
    sys.path.insert(0, repo_path)
    
    deps = [
        "transformers>=4.44.0",
        "datasets>=2.18.0",
        "accelerate>=0.33.0",
        "scipy>=1.11.0",
        "bitsandbytes>=0.43.0",
        "peft>=0.10.0",
        "flash-attn>=2.6.0",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", *deps], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"], check=False)
    
    print("Setup complete.")


def week1_calibration():
    """Run real calibration on Qwen1.5-MoE-A2.7B (fits T4 with 4-bit).
    Falls back to mock if GPU not available or torchvision missing."""
    print("\n=== Week 1: Real Calibration (Qwen1.5-MoE-A2.7B, 4-bit) ===")
    cmd = [
        sys.executable, "experiments/week01_02_calibration.py",
        "--model", "qwen1.5-moe-a2.7b",
        "--device", "cuda",
        "--quantize", "4bit",
        "--prompt-sets", "general", "code", "math",
        "--total-budget", "2048",
        "--max-length", "128",
        "--out-dir", "output",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: Real calibration failed (likely no GPU or missing deps).")
        print("Falling back to mock calibration for all models...")
        # Generate mock for Qwen as well
        subprocess.run([
            sys.executable, "experiments/generate_mock_calibration.py",
            "--model", "qwen1.5-moe-a2.7b", "--out-dir", "output",
        ])
        return False

    # Also generate mock calibration for Mixtral + DeepSeek (real weights too heavy for T4)
    print("\n=== Week 1b: Mock calibration for Mixtral + DeepSeek ===")
    for model in ["mixtral-8x7b", "deepseek-moe-16b"]:
        subprocess.run([
            sys.executable, "experiments/generate_mock_calibration.py",
            "--model", model, "--out-dir", "output",
        ])

    # Run RQ1 + RQ2
    print("\n=== RQ1: Layer vs Expert Granularity ===")
    subprocess.run([
        sys.executable, "experiments/rq1_granularity_comparison.py",
        "--model", "qwen1.5-moe-a2.7b",
        "--calibration", "output/qwen1.5-moe-a2.7b_general_phase1.pt",
        "--total-budget", "2048",
        "--budget-fractions", "0.1", "0.2", "0.4", "0.6", "0.8",
        "--out-dir", "output",
    ])

    print("\n=== RQ2: Entropy-Routing Correlation (all 3 models) ===")
    subprocess.run([
        sys.executable, "experiments/rq2_entropy_routing_correlation.py",
        "--models", "qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b",
        "--calibration-dir", "output",
        "--out-dir", "output",
    ])

    print("\n=== Generating figures ===")
    subprocess.run([
        sys.executable, "experiments/plot_rq1_rq2.py",
        "--rq1-results", "output/rq1_granularity_qwen1.5-moe-a2.7b.pt",
        "--rq2-results", "output/rq2_correlation.pt",
        "--out-dir", "output",
    ])

    print("\nWeek 1-2 complete. Check output/ for results.")
    return True


def week4_hook_validation():
    """Validate EKVACacheHook with 4-bit quant on T4."""
    print("\n=== Week 4: Hook Validation (Qwen1.5-MoE-A2.7B, 4-bit) ===")

    cal_path = "output/qwen1.5-moe-a2.7b_general_phase1.pt"
    if not os.path.exists(cal_path):
        print(f"ERROR: Calibration file not found at {cal_path}")
        print("Run week1_calibration() first.")
        return False

    cmd = [
        sys.executable, "experiments/validate_hook.py",
        "--model", "qwen1.5-moe-a2.7b",
        "--calibration", cal_path,
        "--device", "cuda",
        "--quantize", "4bit",
        "--out-dir", "output/week04",
    ]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("WARNING: Hook validation had issues. Check logs above.")
    else:
        print("Week 4 complete. Check output/week04/phase2_hook_validation.json")


def main():
    """Run Weeks 1-4 pipeline. Each step can be run independently."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=["setup", "week1", "week4", "all"], default="all")
    args = ap.parse_args()

    if args.step in ("setup", "all"):
        setup()

    if args.step in ("week1", "all"):
        week1_calibration()

    if args.step in ("week4", "all"):
        week4_hook_validation()

    print("\n=== Free-tier pipeline complete ===")
    print("Results: output/")


if __name__ == "__main__":
    main()