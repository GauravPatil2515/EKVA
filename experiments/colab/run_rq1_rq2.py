"""Colab entry point for RQ1 + RQ2 — FREE TIER compatible (T4, no A100).

Usage in Colab notebook:
    %run experiments/colab/run_rq1_rq2.py

This is the recommended free-tier entry point. It runs:
  1. Real calibration on Qwen1.5-MoE-A2.7B (if model weights available on T4)
  2. Mock calibration for Mixtral + DeepSeek (synthetic, no GPU needed)
  3. RQ1: Layer vs Expert Granularity comparison
  4. RQ2: Entropy-Routing correlation (all 3 models)
  5. Generates figures (PNG) + saves results

If real calibration failed or wasn't run, falls back to fully synthetic data.
"""
import os
import sys
import subprocess

# Ensure we're in the right directory
if os.path.exists("/content/EKVA"):
    os.chdir("/content/EKVA")
else:
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

sys.path.insert(0, ".")

print("=== EKVA RQ1 + RQ2 (Free Tier — Colab T4) ===")
print("Running on:", sys.platform)

# Step 1: Check if real calibration data exists
calibration_files = [
    ("qwen1.5-moe-a2.7b", "output/qwen1.5-moe-a2.7b_general_phase1.pt"),
    ("mixtral-8x7b", "output/mixtral-8x7b_phase1.pt"),
    ("deepseek-moe-16b", "output/deepseek-moe-16b_phase1.pt"),
]

existing_cal = {}
for model_name, path in calibration_files:
    if os.path.exists(path):
        existing_cal[model_name] = path
        print(f"  Found calibration: {path}")
    else:
        print(f"  Missing: {path}")

missing = [m for m, _ in calibration_files if m not in existing_cal]

if missing:
    print(f"\nWARNING: Missing calibration for: {missing}")
    print("Generating synthetic calibration data for missing models...")

    for model_name in missing:
        subprocess.run([
            sys.executable, "experiments/generate_mock_calibration.py",
            "--model", model_name, "--out-dir", "output",
        ])
        existing_cal[model_name] = f"output/{model_name}_phase1.pt"

    # Also generate for Qwen if missing
    if "qwen1.5-moe-a2.7b" in missing:
        existing_cal["qwen1.5-moe-a2.7b"] = "output/qwen1.5-moe-a2.7b_phase1.pt"
else:
    print("All calibration files found.")

# Step 2: Use the local simulator pipeline (handles all 3 models with synthetic fallback)
print("\n--- Running full simulator pipeline (synthetic quality estimates) ---")
subprocess.run([
    sys.executable, "experiments/local_simulator_pipeline.py",
    "--total-budget", "2048",
    "--out-dir", "output",
])

print("\n=== Done ===")
print("Results in output/ directory:")
for f in [
    "local_rq1_all_models.pt",
    "local_rq2_all_models.pt",
    "local_rq3_grid.json",
    "local_rq1.png",
    "local_rq2.png",
    "local_rq3.png",
]:
    path = os.path.join("output", f)
    status = "exists" if os.path.exists(path) else "NOT FOUND"
    print(f"  {f}: {status}")
