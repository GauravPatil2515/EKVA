"""Colab entry point for RQ1 + RQ2 (CPU-only, works on free Colab T4).

Usage:
    %run experiments/colab/run_rq1_rq2.py
"""
import os
import sys

# Ensure we're in the right directory
os.chdir("/content/EKVA")
sys.path.insert(0, "/content/EKVA")

print("=== EKVA RQ1 + RQ2 (CPU) ===")
print("Running on:", sys.platform)

# Step 1: Check if calibration data exists
calibration_files = [
    "output/qwen1.5-moe-a2.7b_general_phase1.pt",
    "output/mixtral-8x7b_general_phase1.pt",
    "output/deepseek-moe-16b_general_phase1.pt",
]

missing = [f for f in calibration_files if not os.path.exists(f)]
if missing:
    print(f"WARNING: Missing calibration files: {missing}")
    print("You need to run calibration first on a machine with the model weights.")
    print("Or use mock calibration: python experiments/generate_mock_calibration.py --model mixtral-8x7b")
else:
    print("All calibration files found.")

# Step 2: Run RQ1 (layer vs expert granularity)
print("\n--- RQ1: Layer vs Expert Granularity ---")
os.system("python experiments/rq1_granularity_comparison.py --model qwen1.5-moe-a2.7b --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt --out-dir output")

# Step 3: Run RQ2 (entropy-routing correlation)
print("\n--- RQ2: Entropy-Routing Correlation ---")
os.system("python experiments/rq2_entropy_routing_correlation.py --models qwen1.5-moe-a2.7b mixtral-8x7b deepseek-moe-16b --calibration-dir output --out-dir output")

# Step 4: Generate figures
print("\n--- Visualization ---")
os.system("python experiments/plot_rq1_rq2.py --rq1-results output/rq1_granularity_qwen1.5-moe-a2.7b.pt --rq2-results output/rq2_correlation.pt --out-dir output")

print("\n=== Done ===")
print("Results in output/ directory.")
