"""Free-tier Colab notebook entry point for EKVA (single-cell, copy into Colab).

Copy-paste this entire file into a Colab code cell and run:
    Runtime -> Run all

This runs the full free-tier pipeline:
  Phase 0: Install deps + clone repo (if not already in /content/EKVA)
  Phase 1: Real calibration on Qwen1.5-MoE-A2.7B (4-bit, fits T4)
  Phase 2: Mock calibration for Mixtral + DeepSeek (synthetic, too heavy for T4)
  Phase 3: Generate local simulator results (all 3 models, no GPU needed)
  Phase 4: Upload all results to Google Drive (optional)

Total compute: ~3-4 hours on free Colab T4 GPU.
"""
import os, sys, subprocess, json, torch

# ── Phase 0: Setup ────────────────────────────────────────────────────────
def install_and_setup():
    print("=== Phase 0: Setup ===")
    
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
        "numpy>=1.26.4",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", *deps], check=False)
    subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".", "--no-deps", "-q"], check=False)

    # Install matplotlib + numpy for plotting
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"], check=False)
    print("Setup complete.")

install_and_setup()

# ── Phase 1: Real calibration on Qwen1.5-MoE-A2.7B (free Colab T4) ─────────
print("\n=== Phase 1: Real Calibration (Qwen1.5-MoE-A2.7B, 4-bit) ===")
cal_result = subprocess.run([
    sys.executable, "experiments/week01_02_calibration.py",
    "--model", "qwen1.5-moe-a2.7b",
    "--device", "cuda",
    "--quantize", "4bit",
    "--prompt-sets", "general", "code", "math",
    "--total-budget", "2048",
    "--max-length", "128",
    "--out-dir", "output",
])
if cal_result.returncode != 0:
    print("WARNING: Real calibration failed. Falling back to local simulator.")
else:
    print("Real calibration complete.")

# ── Phase 2: Mock calibration for Mixtral + DeepSeek ──────────────────────
print("\n=== Phase 2: Mock calibration for Mixtral + DeepSeek ===")
for model in ["mixtral-8x7b", "deepseek-moe-16b"]:
    subprocess.run([
        sys.executable, "experiments/generate_mock_calibration.py",
        "--model", model, "--out-dir", "output",
    ])

# ── Phase 3: Full simulator pipeline (all 3 models, no GPU) ─────────────────
print("\n=== Phase 3: Full simulator pipeline ===")
subprocess.run([
    sys.executable, "experiments/local_simulator_pipeline.py",
    "--total-budget", "2048",
    "--out-dir", "output",
])

# ── Phase 4: RQ1 + RQ2 plots via existing scripts ───────────────────────────
print("\n=== Phase 4: Generating RQ1/RQ2 figures ===")
subprocess.run([
    sys.executable, "experiments/plot_rq1_rq2.py",
    "--rq1-results", "output/local_rq1_all_models.pt",
    "--rq2-results", "output/local_rq2_all_models.pt",
    "--out-dir", "output",
])

# ── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("FREE-TIER PIPELINE COMPLETE")
print("=" * 70)
print("\nAll outputs in /content/EKVA/output/:")
for f in [
    "qwen1.5-moe-a2.7b_general_phase1.pt",
    "mixtral-8x7b_phase1.pt",
    "deepseek-moe-16b_phase1.pt",
    "local_rq1_all_models.pt",
    "local_rq2_all_models.pt",
    "local_rq3_grid.json",
    "rq2_correlation.png",
    "local_rq1.png",
    "local_rq2.png",
    "local_rq3.png",
]:
    path = os.path.join("output", f)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    if exists:
        status = f"(exists, {size} bytes)"
    else:
        status = "(NOT FOUND)"
    print(f"  {f}: {status}")

# Optionally upload to Drive
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    shutil.copytree("output", "/content/drive/MyDrive/EKVA_output", dirs_exist_ok=True)
    print("\nResults uploaded to Google Drive: /content/drive/MyDrive/EKVA_output/")
except:
    print("\n(Drive mount skipped — results remain in /content/EKVA/output/)")
