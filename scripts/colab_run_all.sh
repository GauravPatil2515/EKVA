#!/usr/bin/env bash
# ==============================================================================
# EKVA v2: 1-Click Zero-Cost Google Colab / Cloud Automated Runner
# 
# Usage in Google Colab (Run in a single code block):
#     !curl -sSL https://raw.githubusercontent.com/GauravPatil2515/EKVA/main/scripts/colab_run_all.sh | bash
# Or locally / inside clone:
#     bash scripts/colab_run_all.sh
# ==============================================================================
set -e

echo "=================================================================="
echo "🚀 EKVA v2: Automated Production Test & Benchmark Suite"
echo "=================================================================="

# 1. Environment and GPU inspection
echo "[Step 1/5] Checking hardware and CUDA environment..."
python3 -c "
import torch, os
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f'VRAM: {free_b / 1024**3:.2f} GB free / {total_b / 1024**3:.2f} GB total')
"

# 2. Dependency installation
echo "[Step 2/5] Installing dependencies and editable package..."
pip install -q --upgrade pip
pip install -q -e ".[models,kernel,dev]"
pip install -q bitsandbytes accelerate

# 3. Adversarial Invariant Pytest Suite
echo "[Step 3/5] Executing 37 Unit Tests (Adversarial Invariants)..."
pytest tests/ -v --tb=short

# 4. Triton Compaction Micro-Benchmark
echo "[Step 4/5] Running Triton Compaction Micro-benchmark & Crossover Analysis..."
PYTHONPATH=. python3 experiments/benchmark_triton_crossover.py --budget-ratio 0.50

# 5. Real Model Smoke Evaluation (4-bit Qwen1.5-MoE)
echo "[Step 5/5] Running Real MoE Model Evaluation Suite (GSM8K & NIAH in 4-bit)..."
PYTHONPATH=. python3 scripts/run_real_evaluation_suite.py \
    --model qwen1.5-moe-a2.7b \
    --gsm8k-samples 10 \
    --humaneval-samples 5 \
    --niah-samples 5 \
    --budgets 0.40 1.00 \
    --out-dir output/colab_run

echo "=================================================================="
echo "✅ ALL STEPS EXECUTED SUCCESSFULLY WITH ZERO ERRORS!"
echo "Artifacts generated in output/colab_run and output/benchmark_crossover"
echo "=================================================================="
