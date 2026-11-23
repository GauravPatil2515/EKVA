#!/usr/bin/env bash
# ==============================================================================
# EKVA v2: 1-Click Zero-Cost Automated Runner (Colab / Local / Cloud)
# 
# Works on both GPU (NVIDIA T4/A100/RTX) and CPU fallback.
# ==============================================================================
set -e

echo "=================================================================="
echo "🚀 EKVA v2: Automated Production Test & Benchmark Suite"
echo "=================================================================="

# 1. Environment and GPU inspection
echo "[Step 1/5] Checking hardware and CUDA environment..."
python3 -c "
import torch
cuda_avail = torch.cuda.is_available()
print(f'CUDA Available: {cuda_avail}')
if cuda_avail:
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    free_b, total_b = torch.cuda.mem_get_info(0)
    print(f'VRAM: {free_b / 1024**3:.2f} GB free / {total_b / 1024**3:.2f} GB total')
else:
    print('Running in CPU mode. (To test Triton GPU kernels in Colab, select: Runtime -> Change runtime type -> T4 GPU)')
"

# 2. Dependency installation
echo "[Step 2/5] Installing dependencies and package in editable mode..."
pip install -q --upgrade pip
pip install -q -e ".[models,kernel,dev]"
# Install quantization libraries if GPU available
python3 -c "
import torch, subprocess
if torch.cuda.is_available():
    subprocess.run(['pip', 'install', '-q', 'bitsandbytes', 'accelerate'])
"

# 3. Unit Tests & Invariants Suite
echo "[Step 3/5] Executing Unit Tests & Adversarial Invariants..."
pytest tests/ -v --tb=short

# 4. Compaction Micro-Benchmark
echo "[Step 4/5] Running Compaction Micro-benchmark & Crossover Analysis..."
PYTHONPATH=. python3 experiments/benchmark_triton_crossover.py --budget-ratio 0.50

# 5. Real Model Smoke Evaluation
echo "[Step 5/5] Running Real MoE Evaluation..."
python3 -c "
import torch, subprocess, sys
if torch.cuda.is_available():
    print('GPU detected. Launching 4-bit MoE evaluation on Qwen1.5-MoE...')
    cmd = [
        sys.executable, 'scripts/run_real_evaluation_suite.py',
        '--model', 'qwen1.5-moe-a2.7b',
        '--gsm8k-samples', '10',
        '--humaneval-samples', '5',
        '--niah-samples', '5',
        '--budgets', '0.40', '1.00',
        '--out-dir', 'output/colab_run'
    ]
    subprocess.run(cmd, check=True)
else:
    print('No GPU detected. Skipping heavy model evaluation on CPU to avoid timeout.')
    print('To run real model evaluations, enable T4 GPU in Google Colab.')
"

echo "=================================================================="
echo "✅ ALL EXECUTIONS COMPLETED WITH ZERO ERRORS!"
echo "=================================================================="
