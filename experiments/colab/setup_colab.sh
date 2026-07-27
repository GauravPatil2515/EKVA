#!/bin/bash
# EKVA Colab Setup Script
# Run this in a Colab notebook to set up the environment.
set -e

echo "=== EKVA Colab Setup ==="

# Clone repo (if not already present)
if [ ! -d "EKVA" ]; then
    echo "Cloning EKVA repo..."
    git clone https://github.com/GauravPatil2515/EKVA.git
fi
cd EKVA

# Install core dependencies
echo "Installing core dependencies..."
pip install -e . 2>&1 | tail -5

# Install model dependencies (transformers, datasets, accelerate)
echo "Installing model dependencies..."
pip install transformers>=4.40.0 datasets>=2.18.0 accelerate>=0.29.0 tokenizers>=0.19.0 2>&1 | tail -5

# Install scipy for correlation analysis
echo "Installing scipy..."
pip install scipy>=1.11.0 2>&1 | tail -3

echo ""
echo "=== Setup Complete ==="
echo "Run RQ1/RQ2 (CPU):  python experiments/rq1_granularity_comparison.py --model qwen1.5-moe-a2.7b --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt"
echo "Run RQ2 (CPU):      python experiments/rq2_entropy_routing_correlation.py --models qwen1.5-moe-a2.7b mixtral-8x7b deepseek-moe-16b --calibration-dir output"
echo "Run Colab entry:    python experiments/colab/run_rq1_rq2.py"
