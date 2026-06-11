# EKVA: Expert-Aware KV Budget Allocation for Sparse MoE Inference

> **A Roofline-Guided Triton Kernel Approach to KV Cache Optimization in Sparse Mixture-of-Experts LLMs**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Triton](https://img.shields.io/badge/kernel-Triton-green.svg)](https://github.com/openai/triton)
[![Status: Research](https://img.shields.io/badge/status-research-orange.svg)]()

---

## Overview

Sparse Mixture-of-Experts (MoE) language models achieve parameter efficiency by activating only a small subset of experts per token at inference time. However, existing KV cache optimization methods — including token eviction schemes like **SnapKV** and budget allocation strategies like **PyramidKV** — apply **uniform memory budgets** across all experts and layers.

We demonstrate this assumption is both **algorithmically suboptimal** and **hardware-inefficient** for sparse MoE architectures.

**EKVA** (Expert-Aware KV Budget Allocation) addresses this by:

1. Performing a lightweight calibration pass to measure **per-expert attention entropy**
2. Deriving a **differentiated KV budget tensor** proportional to each expert's attention complexity
3. Implementing the allocation as a **fused Triton kernel** extending FlashAttention-2 tiling with variable per-expert tile sizes

This approach eliminates the static shape assumption in existing kernels and exploits expert specialization as a signal for both algorithmic and hardware-level inference optimization.

---

## Key Insight: Roofline Heterogeneity Across Experts

```
High-Entropy Expert         Low-Entropy Expert
  (broad attention)            (sharp attention)
        │                            │
  Higher Op Intensity          Lower Op Intensity
  → Compute-Bound              → Memory-Bound
  → Needs more KV budget       → Wastes HBM loading
                                 negligible KV entries
```

Through systematic **roofline analysis** of attention computation across individual MoE experts, we find that experts occupy fundamentally different positions on the compute-memory bandwidth tradeoff curve:

| Expert Type | Attention Pattern | Op Intensity | Hardware Regime | EKVA Action |
|---|---|---|---|---|
| High-entropy | Broad, distributed | High | Near compute-bound | Larger KV budget |
| Low-entropy | Sharp, focused | Low | Memory-bound | Smaller KV budget |

---

## Method: EKVA Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                   CALIBRATION PASS                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │  Few prompts │ →  │  Per-expert  │ →  │  KV Budget   │  │
│  │  (offline)   │    │  entropy H_e │    │  Tensor B_e  │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  INFERENCE (EKVA KERNEL)                    │
│  FlashAttention-2 Tiling + Variable Per-Expert Tile Sizes   │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Expert 0 (high H) → large tile → full KV budget    │   │
│  │  Expert 1 (low H)  → small tile → reduced KV budget │   │
│  │  Expert 2 (high H) → large tile → full KV budget    │   │
│  │  Expert 3 (low H)  → small tile → reduced KV budget │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

---

## Results

Evaluated on **Mixtral-8x7B** and **DeepSeek-V2** on long-context benchmarks:

- ✅ Competitive perplexity retention vs uniform-budget baselines
- ✅ Reduced KV cache memory footprint
- ✅ Improved HBM bandwidth utilization for memory-bound expert classes
- ✅ Roofline profiling confirms low-entropy experts shift measurably closer to memory bandwidth ceiling

> **Note:** Full quantitative results coming soon.

---

## Repository Structure

```
EKVA/
├── ekva/
│   ├── __init__.py
│   ├── calibration.py         # Per-expert attention entropy measurement
│   ├── budget.py              # KV budget tensor derivation
│   ├── kernel/
│   │   ├── ekva_triton.py     # Fused Triton kernel (variable tile sizes)
│   │   └── flash_attn_base.py # FlashAttention-2 base tiling reference
│   ├── models/
│   │   ├── mixtral_patch.py   # Mixtral-8x7B integration
│   │   └── deepseek_patch.py  # DeepSeek-V2 integration
│   └── utils/
│       ├── roofline.py        # Roofline analysis utilities
│       └── profiling.py       # HBM bandwidth profiling tools
├── benchmarks/
│   ├── run_longbench.py       # LongBench evaluation
│   ├── run_perplexity.py      # Perplexity evaluation
│   └── roofline_plots.py      # Roofline visualization scripts
├── configs/
│   ├── mixtral_ekva.yaml      # EKVA config for Mixtral-8x7B
│   └── deepseek_ekva.yaml     # EKVA config for DeepSeek-V2
├── notebooks/
│   ├── 01_calibration_demo.ipynb
│   ├── 02_roofline_analysis.ipynb
│   └── 03_benchmark_results.ipynb
├── tests/
│   ├── test_calibration.py
│   ├── test_budget_allocation.py
│   └── test_triton_kernel.py
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/GauravPatil2515/EKVA.git
cd EKVA
pip install -r requirements.txt
```

**Requirements:**
- Python 3.10+
- PyTorch 2.2+
- Triton 2.3+
- transformers
- flash-attn (for baseline comparison)

---

## Quick Start

### 1. Calibration Pass

```python
from ekva.calibration import calibrate_expert_entropy
from ekva.budget import derive_kv_budget

# Load your MoE model (e.g., Mixtral-8x7B)
model, tokenizer = load_model("mistralai/Mixtral-8x7B-v0.1")

# Run calibration on a few representative prompts
entropy_map = calibrate_expert_entropy(
    model=model,
    tokenizer=tokenizer,
    calibration_prompts=calibration_data,
    num_experts=8
)

# Derive differentiated KV budgets
budget_tensor = derive_kv_budget(
    entropy_map=entropy_map,
    total_budget=2048,   # total KV budget tokens
    strategy="proportional"
)
```

### 2. EKVA Inference

```python
from ekva.kernel.ekva_triton import ekva_attention

# Drop-in replacement for standard attention in MoE layers
output = ekva_attention(
    q=query,
    k=key,
    v=value,
    expert_id=current_expert,
    budget_tensor=budget_tensor
)
```

---

## Baselines Compared

| Method | Budget Type | MoE-Aware | Kernel |
|---|---|---|---|
| Full KV Cache | Uniform (100%) | ✗ | Standard |
| SnapKV | Per-head, uniform across experts | ✗ | Custom |
| PyramidKV | Per-layer curve | ✗ | Standard |
| Ada-SnapKV | Per-head adaptive | ✗ | Custom |
| **EKVA (Ours)** | Per-expert, entropy-guided | ✅ | Triton (variable tile) |

---

## Technical Background

### Why Uniform Budgets Fail for MoE

In a sparse MoE model, each expert **specializes** during training — some experts handle broad, contextual reasoning (high entropy), others handle sharp, token-specific operations (low entropy). Applying the same KV budget to both:

- **Wastes HBM bandwidth** loading KV entries from low-entropy experts that barely affect output
- **Starves** high-entropy experts that genuinely need broader context access

### Roofline Analysis

The roofline model characterizes whether a kernel is **compute-bound** or **memory-bound**:

```
Attainable Performance = min(Peak FLOPS, Bandwidth × Operational Intensity)
```

For attention:
- **Operational Intensity** = FLOPs / bytes accessed from HBM
- Low-entropy experts have low OI → deeply memory-bound → primary optimization target
- High-entropy experts have higher OI → approaching compute-bound → retain larger budget

### FlashAttention-2 + Variable Tiling

EKVA extends the FlashAttention-2 tiling strategy (which keeps Q, K, V tiles in SRAM to avoid HBM reads/writes) with **variable per-expert tile sizes** controlled by the precomputed budget tensor. This eliminates the static shape assumption in standard FlashAttention implementations.

---

## Related Work

- **SnapKV** — LLM Knows What You are Looking for Before Generation (2024)
- **PyramidKV** — Dynamic KV Cache Compression Based on Pyramidal Information Funneling (2024)
- **FlashAttention-2** — Faster Attention with Better Parallelism and Work Partitioning (2023)
- **FlashAttention-4** — Algorithm and Kernel Pipelining Co-design (2026)
- **MoQAE** — Mixed-Precision Quantization via Mixture of Quantization-Aware Experts (2025)
- **KVTC** — KV Cache Transform Coding for Compact Storage in LLM Inference (2025)
- **KVzip** — Query-Agnostic KV Cache Compression (NeurIPS 2025)

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{patil2026ekva,
  title   = {Expert-Aware KV Budget Allocation for Sparse Mixture-of-Experts Inference: A Roofline-Guided Triton Kernel Approach},
  author  = {Gaurav Patil},
  year    = {2026},
  url     = {https://github.com/GauravPatil2515/EKVA}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

**Gaurav Patil** — Final-year B.E., AI/ML Engineer  
GitHub: [@GauravPatil2515](https://github.com/GauravPatil2515)
