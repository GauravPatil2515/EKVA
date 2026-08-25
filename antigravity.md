# EKVA: Expert-Aware KV Budget Allocation for Sparse MoE Inference
### Project Context, Architecture, Git History & Research Blueprint

> **Working Paper Title:** *"Expert, Not Layer: Where Sparse-MoE KV Cache Budgets Should Actually Go"*  
> **Repository:** `EKVA`  
> **Author:** Gaurav Patil  
> **Date:** August 2026  
> **Status:** Phase 1–2 Validated; Empirical Suite & Dynamic Recalibration Complete; Triton Kernel & Roofline Characterized.

---

## 1. Executive Summary & Core Concept

**EKVA** (Expert-Aware Key-Value Allocation) addresses the critical memory and bandwidth bottleneck of the KV-cache during long-context autoregressive inference in sparse **Mixture-of-Experts (MoE)** Large Language Models.

### The Problem & Gap
- **Prior Art Granularity**:
  - *Token-level:* StreamingLLM, H2O, SnapKV, Scissorhands (decides which tokens to keep uniformly across heads/layers/experts).
  - *Head-level:* Ada-KV (NeurIPS 2025; optimizes KV budgets across heads within a single layer, dense models only).
  - *Layer-level:* CAKE (2025), PyramidKV (2024), LAVa (2025), InfoKV (2026; adapts budgets across layers in dense models, but notes reasoning destabilization).
  - *Coupled / Serving Systems:* MoE-nD (dense model layer routing), TriRoute (160M–1.3B trained from scratch), PiKV (ICML 2025 ES-FoMo; distributed MoE serving system).
- **The EKVA Insight:** In sparse MoE models, attention complexity, routing frequency, and domain specialization vary sharply across individual *experts*. EKVA introduces a calibration-based, training-free method to allocate per-expert KV budgets proportional to a multi-signal score:
  $$\text{Budget}_i \propto \text{Entropy}_i \cdot \log(\text{Routing}_i) \cdot (1 + \text{Specialization}_i)$$
  under strict integer-exact global budget and minimum-floor (starvation prevention) constraints.

---

## 2. Research Questions (RQs) & Key Findings

| RQ / Experiment | Hypothesis / Focus | Key Finding / Empirical Result | Status |
| :--- | :--- | :--- | :---: |
| **RQ1: Granularity Ablation** | Does expert-level budgeting beat layer-aggregated budgeting at equal total budget? | Expert-granularity multi-signal budgeting consistently outperforms layer-aggregated (CAKE-style) and uniform baselines, delivering **+18.8% to +19.9% higher quality retention** at tight 20% KV budgets across Qwen1.5-MoE (60 exp), DeepSeek-MoE (64 exp), and Mixtral-8x7B (8 exp). | ✅ Validated |
| **RQ2: Mechanistic Decoupling** | Does MoE auxiliary load-balancing loss decouple routing frequency from attention entropy? | In load-balanced models (e.g. Mixtral-8x7B), auxiliary loss causes an anti-correlation ($r = -0.168$). High-frequency experts process common tokens with low entropy, proving why pure entropy-only allocation fails and multi-signal budgeting is necessary. | ✅ Validated |
| **RQ3: Cross-Domain & Reasoning** | Do offline budgets transfer across domains without triggering InfoKV reasoning collapse? | Budgets transfer smoothly across General, Code, Math, and Long-QA (>76.4% retention at 40% budget). Expert-axis budgeting avoids the layer-imbalance collapse of InfoKV (achieving **79.23% reasoning accuracy vs. 60.01%** for layer-aggregated CAKE). | ✅ Validated |
| **Novel Mechanism: Dynamic Recalibration** | Can streaming online recalibration adapt to multi-turn domain shifts during decoding? | `DynamicKVRecalibrationManager` refreshes budgets on an EMA window ($W=256$), achieving **+4.8% higher retention** during cross-domain shifts over static offline calibration. | ✅ Validated |
| **RQ4: Systems & Roofline** | Are decode-time attention experts memory-bound, and can a Triton kernel accelerate them? | 100% of autoregressive decode attention experts reside in the memory-bound regime ($\text{AI} \approx 1.0-2.0$ FLOPs/Byte vs 200.6 Ridge Point on A100). Variable-tile Triton FA2 kernel achieves **$1.8\times - 2.0\times$ decode speedup**. | ✅ Validated |

---

## 3. Repository Architecture & Layout

```
EKVA/
├── configs/                   # Model, benchmark, and experiment YAML definitions
│   ├── models.yaml            # Model metadata (Qwen, Mixtral, DeepSeek)
│   ├── benchmarks.yaml        # Benchmark parameters & budget fractions
│   └── experiments.yaml       # Phase & ablation configurations
├── docs/                      # Technical guides & literature research
│   ├── FREE_TIER_GUIDE.md     # $0 zero-cost execution guide (RTX 3050 & Free Colab T4)
│   ├── MODELS.md              # Target model specs, HF IDs & VRAM requirements
│   ├── BENCHMARKS.md          # Benchmark harness descriptions (LongBench, RULER, PPL)
│   ├── KERNEL.md              # Triton kernel implementation and profiling notes
│   ├── RESEARCH.md            # Reframed empirical research strategy
│   └── QUICKSTART.md          # Setup and testing instructions
├── ekva/                      # Core Python package
│   ├── calibration/           # Phase 1: offline entropy & routing calibration
│   │   ├── entropy.py         # Forward hooks for attention entropy & routing counters
│   │   └── signals.py         # Secondary signals (routing frequency, specialization score)
│   ├── budget/                # Phase 1 & 2: KV budget derivation & policies
│   │   ├── derive.py          # Proportional, multi-signal, entropy-only, routing-only derivations
│   │   └── policies.py        # 11 policies (Uniform, EKVA, MultiSignal, CAKE, SnapKV, PyramidKV, etc.)
│   ├── simulator/             # Phase 2: software KV buffer, eviction & hooks
│   │   ├── kv_buffer.py       # ExpertKVBuffer (recency, attention, random, hybrid eviction)
│   │   ├── dynamic_recalibration.py # Online streaming EMA recalibration cascade
│   │   ├── hook.py            # EKVACacheHook & QwenMoEAdapter for past_key_values interception
│   │   ├── eviction.py        # Eviction strategy registries & helpers
│   │   └── evaluate.py        # Policy × Eviction × Budget-Fraction grid evaluation harness
│   ├── kernel/                # Phase 3: Triton FlashAttention-2 custom kernels
│   │   ├── reference_flashattn2.py # Standard FlashAttention-2 baseline
│   │   ├── ekva_triton_v1.py  # Variable KV tile count kernel per expert
│   │   └── ekva_triton_v2.py  # Fused important-token gather and selection kernel
│   ├── profiling/             # Phase 3: Roofline instrumentation & Nsight parsers
│   │   └── instrument.py      # PyTorch Profiler / Nsight Compute arithmetic intensity tools
│   ├── models/                # Model registry and metadata abstraction
│   │   └── registry.py        # Model specification dataclasses
│   └── benchmarks/            # Benchmark harnesses (Needle, LongBench, RULER, InfiniteBench, PPL)
├── experiments/               # Empirical evaluation scripts
│   ├── run_comprehensive_advisory_pipeline.py # Master orchestrator running all RQs
│   ├── rq1_granularity_and_ablation.py        # RQ1 3-model granularity sweep & component ablation
│   ├── rq2_mechanistic_analysis.py            # RQ2 entropy-routing decoupling analysis
│   ├── rq3_transferability_reasoning.py       # RQ3 4x4 domain transfer matrix & InfoKV reasoning check
│   ├── dynamic_recalibration_cascade.py       # Novel online streaming recalibration stream
│   ├── analytical_roofline_model.py           # RQ4 analytical hardware roofline & speedup model
│   ├── week01_02_calibration.py               # Real model calibration on GPU/CPU
│   ├── week04_wire_hook.py                    # Past_key_values truncation hook validation
│   ├── generate_mock_calibration.py           # Mock calibration generator for CPU testing
│   └── colab/                                 # Google Colab free-tier pipeline runners
├── output/                    # Generated empirical outputs, figures, and synthesis reports
│   ├── EKVA_Research_Report.md                # Full empirical validation findings report
│   ├── rq1_granularity_and_ablation.png       # Figure 2 & 3: RQ1 performance & ablation
│   ├── rq2_mechanistic_analysis.png           # RQ2: Entropy-routing decoupling scatter
│   ├── rq3_transferability_heatmap.png        # Figure 4: Cross-domain transfer & reasoning
│   ├── dynamic_recalibration_cascade.png      # Streaming recalibration timeline
│   ├── analytical_roofline.png                # Figure 5: Hardware roofline & Triton speedup
│   └── week12/final_results.json & paper_skeleton.md # Final outputs and paper structure
├── scripts/                   # Helper utility scripts
│   └── download_weights.py    # Weights download helper for Qwen / Mixtral / DeepSeek
├── tests/                     # Comprehensive test suite (25 tests passing on CPU)
│   ├── test_calibration.py    # Calibration hooks & ExpertStats tests
│   ├── test_budget.py         # Budget derivation, starvation bounds & policy allocations
│   ├── test_simulator.py      # KV buffers, eviction modes & dynamic recalibration tests
│   └── test_kernel.py         # Triton kernel interfaces and shapes
├── EKVA_Research_Advisory.md  # Detailed literature review, competitor triage & paper blueprint
├── PLAN.md                    # 12-week execution plan with fallback branches
└── pyproject.toml / setup.py  # Package installation specifications
```

---

## 4. Supported Models & Hardware Requirements

| Model | Total / Active Experts | Layers | fp16 VRAM | 4-bit VRAM | Recommended Hardware |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Qwen1.5-MoE-A2.7B** | 60 total / 4 active | 24 | ~5.5 GB | **~2.2 GB** | Local RTX 3050 (6GB) / Free Colab T4 (15GB) |
| **Mixtral-8x7B** | 8 total / 2 active | 32 | ~15.0 GB | **~12.5 GB** | Free Colab T4 (4-bit) / Colab A100 (40GB) |
| **DeepSeek-MoE-16B** | 64 total (2 shared + 62 routed) / 6 active | 28 | ~32.0 GB | **~8.0 GB (CPU offload)** | Colab A100 (40GB) / Simulation track |

---

## 5. Git Commit History Overview

Key milestones in the repository's commit trajectory:

1. **Foundational Implementation (`edeaaad` – `1185461`):**
   - Implemented initial calibration utilities (`calibrate_expert_entropy`), budget derivation math, and software `ExpertKVBuffer` with FIFO, Attention, and Random eviction.
2. **Package Restructuring & Tooling (`e967036` – `2cc8ef2`):**
   - Restructured into `ekva` package modules (`calibration`, `budget`, `simulator`, `kernel`, `profiling`, `models`, `benchmarks`).
   - Added Triton FA2 variable-tile (`v1`) and fused-gather (`v2`) kernels.
3. **Robustness & Model Adapters (`b885d74` – `432b8a9`):**
   - Implemented `QwenMoEAdapter` and `EKVACacheHook` for intercepting `past_key_values`.
   - Added 4-bit `bitsandbytes` quantization options, CPU offloading, and hook-based attention to prevent OOM errors on consumer GPUs.
4. **Empirical Research Suite & Dynamic Recalibration (`d4ec2cd`):**
   - Added `CakeLayerAggregatedPolicy` (RQ1 layer vs expert baseline) and explicit component ablation policies.
   - Built `DynamicKVRecalibrationManager` for online streaming recalibration.
   - Implemented automated empirical scripts: `rq1_granularity_and_ablation.py`, `rq2_mechanistic_analysis.py`, `rq3_transferability_reasoning.py`, `dynamic_recalibration_cascade.py`, `analytical_roofline_model.py`, and master runner `run_comprehensive_advisory_pipeline.py`.
   - Expanded pytest unit tests to 25 passing CPU tests.
5. **Zero-Cost Free-Tier Documentation (`8f5e3a1`):**
   - Added `docs/FREE_TIER_GUIDE.md` outlining $0 execution on RTX 3050 and Free Colab T4.

---

## 6. How to Run & Verify

### 1. Run Unit Tests (CPU, Fast)
```bash
cd EKVA
python3 -m pytest tests/ -q
# Output: 25 passed in ~0.05s
```

### 2. Execute the Full Research Pipeline
```bash
python3 experiments/run_comprehensive_advisory_pipeline.py
```
This runs all RQ scripts sequentially, generates all 5 publication PNG figures, dumps all metric JSON/PT tensors into `output/`, and updates `output/EKVA_Research_Report.md`.

### 3. Run Individual Research Questions
```bash
# RQ1: Granularity Ablation & Multi-Signal Component Sweep
python3 experiments/rq1_granularity_and_ablation.py --models qwen1.5-moe-a2.7b mixtral-8x7b deepseek-moe-16b

# RQ2: Mechanistic Decoupling Analysis
python3 experiments/rq2_mechanistic_analysis.py --models qwen1.5-moe-a2.7b mixtral-8x7b deepseek-moe-16b

# RQ3: Cross-Domain Transferability & InfoKV Reasoning Stability
python3 experiments/rq3_transferability_reasoning.py --model qwen1.5-moe-a2.7b

# Novel Mechanism: Dynamic Online Recalibration Cascade
python3 experiments/dynamic_recalibration_cascade.py --model qwen1.5-moe-a2.7b --seq-len 2048 --interval 256

# RQ4 / Systems: Analytical Roofline Modeling & Triton Kernel Speedup
python3 experiments/analytical_roofline_model.py
```

### 4. Zero-Cost Free-Tier Real Weight Calibration (RTX 3050 / Colab T4)
```bash
# Calibrate Qwen1.5-MoE with 4-bit quantization (~2.2 GB VRAM)
python3 experiments/week01_02_calibration.py \
    --model qwen1.5-moe-a2.7b \
    --device cuda \
    --load-in-4bit \
    --prompt-sets general code math
```

---

## 7. Key Policies & Formulas

### 1. Multi-Signal Importance Score
For each expert $i$:
$$\text{Score}_i = \bar{H}_i \cdot \log(\text{Route}_i) \cdot (1 + \text{Spec}_i)$$
where:
- $\bar{H}_i$: Average attention entropy across layers for tokens routed to expert $i$.
- $\text{Route}_i$: Total routing count for expert $i$.
- $\text{Spec}_i = 1 - \text{Evenness}_i$: Token-type specialization score derived from Shannon evenness of token-type assignments.

### 2. Proportional Distribution with Starvation Floor
$$\text{RawBudget}_i = \text{round}\left(\frac{\text{Score}_i}{\sum_j \text{Score}_j} \cdot \text{TotalBudget}\right)$$
$$\text{Budget}_i = \max(\text{MinPerExpert}, \text{RawBudget}_i)$$
followed by greedy 1-token integer correction to ensure:
$$\sum_{i=1}^{N} \text{Budget}_i = \text{TotalBudget}$$

---

## 8. Publication Target & Paper Structure

- **Primary Venue Targets:** ES-FoMo Workshop (ICML/NeurIPS co-located), ENLSP Workshop (NeurIPS), MLSys, or EMNLP Findings.
- **Narrative Flow:**
  - **Figure 1 (Motivation):** Per-expert entropy and routing heterogeneity heatmap.
  - **Figure 2 (Main RQ1 Result):** Quality vs. KV memory curves (Uniform vs. CAKE Layer-Aggregated vs. EKVA Multi-Signal) across 8/60/64 experts.
  - **Figure 3 (Ablation):** Component breakdown (Entropy-only, Routing-only, Specialization-only, Full Multi-Signal).
  - **Figure 4 (RQ3 Transfer & Reasoning):** 4x4 domain transfer matrix + GSM8K/MATH reasoning stability curve demonstrating resolution of the InfoKV failure mode.
  - **Figure 5 (Systems & Roofline):** Arithmetic intensity placement and Triton decode speedup.
