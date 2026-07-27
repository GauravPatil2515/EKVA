# EKVA — Expert-Aware KV Budget Allocation for Sparse MoE Inference

> A roofline-guided, entropy-driven approach to KV-cache optimization in sparse
> Mixture-of-Experts LLMs. Allocate per-expert KV budgets proportional to each
> expert's attention complexity instead of a uniform budget.

Status: **research prototype (Phase 1–2 implemented; kernel + roofline in progress).**
This README reflects the *actual* code on disk — see `PLAN.md` for the 12-week roadmap.

---

## What exists today

| Component | Status | Location |
|---|---|---|
| Per-expert attention-entropy + routing calibration | working | `ekva/calibration/` |
| KV budget derivation (proportional + multi-signal) | working | `ekva/budget/derive.py` |
| Allocation policies (7: uniform, ekva, ekva-multi, random, snapkv, pyramidkv, dynamickv) | working | `ekva/budget/policies.py` |
| Software KV-cache simulator + eviction (recency/attention/random/**hybrid**) | working | `ekva/simulator/` |
| Benchmark grid harness (policy x eviction x budget-fraction) | working | `ekva/simulator/evaluate.py` |
| past_key_values truncation hook | scaffold (Week 4) | `ekva/simulator/hook.py` |
| Roofline instrumentation | scaffold (Weeks 8–9) | `ekva/profiling/` |
| Triton FA2 kernel (variable tile / fused) | scaffold (Weeks 10–11) | `ekva/kernel/` |
| Benchmark harnesses (LongBench/RULER/Needle/InfiniteBench/PPL) | scaffold | `ekva/benchmarks/` |

**Honest scope:** the simulator and budget math are real and CPU-testable
(22 pytest tests pass). The PPL *scoring* path is a placeholder
(`score_fn=None` falls back to budget-aware heuristic in
`run_policy_eviction_grid`); the real truncated-cache PPL wiring is the Week-4
hook. The kernel, roofline, and benchmark *harnesses* need `triton` + a CUDA
GPU (Colab A100), not the 3050 laptop.

---

## Repository layout

```
EKVA/
├── ekva/
│   ├── calibration/   # per-expert entropy + routing/specialization signals
│   ├── budget/        # derive_kv_budget + 7 allocation policies
│   ├── simulator/     # KV buffer, eviction, hook, eval harness
│   ├── kernel/        # Triton FA2 reference + v1 (variable tile) + v2 (fused)
│   ├── profiling/     # roofline instrumentation (PyTorch Profiler / Nsight)
│   ├── models/        # model registry (HF ids, expert counts, VRAM)
│   └── benchmarks/    # LongBench / RULER / Needle / InfiniteBench / PPL
├── configs/           # models.yaml, benchmarks.yaml, experiments.yaml
├── experiments/       # week01..week12 + fallback_branches + mock + plot
├── tests/             # pytest suite (core runs on CPU, no weights)
├── docs/              # MODELS.md, QUICKSTART.md, BENCHMARKS.md, KERNEL.md
├── PLAN.md            # 12-week combination-matrix plan + fallback branches
├── requirements.txt   # grouped, commented; install per phase
└── pyproject.toml     # PEP 621 metadata
```

---

## Install

```bash
cd EKVA
python3 -m pip install -e .      # core: torch, numpy, matplotlib, pyyaml, tqdm
```
Phase-specific deps (transformers for real models, triton+CUDA for the kernel)
are documented per phase in `requirements.txt` and `docs/QUICKSTART.md`.

---

## Quick start (CPU, no model download)

```bash
python3 -m pytest tests/ -q                          # 22 tests, all CPU
python3 experiments/generate_mock_calibration.py --model mixtral-8x7b
python3 experiments/plot_calibration.py --input output/mixtral-8x7b_phase1.pt
```

Real-model runs (Weeks 1–6, 10–11) need `transformers`, a GPU, and weights —
see `docs/MODELS.md` and `docs/QUICKSTART.md`.

---

## Minimal API

```python
import torch
from ekva.calibration.entropy import calibrate_expert_entropy
from ekva.budget.derive import derive_kv_budget
from ekva.budget import get_policy
from ekva.simulator import ExpertKVBuffer, run_policy_eviction_grid

# entropy_map comes from calibrate_expert_entropy(model, tokenizer, prompts, num_experts);
# each expert's stats dict MUST contain `avg_entropy` and `routing_count`.
entropy_map = {
    eid: {"avg_entropy": torch.tensor([0.4 + 0.1 * eid]),
          "routing_count": torch.tensor([1.0 + eid])}
    for eid in range(8)
}

# Derive a per-expert budget tensor (sums to total_budget, respects min floor):
budget = derive_kv_budget(entropy_map, total_budget=2048, min_per_expert=64)

# Or go through a named policy:
budgets = get_policy("ekva").allocate(num_experts=8, total_budget=2048,
                                      entropy_map=entropy_map)  # -> {expert_id: int}

# Run the benchmark grid (placeholder scoring unless score_fn supplied):
grid = run_policy_eviction_grid(
    num_experts=8, total_budget=2048,
    policy_names=["uniform", "ekva"],
    eviction_names=["attention", "recency"],
    budget_fractions=[0.2, 0.4, 0.6],
)
```

---

## The 12-week plan

See [`PLAN.md`](PLAN.md) — full combination matrix (models × policies × evictions ×
budget fractions × benchmarks × kernel variants) mapped week-by-week, with explicit
fallback branches at every decision point.

---

## Citation

```bibtex
@misc{patil2026ekva,
  title  = {Expert-Aware KV Budget Allocation for Sparse Mixture-of-Experts Inference:
            A Roofline-Guided Triton Kernel Approach},
  author = {Gaurav Patil},
  year   = {2026},
  url    = {https://github.com/GauravPatil2515/EKVA}
}
```

License: MIT.

---

## Colab Quickstart

Run EKVA on Google Colab (free T4 or A100):

```bash
# 1. Clone the repo
git clone https://github.com/GauravPatil2515/EKVA.git
cd EKVA

# 2. Install dependencies
pip install -e .
pip install transformers datasets accelerate scipy

# 3. Run RQ1/RQ2 (CPU-only, works on free Colab T4)
python experiments/colab/run_rq1_rq2.py
```

For A100 runs (roofline + Triton kernel):
```bash
# Change runtime type to GPU (A100) in Colab settings
pip install -e ".[kernel]"
python experiments/colab/run_rq4_roofline.py
python experiments/colab/run_triton_kernel.py
```

### Weights Download
```bash
# Download Qwen1.5-MoE-A2.7B (fits RTX 3050 with cpu_offload)
python scripts/download_weights.py --model qwen1.5-moe-a2.7b --device cpu

# Download Mixtral-8x7B (needs A100)
python scripts/download_weights.py --model mixtral-8x7b --device cuda
```

### Local Development (RTX 3050)
```bash
# CPU-only mock calibration
python experiments/generate_mock_calibration.py --model mixtral-8x7b

# Real calibration (Qwen fits 3050 with cpu_offload)
python experiments/week01_02_calibration.py \
    --model qwen1.5-moe-a2.7b --device cpu \
    --prompt-sets general code math

# RQ1/RQ2 analysis (CPU)
python experiments/rq1_granularity_comparison.py \
    --model qwen1.5-moe-a2.7b \
    --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt

python experiments/rq2_entropy_routing_correlation.py \
    --models qwen1.5-moe-a2.7b mixtral-8x7b deepseek-moe-16b \
    --calibration-dir output

# Hook validation (Qwen on 3050)
python experiments/week04_wire_hook.py \
    --model qwen1.5-moe-a2.7b \
    --calibration output/qwen1.5-moe-a2.7b_general_phase1.pt \
    --device cpu
```
