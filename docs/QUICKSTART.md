# Quick Start

Two tracks: a **CPU / no-download** track to develop the pipeline on the 3050
laptop, and a **GPU / real-model** track for the actual experiments (Weeks 1–6, 10–11).

## Track A — CPU, no model download (verify the pipeline works)

```bash
cd EKVA
python3 -m pip install -e .                 # core: torch, numpy, matplotlib, pyyaml, tqdm
python3 -m pytest tests/ -q                 # 22 tests, all CPU-only

# Mock calibration (synthetic MoE, no weights):
python3 experiments/generate_mock_calibration.py --model mixtral-8x7b

# Visualize entropy heatmap + budget scatter:
python3 experiments/plot_calibration.py --input output/mixtral-8x7b_phase1.pt
```

Expect: `output/mixtral-8x7b_phase1.pt`, `output/entropy_heatmap.png`,
`output/entropy_vs_budget.png`.

## Track B — Real models (needs transformers + GPU)

```bash
# Install model deps (see docs/MODELS.md for which GPU):
python3 -m pip install transformers datasets accelerate

# Week 1-2: calibrate a real model (Qwen first — fits the 3050):
python3 experiments/week01_02_calibration.py \
    --model qwen1.5-moe-a2.7b --device cuda \
    --prompt-sets general code math

# Plot the result:
python3 experiments/plot_calibration.py \
    --input output/qwen1.5-moe-a2.7b_general_phase1.pt
```

## Calibration → budget → simulator (minimal Python)

```python
from ekva.calibration.entropy import calibrate_expert_entropy
from ekva.budget.derive import derive_kv_budget
from ekva.budget.policies import EKVAPolicy

# entropy_map = calibrate_expert_entropy(model, tok, prompts, num_experts=8)
budget_tensor = derive_kv_budget(entropy_map, total_budget=2048, strategy="proportional")

policy = EKVAPolicy()
budgets = policy.allocate(num_experts=8, total_budget=2048, entropy_map=entropy_map)
# -> {0: int, 1: int, ...} per-expert KV token budgets
```

## Common flags
- `--device cuda|cpu` — all week scripts accept it.
- `--total-budget N` — total KV tokens across experts (default 2048 / 4096).
- `--min-per-expert N` — floor so no expert starves (default 64).
- `--out-dir` — where artifacts land (everything under `output/`, git-ignored).

## Installing per phase
Don't install everything at once. `requirements.txt` is grouped:
- Core (Phase 1–2 CPU): already in `[project.dependencies]`.
- Models (Weeks 1–6): `pip install transformers datasets accelerate`.
- Kernel (Weeks 10–11): needs `triton` + CUDA GPU (Colab A100, not the 3050).
- Benchmarks: pull LongBench / RULER / InfiniteBench per `docs/BENCHMARKS.md`.
