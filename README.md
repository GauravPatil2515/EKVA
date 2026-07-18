# EKVA — Expert-Aware KV Budget Allocation for Sparse MoE Inference

> A roofline-guided, entropy-driven approach to KV-cache optimization in sparse
> Mixture-of-Experts LLMs. Allocate per-expert KV budgets proportional to each
> expert's attention complexity instead of a uniform budget.

Status: **research prototype (Phase 1–2 implemented; Phase 3 / kernel in progress).**
This README reflects the *actual* code on disk — see `PLAN.md` for the 12-week roadmap.

---

## What exists today

| Component | Status | Location |
|---|---|---|
| Per-expert attention-entropy calibration | ✅ working | `ekva/calibration/` |
| KV budget derivation (proportional + multi-signal) | ✅ working | `ekva/budget/derive.py` |
| Allocation policies (7: uniform, ekva, ekva-multi, random, snapkv, pyramidkv, dynamickv) | ✅ working | `ekva/budget/policies.py` |
| Software KV-cache simulator + eviction (recency/attention/random/**hybrid**) | ✅ working | `ekva/simulator/` |
| past_key_values truncation hook | 🟡 scaffold (Week 4) | `ekva/simulator/hook.py` |
| Roofline instrumentation | 🟡 scaffold (Weeks 8–9) | `ekva/profiling/` |
| Triton FA2 kernel (variable tile / fused) | 🟡 scaffold (Weeks 10–11) | `ekva/kernel/` |
| Benchmark harnesses (LongBench/RULER/Needle/InfiniteBench/PPL) | 🟡 scaffold | `ekva/benchmarks/` |

**Not yet real:** the simulator currently reports budgets and a *placeholder* PPL
path — the actual truncated-cache PPL wiring lives in `ekva/simulator/hook.py`
(Week 4). The kernel and roofline modules need `triton` + a CUDA GPU (Colab A100),
not the 3050 laptop.

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

## Quick start (CPU, no model download)

```bash
cd EKVA
python3 -m pip install -e .                 # core deps (torch, numpy, matplotlib, pyyaml)
python3 -m pytest tests/ -q                 # 22 tests, all CPU
python3 experiments/generate_mock_calibration.py --model mixtral-8x7b
python3 experiments/plot_calibration.py --input output/mixtral-8x7b_phase1.pt
```

Real-model runs (Weeks 1–6, 10–11) need `transformers`, a GPU, and model weights —
see `docs/MODELS.md` and `docs/QUICKSTART.md`.

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
