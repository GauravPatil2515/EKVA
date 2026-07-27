# EKVA — Complete Implementation Plan

## Repository State (as of 2026-07-27)

**Git Status:**
- Branch: main (up-to-date with origin)
- Modified: PLAN.md, README.md
- Untracked: .kilo/, docs/RESEARCH.md, docs/ekva_research_strategy.md, docs/review.md
- 22 tests passing

**Existing Infrastructure (verified):**
- Calibration pipeline (entropy + routing) — working
- Budget derivation (proportional + multi-signal) — working
- 7 allocation policies — working
- Simulator with 4 evictions — working
- Evaluation harness — working
- Hook scaffold — placeholder only
- Reference FA2 kernel — working
- EKVA Triton v1/v2 kernels — scaffolds
- Benchmark harnesses — scaffolds
- 22 CPU tests passing

**Gaps to Fill:**
1. RQ1/RQ2 analysis scripts (CPU-only, use existing calibration)
2. Qwen-specific hook adapter for real truncated-cache PPL
3. Colab-ready setup scripts and requirements
4. Profiling instrumentation (roofline)
5. Triton kernel correctness tests

---

## Phase 0: Colab Hardening (Week 0 — 2 days)

### 0.1 Colab Setup Script
**File:** `experiments/colab/setup_colab.sh`

### 0.2 Colab Entry Points
- `experiments/colab/run_rq1_rq2.py` — CPU-only RQ1+RQ2
- `experiments/colab/run_rq3_transfer.py` — Cross-domain calibration (needs T4)
- `experiments/colab/run_rq4_roofline.py` — Per-expert roofline (needs A100)
- `experiments/colab/run_triton_kernel.py` — Triton v1/v2 (needs A100)

### 0.3 Requirements Split
- `requirements/base.txt` — core (torch, numpy, matplotlib, pyyaml, tqdm)
- `requirements/models.txt` — transformers, datasets, accelerate, tokenizers
- `requirements/kernel.txt` — triton
- `requirements/colab_a100.txt` — all above + nsight tools

### 0.4 Weights Download Helper
**File:** `scripts/download_weights.py`

---

## Phase 1: RQ1 + RQ2 (Week 1-2 — Local CPU/RTX 3050)

### 1.1 RQ1: Layer vs Expert Granularity Ablation
**File:** `experiments/rq1_granularity_comparison.py`

**Inputs:** `output/*_phase1.pt` (already exist for 3 models)

**Algorithm:**
1. Load entropy_map for each model
2. Expert-level budget: current EKVA (per-expert proportional)
3. Layer-level budget:
   a. Compute layer entropy = mean(entropy_map[e]["avg_entropy"][layer] for all e)
   b. Allocate budget proportional to layer entropy
   c. Distribute layer budget equally across experts in that layer
4. Run both through simulator (run_policy_eviction_grid)
5. Compare PPL degradation curves across budget fractions (10%-80%)

**Outputs:**
- `output/rq1_granularity_comparison.pt` — results dict
- `output/rq1_granularity_comparison.png` — figure

### 1.2 RQ2: Entropy–Routing Correlation
**File:** `experiments/rq2_entropy_routing_correlation.py`

**Algorithm:**
1. For each model (Qwen: 60 experts, Mixtral: 8, DeepSeek: 64):
2. Extract per-expert: entropy_mean = avg_entropy.mean(), routing = routing_count
3. Compute Pearson + Spearman correlation
4. Per-layer analysis: entropy vs routing per layer
5. Plot multi-panel figure (3 models × layers)

**Outputs:**
- `output/rq2_correlation.pt` — correlation table
- `output/rq2_correlation.png` — multi-panel figure

### 1.3 Visualization Script
**File:** `experiments/plot_rq1_rq2.py` — reusable plotting for both

---

## Phase 2: Hook Wiring (Week 2-3 — RTX 3050 with Qwen)

### 2.1 Identify Qwen MoE Structure
Quick script to print layer classes:
```python
from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-MoE-A2.7B", torch_dtype="auto")
for n, c in m.named_modules():
    if "moe" in type(c).__name__.lower() or "expert" in type(c).__name__.lower():
        print(n, type(c).__name__)
```

Expected: `Qwen1_5MoeSparseMoeBlock`, router=`gate`, experts=`experts` (ModuleList)

### 2.2 Wire Hook for Qwen
**File:** `experiments/week04_wire_hook.py` — replace placeholder with real adapter

**Implementation:**
- Find all Qwen1_5MoeSparseMoeBlock layers
- Register forward_pre_hook to capture router logits
- Register forward_hook to truncate KV for routed experts
- Call hook.truncate(expert_id, k, v, attn) in MoE forward

### 2.3 Validation Script
**File:** `experiments/validate_hook.py`
- Baseline PPL (full KV) — must match within 0.01
- 16 combos: 4 policies × 4 evictions at 20% budget
- Compare simulator PPL vs real truncated PPL

### 2.4 3050 CPU Offload Config
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
    max_memory={0: "5GB", "cpu": "16GB"}
)
```

---

## Phase 3: A100 Campaign (When Available — ~1 week)

### 3.1 RQ3: Cross-Domain Calibration Transfer
**File:** `experiments/colab/run_rq3_transfer.py`

**Calibration domains:** WikiText, Code, Long-context QA, Math
**Evaluation domains:** LongBench (4 tasks), RULER (2 tasks), Needle, WikiText PPL
**Matrix:** 4 × 7 = 28 runs per model
**Output:** `output/rq3_cross_domain_degradation.pt` + heatmap

### 3.2 RQ4a: Per-Expert Roofline
**File:** `experiments/colab/run_rq4_roofline.py`

**Uses:** `ekva/profiling/instrument.py` (needs completion)
1. PyTorch Profiler + Nsight on Mixtral-8x7B
2. Extract FLOPs/Bytes per expert per layer
3. Compute Arithmetic Intensity = FLOPs / Bytes
4. Roofline plot: x=AI, y=attained GFLOP/s, color=entropy bucket
5. Overlay A100 ceilings: 312 TFLOP/s fp16, 1.55 TB/s HBM

### 3.3 RQ4b: Triton Kernels
**Files:** `ekva/kernel/ekva_triton_v1.py`, `ekva/kernel/ekva_triton_v2.py`

**v1:** Variable tile count — stops after `kv_budget // BLOCK_N` tiles
**v2:** Fused important-token index selection — gather K/V before tile loop

**Correctness gate:** Match FA2 reference at `kv_budget == N_KV`
**Benchmark:** kernel time, HBM reads, SM occupancy vs baseline

---

## Phase 4: Paper Assembly (Post-A100)

### 4.1 Figure Pipeline
```
output/
  rq1_granularity_comparison.png     # RQ1 bars
  rq2_correlation.png                # RQ2 scatter (3 models)
  rq3_cross_domain_heatmap.png       # RQ3 matrix
  rq4_roofline.png                   # RQ4 per-expert roofline
  rq4_triton_speedup.png             # RQ4 kernel bars
```

### 4.2 Paper Structure (MLSys/ES-FoMo)
1. Introduction + problem (InfoKV gap)
2. Related work (PiKV, InfoKV, MoE-nD, TriRoute)
3. RQ1: Granularity ablation (8 vs 60 vs 64 experts)
4. RQ2: Entropy-routing correlation (load-balancing interference)
5. RQ3: Calibration transfer robustness
6. RQ4: Hardware characterization + kernel validation
7. Discussion: When expert-level budgeting helps/fails
8. Conclusion

---

## File Inventory

### New Files to Create (20 files)
| File | Phase | Description |
|------|-------|-------------|
| `experiments/rq1_granularity_comparison.py` | 1 | RQ1 ablation |
| `experiments/rq2_entropy_routing_correlation.py` | 1 | RQ2 correlation |
| `experiments/plot_rq1_rq2.py` | 1 | Visualization |
| `experiments/colab/setup_colab.sh` | 0 | Colab environment |
| `experiments/colab/run_rq1_rq2.py` | 0 | Colab CPU entry |
| `experiments/colab/run_rq3_transfer.py` | 3 | Colab RQ3 |
| `experiments/colab/run_rq4_roofline.py` | 3 | Colab RQ4a |
| `experiments/colab/run_triton_kernel.py` | 3 | Colab RQ4b |
| `requirements/base.txt` | 0 | Core deps |
| `requirements/models.txt` | 0 | Model deps |
| `requirements/kernel.txt` | 0 | Kernel deps |
| `requirements/colab_a100.txt` | 0 | A100 deps |
| `scripts/download_weights.py` | 0 | Weights helper |
| `ekva/kernel/ekva_triton_v1.py` | 3 | Complete Triton v1 |
| `ekva/kernel/ekva_triton_v2.py` | 3 | Complete Triton v2 |
| `ekva/profiling/instrument.py` | 3 | Roofline implementation |
| `ekva/simulator/hook.py` | 2 | Real truncation logic |
| `experiments/week04_wire_hook.py` | 2 | Qwen adapter |
| `experiments/validate_hook.py` | 2 | Hook validation |
| `tests/test_rq1_rq2.py` | 1 | Tests for new scripts |

### Files to Modify
| File | Change |
|------|--------|
| `README.md` | Add Colab quickstart, update badges |
| `PLAN.md` | Sync with this implementation plan |
| `docs/QUICKSTART.md` | Add Colab instructions |
| `ekva/__init__.py` | Export new public APIs |

---

## Validation Gates

| Phase | Gate | Success Criteria |
|-------|------|------------------|
| 0 | Colab runs | `bash setup_colab.sh && python run_rq1_rq2.py` works on fresh Colab T4 |
| 1 | RQ1 | Granularity figure shows clear difference (or honest null) |
| 1 | RQ2 | Pearson/Spearman coeffs for all 3 models + multi-panel figure |
| 2 | Hook | Baseline PPL match < 0.01; 16 combos produce real metrics |
| 3 | RQ3 | 28-cell cross-domain matrix complete |
| 3 | RQ4a | Roofline plot with per-expert points + A100 ceilings |
| 3 | RQ4b | v1 correct at FullKV; v2 speedup on memory-bound experts |

---

## Immediate Next Steps (Start Today)

1. **Create `experiments/rq1_granularity_comparison.py`** (~200 lines)
2. **Create `experiments/rq2_entropy_routing_correlation.py`** (~150 lines)
3. **Create `experiments/plot_rq1_rq2.py`** (~100 lines)
4. **Run both locally on RTX 3050 (CPU mode)** — verify outputs
5. **Create `experiments/colab/setup_colab.sh`** — test on Colab T4 free tier
6. **Wire Qwen hook** — after RQ1/RQ2 done, before A100 campaign

---

## Git Commit Strategy

```
commit 1: "docs: add RESEARCH.md, update PLAN.md, add implementation plan"
commit 2: "feat: RQ1 layer vs expert granularity ablation script"
commit 3: "feat: RQ2 entropy-routing correlation analysis script"
commit 4: "feat: visualization scripts for RQ1/RQ2"
commit 5: "feat: Colab setup scripts and requirements"
commit 6: "feat: Qwen MoE hook adapter for EKVACacheHook"
commit 7: "feat: hook validation script"
commit 8: "feat: Triton v1 kernel with variable tile count"
commit 9: "feat: Triton v2 kernel with fused gather"
commit 10: "feat: roofline profiling instrumentation"
commit 11: "test: add tests for RQ1/RQ2/hook/kernels"
commit 12: "docs: update README, QUICKSTART, PLAN.md"
```
