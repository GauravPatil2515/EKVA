# EKVA Implementation Plan

## Context & Constraints

| Constraint | Value |
|------------|-------|
| **Local GPU** | RTX 3050 6GB (cpu_offload only) |
| **Target** | MLSys workshop paper (requires A100 for roofline + Triton) |
| **Calibration data** | All 3 models done (*_phase1.pt exists) |
| **Hook status** | Placeholder only - needs model-specific adapter |
| **Deadline** | No fixed deadline |
| **Novel angles** | Skip for now (focus RQ1-RQ4) |

## Phased Strategy

### Phase 0: Repo Hardening for Colab (Week 0 - 2 days)
Make repo clone-and-run on Colab A100.

### Phase 1: CPU-Only RQ1 + RQ2 (Week 1-2)
- RQ1: Layer vs Expert granularity ablation (simulator only)
- RQ2: Entropy–Routing correlation analysis (all 3 models)
- Deliverable: `output/rq1_granularity_comparison.pt`, `output/rq2_correlation.pt` + figures

### Phase 2: Hook Wiring + Real Validation on 3050 (Week 2-3)
- Wire EKVACacheHook for Qwen1.5-MoE-A2.7B on 3050 with cpu_offload
- Run RQ1/RQ2 with real truncated-cache PPL
- Deliverable: `output/week04/phase2_hook_validation.json` with real metrics

### Phase 3: Colab A100 Campaign (When available - ~1 week)
- RQ3: Cross-domain calibration transfer (T4 or A100)
- RQ4a: Per-expert roofline on Mixtral-8x7B (A100 + Nsight)
- RQ4b: Triton kernel v1/v2 + profiling (A100)
- RQ3/RQ4 deliverables

### Phase 4: Paper Assembly (Week after A100)
- Assemble figures, write paper, submit to MLSys/ES-FoMo

---

## Detailed Tasks

### Phase 0: Colab-Ready Repo

#### 0.1 Add Colab launch scripts
```
experiments/colab/
  setup_colab.sh          # pip installs, git clone, weights download
  run_rq1_rq2.py          # CPU work (can run on Colab T4 free)
  run_rq3_transfer.py     # Cross-domain calibration (needs T4)
  run_rq4_roofline.py     # Per-expert roofline (needs A100)
  run_triton_kernel.py    # Triton v1/v2 (needs A100)
```

#### 0.2 Add requirements files
```
requirements/
  base.txt                # torch, numpy, matplotlib, pyyaml, tqdm, scipy
  models.txt              # transformers, datasets, accelerate
  kernel.txt              # triton (A100 only)
  colab_a100.txt          # all of above + nvidia-nsight-cli
```

#### 0.3 Add weights download helper
```
scripts/download_weights.py
  --model qwen1.5-moe-a2.7b|mixtral-8x7b|deepseek-moe-16b
  --device cuda|cpu
```

#### 0.4 Fix import paths for Colab
- Ensure `sys.path.append` works from `/content/EKVA`
- Use relative imports in `ekva/` modules

#### 0.5 Add .gitignore for Colab artifacts
```
/content/EKVA/output/
/content/EKVA/models/
*.ncu-rep
```

---

### Phase 1: CPU-Only RQ1 + RQ2

#### 1.1 RQ1: Layer vs Expert Granularity (`experiments/rq1_granularity_comparison.py`)
```python
# Input: calibration outputs (*_phase1.pt)
# Method:
#   1. Load entropy_map for each model
#   2. Compute layer-aggregated budgets: avg entropy per layer -> proportional
#   3. Compute expert-level budgets: current EKVA method
#   4. Run both through simulator (run_policy_eviction_grid)
#   5. Compare PPL degradation curves across budget fractions (10%-80%)
# Output: output/rq1_granularity_comparison.pt + figure
```

**Key insight:** Layer-aggregated = average entropy across experts per layer, then allocate budget per layer. Expert-level = current per-expert allocation.

#### 1.2 RQ2: Entropy–Routing Correlation (`experiments/rq2_entropy_routing_correlation.py`)
```python
# Input: calibration outputs (*_phase1.pt) for all 3 models
# Method:
#   1. Extract per-expert: entropy (avg_entropy.mean()), routing_count
#   2. Compute Pearson + Spearman per model, per layer
#   3. Plot: entropy vs routing_count scatter per model
#   4. Test hypothesis: load-balanced MoE → anti-correlation
# Output: output/rq2_correlation.pt + multi-panel figure
```

#### 1.3 Visualization script (`experiments/plot_rq1_rq2.py`)
- Granularity comparison bars per model per budget fraction
- Entropy-routing scatter with correlation coefficients
- Save as PNG + PDF for paper

---

### Phase 2: Hook Wiring on RTX 3050 (Qwen Only)

#### 2.1 Model-specific hook adapter (`experiments/week04_wire_hook.py`)
```python
# Current state: EKVACacheHook has placeholder _make_moe_hook
# Need: Model-specific forward wrapper that:
#   1. Intercepts MoE layer forward
#   2. Gets routed expert IDs from router logits
#   3. For each routed expert: hook.truncate(k, v, attn)
#   4. Reassembles past_key_values with truncated KV
# Target: Qwen1.5-MoE-A2.7B (fits 3050 with cpu_offload)
```

#### 2.2 Qwen MoE layer identification
- Qwen uses `Qwen1_5MoeSparseMoeBlock` (check HF model)
- Router: `gate` or `gate_proj` module
- Experts: `experts` ModuleList
- Attention: standard `Qwen1_5Attention`

#### 2.3 Validation script (`experiments/validate_hook.py`)
```python
# 1. Baseline PPL (full KV) - must match within 0.01
# 2. 16 combos: 4 policies × 4 evictions at 20% budget
# 3. Compare simulator PPL vs real truncated PPL
# 4. Output: phase2_hook_validation.json with real metrics
```

#### 2.4 CPU offload config for 3050
```python
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen1.5-MoE-A2.7B",
    torch_dtype=torch.float16,
    device_map="auto",           # enables cpu_offload
    low_cpu_mem_usage=True,
    max_memory={0: "5GB", "cpu": "16GB"}
)
```

---

### Phase 3: Colab A100 Campaign (When Available)

#### 3.1 RQ3: Cross-Domain Calibration Transfer (`experiments/run_rq3_transfer.py`)
```python
# Calibration domains: WikiText, Code, Long-context QA, Math
# Evaluation domains: LongBench (4 tasks), RULER (2 tasks), Needle, WikiText PPL
# Matrix: 4 × 7 = 28 runs per model
# Metric: PPL degradation = (truncated PPL - baseline PPL) / baseline PPL
# Output: cross_domain_degradation_matrix.pt + heatmap figure
```

#### 3.2 RQ4a: Per-Expert Roofline (`experiments/run_rq4_roofline.py`)
```python
# Uses ekva/profiling/instrument.py (scaffold exists)
# 1. PyTorch Profiler + Nsight Compute on Mixtral-8x7B
# 2. Instrument per-expert attention forward
# 3. Extract: FLOPs, HBM bytes per expert
# 4. Compute Arithmetic Intensity = FLOPs / Bytes
# 5. Roofline plot: x=AI, y=attained GFLOP/s, color=entropy bucket
# 6. Overlay A100 ceilings: 312 TFLOP/s fp16, 1.55 TB/s HBM
```

#### 3.3 RQ4b: Triton Kernel v1 (`ekva/kernel/ekva_triton_v1.py`)
```python
# Based on reference_flashattn2.py
# Modification: variable tile count per expert
#   - Input: KV_budget[expert_id] tensor
#   - Tile loop stops after budget // BLOCK_N tiles
#   - Correctness test: match FA2 at budget=FullKV
# Benchmark: kernel time, HBM reads vs baseline FA2
```

#### 3.4 RQ4b: Triton Kernel v2 (`ekva/kernel/ekva_triton_v2.py`)
```python
# Extends v1: fused important-token index selection
#   - Precompute indices per expert (from attention scores)
#   - Gather only those KV during tile loop
# Nsight Compute: HBM bandwidth, SM occupancy, kernel time
# Compare: baseline vs v1 vs v2 per expert class (low/mid/high entropy)
```

---

### Phase 4: Paper Assembly

#### 4.1 Figure pipeline
```
output/
  rq1_granularity_comparison.png     # RQ1 bars
  rq2_correlation.png                # RQ2 scatter panels
  rq3_cross_domain_heatmap.png       # RQ3 matrix
  rq4_roofline.png                   # RQ4 roofline
  rq4_triton_speedup.png             # RQ4 kernel bars
```

#### 4.2 Paper structure (MLSys/ES-FoMo)
1. Introduction + problem statement
2. Related work (PiKV, InfoKV, MoE-nD, TriRoute gaps)
3. RQ1: Granularity ablation
4. RQ2: Entropy-routing correlation (the "why" for destabilization)
5. RQ3: Calibration transfer robustness
6. RQ4: Hardware characterization + kernel validation
7. Discussion: When expert-level budgeting helps/fails
8. Conclusion

---

## Code Changes Required

### New Files
| File | Purpose |
|------|---------|
| `experiments/rq1_granularity_comparison.py` | RQ1 ablation |
| `experiments/rq2_entropy_routing_correlation.py` | RQ2 correlation |
| `experiments/plot_rq1_rq2.py` | Visualization |
| `experiments/colab/setup_colab.sh` | Colab environment |
| `experiments/colab/run_rq1_rq2.py` | Colab entrypoint |
| `experiments/colab/run_rq3_transfer.py` | Colab RQ3 |
| `experiments/colab/run_rq4_roofline.py` | Colab RQ4a |
| `experiments/colab/run_triton_kernel.py` | Colab RQ4b |
| `scripts/download_weights.py` | Weights helper |
| `requirements/colab_a100.txt` | Colab A100 deps |
| `ekva/kernel/ekva_triton_v1.py` | Triton v1 |
| `ekva/kernel/ekva_triton_v2.py` | Triton v2 |

### Modified Files
| File | Change |
|------|--------|
| `experiments/week04_wire_hook.py` | Qwen-specific hook adapter |
| `ekva/simulator/hook.py` | Remove placeholder, add real truncation logic |
| `ekva/profiling/instrument.py` | Complete roofline instrumentation |
| `ekva/calibration/entropy.py` | Add token-type entropy for Novel B (future) |
| `README.md` | Add Colab quickstart section |

---

## Validation Gates

| Phase | Gate | Success Criteria |
|-------|------|------------------|
| 0 | Colab runs | `bash setup_colab.sh && python run_rq1_rq2.py` works on fresh Colab T4 |
| 1 | RQ1 | Granularity figure shows clear difference (or honest null result) |
| 1 | RQ2 | Correlation coefficients computed for all 3 models |
| 2 | Hook | Baseline PPL matches within 0.01; 16 combos produce real metrics |
| 3 | RQ3 | 28-cell cross-domain matrix complete |
| 3 | RQ4a | Roofline plot with per-expert points + A100 ceilings |
| 3 | RQ4b | Triton v1 correct at FullKV; v2 shows speedup on memory-bound experts |

---

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Hook wiring fails on Qwen | Medium | Fallback: report simulator results only (existing fallback_branches.py) |
| No A100 access for weeks | Medium | Phase 1-2 produce workshop paper (ES-FoMo) without RQ3/RQ4 |
| Triton kernel bugs | High | Start with v1 only; v2 is stretch; use reference_flashattn2.py as base |
| Colab T4 OOM on Qwen | Low | Use cpu_offload + 4-bit quantization if needed |
| Calibration data doesn't match real runs | Medium | Validate hook on Qwen first; adjust calibration prompts if needed |

---

## Execution Order

```
Week 0 (2 days):  Phase 0 - Colab hardening
Week 1-2:         Phase 1 - RQ1 + RQ2 (CPU, local)
Week 2-3:         Phase 2 - Hook wiring on 3050 (Qwen)
Week 3+:          Wait for A100 access
When A100:        Phase 3 - RQ3 + RQ4 (Colab A100, ~1 week)
Week after:       Phase 4 - Paper assembly
```

---

## Open Questions (for implementation agent)

1. **Qwen MoE layer class name:** Need exact HF class names for `Qwen1_5MoeSparseMoeBlock`, router, experts. Check via `print(model)` after loading.

2. **Colab A100 budget:** How many hours available? RQ3 (~6h) + RQ4a (~4h) + RQ4b (~8h) = ~18h total.

3. **Nsight Compute on Colab:** `ncu` requires `sudo` - may need Colab Pro+ or local profiling. Alternative: PyTorch Profiler only.

4. **Triton version:** Colab A100 has Triton preinstalled? Check version compatibility with reference_flashattn2.py.

5. **Paper venue deadline:** MLSys deadline typically Feb/Mar. ES-FoMo typically Jun/Jul. Plan backwards from target.

---

## Immediate Next Steps (Start Today)

1. **Create `experiments/rq1_granularity_comparison.py`** - uses existing calibration outputs, simulator, ~200 lines
2. **Create `experiments/rq2_entropy_routing_correlation.py`** - uses existing calibration outputs, ~150 lines
3. **Run both locally on RTX 3050 (CPU mode)** - verify outputs
4. **Create `experiments/colab/setup_colab.sh`** - test on Colab T4 free tier
5. **Wire hook for Qwen** - after RQ1/RQ2 done, before A100 campaign
