# EKVA Implementation Plan — Detailed Action Plan

## Context
- **Hardware:** Local RTX 3050 (6GB) only for now; Colab A100 later
- **Calibration:** All 3 models done (outputs exist)
- **Hook:** Not wired (placeholder only)
- **Target:** Full paper with hardware (MLSys) — two-phase approach
- **Novel angles:** Skip for now per user decision

## Execution Phases

### Phase 0: Colab Hardening (Week 0, 2 days)
- [ ] Create `experiments/colab/setup_colab.sh` - installs deps, downloads weights
- [ ] Create `experiments/colab/run_rq1_rq2.py` - CPU-only entrypoint
- [ ] Test on Colab T4 free tier (verify < 15GB VRAM for Qwen)
- [ ] Create `requirements/colab_a100.txt` - A100-specific deps (triton, nsight)

### Phase 1: RQ1 + RQ2 (Week 1-2, local CPU)
- [ ] Create `experiments/rq1_granularity_comparison.py`
  - Load calibration outputs from `output/*_phase1.pt`
  - Compute layer-aggregated entropy (mean per layer)
  - Derive budgets at layer level vs expert level (same total budget)
  - Run both through `run_policy_eviction_grid` with simulator
  - Output: figure + `output/rq1_granularity.pt`
- [ ] Create `experiments/rq2_entropy_routing_correlation.py`
  - Load all 3 calibration outputs
  - Compute Pearson/Spearman per model per layer
  - Multi-panel figure: entropy vs routing frequency
  - Output: correlation table + `output/rq2_correlation.pt`
- [ ] Run both locally on RTX 3050 (CPU mode)
- [ ] Validate outputs make sense (figures, numbers)

### Phase 2: Hook Wiring (Week 2-3, local RTX 3050)
- [ ] Identify Qwen1.5-MoE MoE layer class names
  - Load model, `print([m.__class__.__name__ for m in model.modules() if 'moe' in m.__class__.__name__.lower()])`
- [ ] Wire `experiments/week04_wire_hook.py` for Qwen
  - Implement `_make_moe_hook` with real truncation call
  - Call `hook.truncate(expert_id, k, v, attn)` in MoE forward
- [ ] Sanity check: FullKV budget → baseline PPL within 0.01
- [ ] Run 16-combo grid on Qwen with real truncation
- [ ] Output: `output/week04/phase2_hook_validation.json` with real metrics

### Phase 3: A100 Campaign (When A100 available, ~1 week)
**Prerequisite:** Phase 1-2 complete + Colab A100 access confirmed

- [ ] RQ3: Cross-domain calibration transfer
  - Calibrate on WikiText / Code / Long-context QA
  - Evaluate on LongBench / RULER / Needle
  - 28-cell degradation matrix
  - Script: `experiments/colab/run_rq3_transfer.py`
- [ ] RQ4a: Per-expert roofline
  - PyTorch Profiler + Nsight on Mixtral-8x7B
  - Extract FLOPs/Bytes per expert per layer
  - Roofline plot with A100 ceilings
  - Script: `experiments/colab/run_rq4_roofline.py`
- [ ] RQ4b: Triton kernel
  - `ekva/kernel/ekva_triton_v1.py`: variable tile count
  - `ekva/kernel/ekva_triton_v2.py`: fused index selection
  - Correctness gate: match FA2 at FullKV
  - Profile: kernel time, HBM reads, SM occupancy
  - Script: `experiments/colab/run_triton_kernel.py`

### Phase 4: Paper Assembly (Week after A100)
- [ ] Assemble figures: RQ1, RQ2, RQ3 matrix, RQ4 roofline, RQ4 kernel
- [ ] Write paper using abstract template from `ekva_research_strategy.md`
- [ ] Target venues: MLSys (with kernel), ES-FoMo/ENLSP (without)
- [ ] Submit to arXiv regardless

## File Inventory

### New Files to Create
| Path | Purpose |
|------|---------|
| `experiments/rq1_granularity_comparison.py` | RQ1 analysis |
| `experiments/rq2_entropy_routing_correlation.py` | RQ2 analysis |
| `experiments/colab/setup_colab.sh` | Colab environment |
| `experiments/colab/run_rq1_rq2.py` | Colab RQ1+RQ2 |
| `experiments/colab/run_rq3_transfer.py` | Colab RQ3 |
| `experiments/colab/run_rq4_roofline.py` | Colab RQ4a |
| `experiments/colab/run_triton_kernel.py` | Colab RQ4b |
| `requirements/colab_a100.txt` | A100 deps |
| `scripts/download_weights.py` | Weights helper |
| `ekva/kernel/ekva_triton_v1.py` | Triton v1 |
| `ekva/kernel/ekva_triton_v2.py` | Triton v2 |

### Files to Modify
| Path | Change |
|------|--------|
| `experiments/week04_wire_hook.py` | Qwen-specific hook adapter |
| `ekva/simulator/hook.py` | Real truncation logic |
| `ekva/profiling/instrument.py` | Complete roofline |
| `README.md` | Colab quickstart section |

## Validation Gates

| Phase | Gate | Criteria |
|-------|------|----------|
| 0 | Colab runs | `bash setup_colab.sh && python run_rq1_rq2.py` works fresh |
| 1 | RQ1 | Granularity figure shows clear difference or honest null |
| 1 | RQ2 | Correlation coeffs for all 3 models, multi-panel figure |
| 2 | Hook | Baseline PPL match < 0.01; 16 combos produce real metrics |
| 3 | RQ3 | 28-cell cross-domain matrix complete |
| 3 | RQ4a | Roofline plot with per-expert points + A100 ceilings |
| 3 | RQ4b | v1 correct at FullKV; v2 speedup on memory-bound experts |

## Risks & Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Hook wiring fails on Qwen | Medium | Fallback: simulator-only results (fallback_branches.py) |
| No A100 access for weeks | Medium | Phase 1-2 → workshop paper (ES-FoMo) without RQ3/RQ4 |
| Triton kernel bugs | High | Start v1 only; v2 stretch; base on reference_flashattn2.py |
| Colab T4 OOM on Qwen | Low | cpu_offload + 4-bit quantization if needed |
| Calibration/reality gap | Medium | Validate hook on Qwen first; adjust prompts if needed |

## Open Questions for Implementation Agent

1. **Qwen MoE class names:** Exact HF class names for MoE block, router, experts
2. **Colab A100 budget:** Hours available? (RQ3~6h + RQ4a~4h + RQ4b~8h = ~18h)
3. **Nsight on Colab:** `ncu` needs sudo - alternative: PyTorch Profiler only?
4. **Triton version:** Colab A100 preinstalled version compatible with reference?
5. **Paper deadline:** Target venue deadline to plan backwards from?

## Immediate Next Steps (Start Today)
1. Write `experiments/rq1_granularity_comparison.py` (~200 lines)
2. Write `experiments/rq2_entropy_routing_correlation.py` (~150 lines)
3. Run both locally (CPU) - verify outputs
4. Write `experiments/colab/setup_colab.sh` - test on Colab T4
5. Wire Qwen hook after RQ1/RQ2 done
