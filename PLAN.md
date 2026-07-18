# EKVA — 12-Week Research Plan (Combination Matrix)

This is the working plan for the EKVA paper. Every week maps to cells in the
master matrix below. Fallback branches are baked in at each decision point.

## Master Combination Matrix

| Axis | Options |
|---|---|
| **Models** | Mixtral-8x7B, DeepSeek-V2 (or DeepSeek-MoE-16B if V2 too heavy), Qwen1.5-MoE-A2.7B (fits 3050) |
| **Budget policies** | Uniform, EKVA-entropy, EKVA-multi-signal (entropy+routing+specialization), Random, SnapKV-style, PyramidKV-style, DynamicKV-style |
| **Eviction strategies** | Recency (FIFO), Attention-score, Random, Hybrid (recency+attention) |
| **Total budget fractions** | 10%, 20%, 30%, 40%, 60%, 80% of FullKV |
| **Benchmarks** | LongBench (subset), RULER, Needle-in-Haystack, InfiniteBench, plain perplexity (WikiText/C4) |
| **Hardware profiling** | PyTorch Profiler, Nsight Systems, Nsight Compute (roofline) |
| **Kernel variants** | Baseline FA2, EKVA-Triton (variable tile), EKVA-Triton + fused eviction |

---

## Weeks 1–2: Real-model Phase 1 (entropy + routing signal)
**Goal:** Replace mock calibration with real signal across all 3 candidate models.
- Run `experiments/week01_02_calibration.py` on:
  - **Qwen1.5-MoE-A2.7B** (3050 / Colab T4 — do this first)
  - **Mixtral-8x7B** (Colab A100)
  - **DeepSeek-MoE-16B** (Colab A100, if V2 too large)
- Vary prompt sets: general / long-context / code / math (see `configs/models.yaml`).
- Produce entropy heatmaps + budget scatter per model (`experiments/plot_calibration.py`).
- **Decision point:** if entropy varies meaningfully across experts in ≥2/3 models → proceed. If not → pivot to routing-frequency / specialization signal.
- Deliverable: `output/{model}_{promptset}_phase1.pt` + figures.

## Week 3: Correlation & sensitivity study
**Goal:** Prove entropy *predicts* KV truncation sensitivity (causal, not just correlational).
- Per-expert truncation sweep: force each expert's budget to {25%,50%,75%,100%}, others full.
- Measure PPL delta per expert per level. Plot entropy vs sensitivity slope.
- Also test routing-frequency vs sensitivity and combined-score vs sensitivity.
- Compute Pearson/Spearman for entropy, routing, combined.
- Deliverable: sensitivity table + correlation coefficients.
- Script: `experiments/week03_sensitivity.py`.

## Week 4: Wire the real KV truncation hook
**Goal:** Make Phase 2 scientifically real (currently PPL is computed without truncation).
- Implement `past_key_values` interception per MoE layer routing each expert's K/V through its `ExpertKVBuffer` (`ekva/simulator/hook.py`).
- **Sanity check:** budget=FullKV must reproduce baseline PPL exactly (small model Qwen-MoE).
- Run 4 evictions × 4 policies = 16 combos on the smallest model.
- Deliverable: `output/week04/phase2_hook_validation.json`.

## Weeks 5–6: Full Phase 2 benchmark sweep
**Goal:** 16-combo grid across budget fractions and real benchmarks.
- Benchmarks: LongBench (4–6 tasks), RULER, Needle-in-Haystack.
- Budget fractions: 10%, 20%, 40%, 60%, 80%.
- Record PPL / task accuracy (EM/F1) / actual memory used.
- On Mixtral-8x7B and DeepSeek-MoE-16B (small model was hook validation only).
- **Decision point:** if EKVA doesn't beat Uniform at any fraction → go to Week 7 multi-signal.
- Deliverable: master results CSV (`Method | Memory% | PPL | Throughput`).

## Week 7: Multi-signal EKVA upgrade (if needed) + finalize best policy
- Add specialization score (token-type diversity) + roofline-position placeholder to `derive_kv_budget`.
- Re-run best 2–3 combos from Weeks 5–6 with multi-signal.
- Lock the final EKVA policy formula.
- Script: `experiments/week07_multisignal.py`.

## Week 8: Roofline instrumentation setup
**Goal:** Prove experts occupy different hardware regimes.
- Set up PyTorch Profiler + Nsight Compute on Colab A100 (H100 if available).
- Instrument per-expert attention to extract FLOPs and HBM bytes.
- Run on Mixtral-8x7B with Phase-1 prompts.
- Compute Arithmetic Intensity = FLOPs / Bytes per expert.
- Deliverable: raw per-expert FLOPs/bytes CSV.

## Week 9: Roofline plots & correlation with entropy
- Build roofline plot: x=AI, y=attained GFLOP/s, colored by entropy bucket.
- Overlay hardware ceiling (peak FLOPs, peak bandwidth).
- Test entropy vs AI, routing vs AI, combined vs AI; pick cleanest separation.
- **Decision point:** if experts cluster clearly memory- vs compute-bound → strong paper. If overlap → algorithmic framing still stands alone.
- Deliverable: roofline figure (candidate Figure 3) + correlation analysis.

## Week 10: Triton kernel prototype v1
- Start from reference Triton FA2 (`ekva/kernel/reference_flashattn2.py`).
- Modify tile loop to accept `KV_budget[expert_id]`, stop after `budget // BLOCK_N` tiles.
- **Correctness:** match standard attention at budget=FullKV.
- Benchmark vs baseline FA2 (kernel time, HBM reads).
- Deliverable: `ekva/kernel/ekva_triton_v1.py` + correctness test + timing.

## Week 11: Fused kernel v2 + hardware profiling
- Extend to accept precomputed "important token indices" per expert; gather only those KV during tile loop.
- Profile with Nsight Compute: HBM bandwidth, SM occupancy, kernel time — baseline vs v2 per expert class.
- Try (a) static budget only, (b) budget + fused index selection.
- Deliverable: `ekva/kernel/ekva_triton_v2.py` + Nsight report.

## Week 12: End-to-end evaluation & writeup prep
- Full pipeline on Mixtral-8x7B: calibration → EKVA budget → Triton kernel → LongBench/RULER.
- Final table: FullKV, Uniform, SnapKV, PyramidKV, EKVA-software, EKVA-kernel (memory%, PPL, throughput).
- Consolidate figures; draft paper skeleton.
- Deliverable: final results CSV + all figures + paper draft + README update.

---

## Fallback branches (if a phase fails)

| Failure | Fallback |
|---|---|
| Entropy signal weak (W3) | Switch primary signal to routing-frequency / combined multi-signal (W7 reserved). |
| EKVA doesn't beat Uniform (W6) | Hybrid of PyramidKV layer curve + EKVA expert allocation (novel on its own). |
| Experts don't roofline-separate (W9) | Reframe as "algorithmic KV allocation for MoE"; drop strongest hardware claim. |
| Triton kernel too hard (W10–11) | Report software-simulator results only; describe kernel as "proposed implementation." |

Implemented in `experiments/fallback_branches.py`.
