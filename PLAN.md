# EKVA — Reframed 12-Week Research Plan (Empirical Characterization)

> **Framing:** This is an empirical characterization and systems validation project,
> not a novel-method paper. The contribution is answering open questions about how
> entropy-based KV budgeting behaves in sparse MoE models across architectures,
> granularities, and hardware regimes. See `docs/RESEARCH.md` for the full
> prior-work analysis and RQ justification.

This is the working plan for the EKVA paper. Every week maps to cells in the
master matrix below. Fallback branches are baked in at each decision point.

## Master Combination Matrix

| Axis | Options |
|---|---|
| **Models** | Mixtral-8x7B (primary), Qwen1.5-MoE-A2.7B (cheap validation), DeepSeek-MoE-16B (stretch) |
| **Budget policies** | Uniform, EKVA-entropy, EKVA-multi-signal, Random |
| **Granularity** | Layer-level (CAKE/MEDA-style) vs. Expert-level (EKVA) — **RQ1** |
| **Eviction strategies** | Recency (FIFO), Attention-score, Random, Hybrid (recency+attention) |
| **Budget fractions** | 10%, 20%, 40%, 60%, 80% of FullKV |
| **Benchmarks** | LongBench (2 tasks), RULER (2 tasks), Needle-in-Haystack, WikiText/C4 perplexity |
| **Hardware profiling** | PyTorch Profiler, Nsight Systems, Nsight Compute (roofline) — **RQ4** |
| **Kernel variants** | Baseline FA2, EKVA-Triton (variable tile), EKVA-Triton + fused eviction |

---

## Prior Work Context (see `docs/RESEARCH.md` for full analysis)

The entropy → proportional budget mechanism is already published (PiKV PERouter,
CAKE, Ada-KV, MEDA, InfoKV). The open gaps this project addresses:

1. **No controlled study of per-expert vs. per-layer granularity** in sparse MoE models.
2. **No cross-architecture empirical comparison** across models with different expert counts (8 vs. 60 vs. 64).
3. **No roofline characterization of individual MoE experts** (compute-bound vs. memory-bound per expert).
4. **No calibration transfer study** (does entropy-budgeting generalize across domains?).

The paper is built around four research questions (RQ1–RQ4), not a claimed novel method.

---

## Weeks 1–2: RQ1 — Layer vs. Expert Granularity (CPU only)

**Goal:** Answer the most important empirical question first — does expert-level
budgeting beat layer-level budgeting at equal total budget?

- Run existing calibration on Qwen1.5-MoE-A2.7B (cheap, fits 3050).
- Compute **layer-aggregated entropy budgets** (average per layer across experts).
- Compute **expert-level entropy budgets** (what EKVA currently does).
- Run both through the software simulator with the same total budget.
- Compare PPL under truncation for both approaches.
- Also compare multi-signal policies at both granularities.
- **Decision point:** if expert-level wins → proceed with expert-level as the main story. If not → pivot to "when does granularity matter" as the paper's angle.
- Deliverable: `output/rq1_granularity_comparison.pt` + figure + results table.
- Script: `experiments/rq1_granularity_comparison.py` (new, uses existing calibration + simulator).

## Weeks 2–3: RQ2 — Entropy vs. Routing Frequency Correlation

**Goal:** Map the relationship between attention entropy and routing frequency across 3 MoE architectures.

- Use existing calibration data across Qwen (60 experts), Mixtral (8 experts), DeepSeek (64 experts).
- Compute Pearson/Spearman correlation between per-expert entropy and per-expert routing frequency.
- Plot the relationship per model — are high-entropy experts also frequently routed?
- Test whether the relationship differs across models with different expert counts.
- **Decision point:** if entropy and routing are anti-correlated in some models → this explains why pure-entropy methods destabilize on those architectures (per InfoKV's admitted limitation). Strongest diagnostic finding.
- Deliverable: correlation table + multi-panel figure + `output/rq2_correlation.pt`.
- Script: `experiments/rq2_entropy_routing_correlation.py` (new).

## Week 4: Wire the real KV truncation hook

**Goal:** Make Phase 2 scientifically real (currently PPL is computed without truncation).
- Implement `past_key_values` interception per MoE layer routing each expert's K/V through its `ExpertKVBuffer` (`ekva/simulator/hook.py`).
- **Sanity check:** budget=FullKV must reproduce baseline PPL exactly (small model Qwen-MoE).
- Run 4 evictions × 4 policies = 16 combos on the smallest model.
- Deliverable: `output/week04/phase2_hook_validation.json`.

## Weeks 5–6: Targeted benchmark sweep (2 tasks, not full subset)

**Goal:** Validate the RQ1/RQ2 findings on real benchmarks with real truncation.
- Benchmarks: **2 tasks only** (e.g., Needle-in-Haystack + one LongBench task).
- Budget fractions: 10%, 20%, 40%, 60%, 80%.
- Record PPL / task accuracy / actual memory used.
- On Mixtral-8x7B and Qwen1.5-MoE-A2.7B (two well-executed models > three rushed ones).
- **Decision point:** if EKVA doesn't beat Uniform at any fraction → go to Week 7 multi-signal.
- Deliverable: targeted results CSV (`Method | Memory% | PPL | Throughput`).

## Week 7: Multi-signal EKVA upgrade (if needed) + finalize best policy
- Add specialization score (token-type diversity) + roofline-position placeholder to `derive_kv_budget`.
- Re-run best 2–3 combos from Weeks 5–6 with multi-signal.
- Lock the final EKVA policy formula.
- Script: `experiments/week07_multisignal.py`.

## Week 8: Roofline instrumentation setup — **RQ4**

**Goal:** Prove experts occupy different hardware regimes.
- Set up PyTorch Profiler + Nsight Compute on Colab A100 (H100 if available).
- Instrument per-expert attention to extract FLOPs and HBM bytes.
- Run on Mixtral-8x7B with Phase-1 prompts.
- Compute Arithmetic Intensity = FLOPs / Bytes per expert.
- Deliverable: raw per-expert FLOPs/bytes CSV.

## Week 9: Roofline plots & correlation with entropy — **RQ4**
- Build roofline plot: x=AI, y=attained GFLOP/s, colored by entropy bucket.
- Overlay hardware ceiling (peak FLOPs, peak bandwidth).
- Test entropy vs AI, routing vs AI, combined vs AI; pick cleanest separation.
- **Decision point:** if experts cluster clearly memory- vs compute-bound → strong paper. If overlap → algorithmic framing still stands alone.
- Deliverable: roofline figure (candidate Figure 3) + correlation analysis.

## Week 10: Triton kernel prototype v1 — **RQ4**
- Start from reference Triton FA2 (`ekva/kernel/reference_flashattn2.py`).
- Modify tile loop to accept `KV_budget[expert_id]`, stop after `budget // BLOCK_N` tiles.
- **Correctness:** match standard attention at budget=FullKV.
- Benchmark vs baseline FA2 (kernel time, HBM reads).
- Deliverable: `ekva/kernel/ekva_triton_v1.py` + correctness test + timing.

## Week 11: Fused kernel v2 + hardware profiling — **RQ4**
- Extend to accept precomputed "important token indices" per expert; gather only those KV during tile loop.
- Profile with Nsight Compute: HBM bandwidth, SM occupancy, kernel time — baseline vs v2 per expert class.
- Try (a) static budget only, (b) budget + fused index selection.
- Deliverable: `ekva/kernel/ekva_triton_v2.py` + Nsight report.

## Week 12: End-to-end evaluation & writeup prep
- Full pipeline on Mixtral-8x7B: calibration → EKVA budget → Triton kernel → 2 benchmark tasks.
- Final table: FullKV, Uniform, EKVA-entropy, EKVA-multi-signal, EKVA-software, EKVA-kernel (memory%, PPL, throughput).
- Consolidate figures; draft paper skeleton framed around RQ1–RQ4.
- Deliverable: final results CSV + all figures + paper draft + README update.

---

## Fallback branches (if a phase fails)

| Failure | Fallback |
|---|---|
| RQ1 shows no granularity advantage (W1-2) | Pivot paper to "when does expert-level granularity matter" — the negative result is publishable. |
| Entropy signal weak (W3) | Switch primary signal to routing-frequency / combined multi-signal (W7 reserved). |
| EKVA doesn't beat Uniform (W6) | Hybrid of PyramidKV layer curve + EKVA expert allocation (novel on its own). |
| Experts don't roofline-separate (W9) | Reframe as "algorithmic KV allocation for MoE"; drop strongest hardware claim. |
| Triton kernel too hard (W10–11) | Report software-simulator results only; describe kernel as "proposed implementation." |

Implemented in `experiments/fallback_branches.py`.
