# EKVA — Reframed Research Framing & Strategy

> **Status:** Empirical characterization and systems validation project (not a novel-method paper).
> This document captures the reframed research direction, prior-work analysis, 5 concrete innovations,
> and actionable strategy based on deep literature review and external review.
> See `docs/review.md` for the external review that prompted this reframing.
> See `ekva_research_strategy.md` for the full strategy document with 5 innovations and 3 novel angles.

---

## Why the Original Framing Doesn't Work

The original README framing — "the core novelty is entropy-driven per-expert KV budget allocation" — is no longer accurate. The mechanism (entropy → proportional budget) has been published in multiple forms:

| Prior Work | Venue | Year | What They Do | Gap They Leave |
|---|---|---|---|---|
| **PiKV** (Liu et al.) | arXiv 2508.06526, ICML 2025 ES-FoMo | 2025 | MoE KV cache management with EPLB and BARouter. Code shipped Sept 2025. | No controlled ablation of per-expert vs per-layer budgeting for *inference quality*. Targets serving latency, not accuracy-budget trade-off. |
| **CAKE** | arXiv | 2025 | Layer-wise entropy-guided KV budget allocation | Dense models only. No MoE validation. |
| **Ada-KV** | arXiv | 2025 | Per-head adaptive budget allocation using attention entropy | Dense models only. No cross-architecture comparison. |
| **MEDA** | arXiv | 2025 | Entropy-based dynamic KV budget allocation | Dense models only. No hardware profiling. |
| **InfoKV** (Lin et al.) | arXiv 2606.26875 | 2026 | Information-aware KV compression with adaptive layer-wise entropy budgeting | Explicitly admits destabilization on some models. Flags "architecture-aware allocation strategies" as open. |
| **TriRoute** (Balashov & Ponomarova) | arXiv 2607.06601 | 2026 | Unified learned routing over attention, experts, and KV-cache precision | Only tested on 160M–1.3B. High training complexity. Kernel support gap explicitly noted. |
| **MoE-nD** (Sun et al.) | arXiv 2604.17695 | 2026 | Per-layer multi-axis KV cache compression routing | Per-layer, not per-expert. No roofline characterization per expert. |
| **EntropyInfer** (Xu et al.) | arXiv 2606.09508 | 2026 | Head-level entropy-guided adaptive inference | Dense attention models; not sparse MoE. No hardware profiling. |
| **GeMoE** (Cai et al.) | arXiv 2606.26287 | 2026 | Gating entropy for uncertainty-aware adaptive routing in MoE LVLMs | Routing, not KV cache budget allocation. |
| **UAR** (ACL 2026) | ACL 2026 | 2026 | Uncertainty-aware routing for MoE training dynamics | Training-time, not inference-time KV budgeting. |

**The one true gap that is explicitly documented as open:**

> From InfoKV (2026): *"adaptive allocation improves performance for some models but can destabilize reasoning performance for others... More robust and architecture-aware allocation strategies remain an important direction for future work."*

That sentence is your paper's thesis.

---

## The Reframing: From Novel Method to Empirical Characterization

The paper should be built around **falsifiable research questions**, not a claimed novel method. A paper built around honest RQs is harder to scoop and more defensible because the contribution is the answer, not the idea.

---

## The 5 Concrete Innovations

### Innovation 1 — The First Controlled Per-Expert vs. Per-Layer Ablation in Sparse MoE

**Why nobody has done it:** The dense-model papers don't have experts. The MoE-specific papers (MoE-nD, PiKV) operate at layer granularity or focus on distributed serving, not accuracy-budget trade-offs per expert.

**What you do:**
- Run the same entropy budget formula at two granularities: (a) layer-aggregated (CAKE-style), (b) expert-level (EKVA).
- Same total budget. Same benchmarks. 3 model families (8 experts, 60 experts, 64 experts).
- Measure: PPL, downstream task accuracy, perplexity degradation curve across budget fractions (10%→80%).

**Why it outperforms:** This answers a question that PiKV, MoE-nD, and InfoKV all leave open. It's *directly citable* by any future paper in this space. Either a positive result (validates the expert-level hypothesis) or a negative result (reveals conditions under which it doesn't help) is equally valuable.

**Difficulty:** Low. You have the simulator. This is 2–3 days of work.

---

### Innovation 2 — The First Characterization of Entropy–Routing Frequency Decoupling in MoE

**Why this is novel:** Prior work assumes "high entropy = needs more KV budget" and stops there. No paper has asked: *is that assumption even valid in trained MoE models with load-balancing loss?*

Load-balancing auxiliary losses (used in Mixtral, Qwen, DeepSeek) force routing frequency to be approximately uniform. This means high-frequency experts might be low-entropy (they handle lots of easy/common tokens). That's the opposite of what the entropy-only budget formula assumes.

**What you do:**
- Compute per-expert: (a) attention entropy from your calibration, (b) routing frequency (tokens assigned / total tokens).
- Compute Pearson + Spearman correlation per model per layer.
- Test: is entropy anti-correlated with routing frequency in load-balanced MoE?
- If anti-correlated: you've *explained* why InfoKV's method destabilizes some models. That's a discovery, not just an experiment.

**Why it outperforms InfoKV specifically:** InfoKV observes destabilization but cannot explain it. Your result provides the mechanism. That's a higher-quality scientific contribution.

**Difficulty:** Medium. Uses your existing `output/*_phase1.pt` calibration data. New analysis script only.

**Expected result significance:** This is the kind of diagnostic result (a correlation coefficient across 3 models) that gets cited for years, independent of whether your method wins the benchmark.

---

### Innovation 3 — Cross-Domain Calibration Transfer Study

**Why nobody has done it properly:** Every entropy-KV paper calibrates on one distribution (usually WikiText or C4) and tests on the same distribution. Cross-domain generalization is either not tested or buried in an appendix afterthought.

**What you do:**
- Calibrate entropy budgets on: WikiText (general), code (StarCoder-like prompts), long-context QA.
- Evaluate on: LongBench tasks, RULER retrieval, Needle-in-Haystack, different domain than calibration.
- Report a cross-domain degradation matrix: calibration domain × evaluation domain → PPL change.

**Why it outperforms:** Most papers cannot make robustness claims because they never tested cross-domain. If your budgets transfer → you have a robustness claim no one else has made. If they don't transfer → you've documented a fundamental brittleness of entropy-based methods in MoE (which is equally publishable and directly answers InfoKV's open question).

**Difficulty:** Medium-High. Needs real model runs (Qwen on T4 is feasible). Budget: ~5–8 Colab T4 hours.

---

### Innovation 4 — Per-Expert Roofline Analysis (Hardware Characterization Nobody Has Done)

**Why this is the most defensible contribution:** Systems papers live and die by hardware measurements. The claim "different experts have different hardware regimes" sounds obvious but has never been measured per-expert with actual roofline profiling.

**What you do:**
- For Mixtral-8x7B on A100: use PyTorch Profiler + Nsight to extract per-expert FLOPs and HBM bytes during attention.
- Compute Arithmetic Intensity (AI = FLOPs/Bytes) per expert.
- Build a roofline scatter plot: x=AI, y=attained GFLOP/s, colored by entropy bucket (low/mid/high entropy).
- Test hypothesis: *do low-entropy experts cluster in the memory-bound region (high potential for KV reduction) while high-entropy experts cluster near the compute-bound ceiling?*

**Why this outperforms everything:**
1. MoE-nD did per-layer hardware analysis. Nobody did per-expert.
2. TriRoute explicitly notes "imperfect kernel support for mixed-precision cache and ragged expert batches" as a limitation — you would be providing the hardware evidence for *why* expert-level heterogeneity matters.
3. This is a measurable fact. It either shows up in the data or it doesn't. No reviewer can argue with a roofline plot.

**Difficulty:** High. Needs Colab A100 + Nsight. But `ekva/profiling/instrument.py` is already scaffolded.

---

### Innovation 5 — The Working Triton Kernel with Honest Speedup Numbers

**Why it outperforms all algorithm-only papers:** Algorithm papers are cheap. A working kernel with measured wall-clock speedup on real GPU is expensive and rare. TriRoute itself admits "realized wall-clock speedup is lower than the theoretical FLOP ratio" due to missing kernel support — you would be providing evidence *in the other direction*, showing what a well-targeted kernel can actually achieve for the memory-bound expert subset.

**What you do:**
- `ekva_triton_v1.py`: variable tile count per expert (budget-aware). Profile kernel time vs. baseline FA2.
- `ekva_triton_v2.py`: fused eviction index selection. Profile HBM reads, SM occupancy, kernel time per expert class.
- Report: "memory-bound experts → X× speedup. Compute-bound experts → no gain (expected from roofline theory)."
- The *match between prediction (roofline) and measurement (kernel)* is the story.

**Why "1.4× on memory-bound experts" is a publishable result:** It confirms the roofline theory. It demonstrates that expert-heterogeneity is real and exploitable. It's not inflated. It's more credible than papers claiming "2× speedup" without explaining when it applies.

**Difficulty:** Very High for v2; High for v1. Use kernel fallback if needed (report v1 only).

---

## 3 Novel Angles Not in Your Current Docs

These go beyond the current RQ1–RQ4 framing and could make the paper genuinely new.

### Novel Angle A — The Load-Balancing Loss Interference Hypothesis

**The idea:** MoE models are trained with auxiliary load-balancing losses (Mixtral: z-loss, Qwen: auxiliary, DeepSeek: sequence-level). These losses actively *fight* entropy-differentiated routing — they try to make routing uniform. This creates a fundamental tension: your KV budget formula tries to exploit expert heterogeneity, but the training objective tried to eliminate it.

**What to test:** Measure whether per-expert entropy variance *decreases* after training epochs with strong auxiliary loss (if you can access checkpoints) or whether entropy variance correlates with the strength of auxiliary loss weight across model families.

**Why it's novel:** Nobody has framed the entropy-based KV budgeting problem as adversarial to the training objective in this way. If you confirm this interference, it directly explains why entropy budgeting "works for some models but not others" and gives a principled criterion for *when to use it* (models with weaker auxiliary loss → more entropy variance → entropy budget more effective).

**Feasibility:** Medium. Doesn't require new training. Requires accessing the auxiliary loss weight from model configs (available in HF model cards for Mixtral, Qwen, DeepSeek).

---

### Novel Angle B — Expert Specialization Domain Consistency Metric

**The idea:** If expert specialization is real (and it is, per multiple papers), then high-entropy experts should be "generalist" (handle diverse token types) and low-entropy experts should be "specialist" (handle narrow token types). This should *correlate* with the optimal KV budget independently of entropy.

**What to test:** Compute per-expert token type distribution entropy (using POS tags or token frequency buckets) and see if it aligns with attention entropy. If these two measures of "specialization" agree → you have a much more robust multi-signal budget formula. If they disagree → you've found a case where attention entropy is a misleading signal.

**Why it's novel:** All current methods use only attention patterns to measure "importance." Using token-type distribution as an independent specialization signal and cross-validating against attention entropy has not been done.

**Feasibility:** Medium. Requires tokenizer + small analysis script. No GPU needed.

---

### Novel Angle C — Budget Instability Cascade: When Expert-Level Budgeting Fails

**The idea:** In deep MoE layers, experts process *outputs of other experts' attention*. If early layers have their KV budgets cut aggressively, the hidden state quality degrades, which changes routing distributions in later layers, which changes entropy patterns in those layers. This is a **cascade effect** that static calibration cannot capture.

**What to test:** Run calibration, then apply budgets, then re-calibrate on the compressed-attention outputs. Does entropy change? Do budgets shift? How many "recalibration iterations" until convergence?

**Why it's novel:** Static calibration assumes entropy measured under full KV conditions is valid under truncated KV conditions. This assumption has never been tested. If it's wrong, it explains a whole class of failures in entropy-based KV methods.

**Feasibility:** Medium-High. Requires actual model runs with truncation hook. Qwen-MoE on T4 is feasible.

---

## Research Questions (in priority order)

### RQ1 — Does expert-level granularity beat layer-level granularity at equal total budget?

**Priority:** Highest. Cheapest experiment (CPU simulator only). Tells you immediately whether the project's premise holds.

**Method:**
1. Take existing calibration output (`output/*_phase1.pt`)
2. Compute layer-aggregated entropy (average per layer across experts)
3. Derive budgets at layer level using the same proportional formula
4. Run both through the software simulator with the same total budget
5. Compare PPL under truncation for both approaches
6. Also compare multi-signal policies at both granularities

**Deliverable:** `experiments/rq1_granularity_comparison.py` + `output/rq1_granularity_comparison.pt` + figure

---

### RQ2 — How does attention entropy relate to routing frequency, and does that relationship break the "high entropy = high budget" assumption?

**Priority:** High. Uses existing calibration data. Produces a citable diagnostic finding.

**Method:**
1. Use existing calibration data across all 3 models (Qwen, Mixtral, DeepSeek)
2. Compute Pearson/Spearman correlation between per-expert entropy and per-expert routing frequency
3. Plot the relationship per model
4. Test whether the relationship differs across models with different expert counts (8 vs. 60 vs. 64)

**Deliverable:** `experiments/rq2_entropy_routing_correlation.py` + `output/rq2_correlation.pt` + multi-panel figure

---

### RQ3 — Does calibration transfer, or is this brittle?

**Priority:** Medium. Needs real model runs but is feasible on Qwen (T4).

**Method:**
1. Calibrate entropy on WikiText prompts, evaluate derived budget on LongBench/RULER tasks
2. Calibrate on code prompts, test on QA
3. If budgets hold up across domains → real robustness claim
4. If they don't → document exactly how much they degrade and why

**Deliverable:** Cross-domain degradation matrix + analysis script

---

### RQ4 — What does the hardware actually look like, and does the kernel realize the theoretical savings?

**Priority:** Medium (needs GPU). Most defensible systems contribution.

**Method:**
1. Run roofline instrumentation on Mixtral-8x7B (Weeks 8-9)
2. Build Triton kernel v1/v2 (Weeks 10-11)
3. Profile with Nsight Compute: HBM bandwidth, SM occupancy, kernel time — baseline vs. v2 per expert class
4. Try (a) static budget only, (b) budget + fused index selection

**Deliverable:** Roofline figure + per-expert FLOPs/bytes CSV + Triton kernel + Nsight report

---

## Experiment Priority Matrix

| Experiment | Compute Cost | Days | Novelty Level | Publishability |
|---|---|---|---|---|
| **RQ1: Layer vs Expert Granularity** | CPU only | 2–3 | ⭐⭐⭐ | ✅ Required |
| **RQ2: Entropy–Routing Correlation** | CPU only | 1–2 | ⭐⭐⭐⭐ | ✅ Stand-alone finding |
| **RQ3: Calibration Transfer** | T4 (5–8h) | 3–5 | ⭐⭐⭐ | ✅ Required |
| **RQ4a: Per-Expert Roofline** | A100 (4–6h) | 5–7 | ⭐⭐⭐⭐⭐ | ✅ Strongest systems claim |
| **RQ4b: Triton Kernel v1** | A100 (8–12h) | 7–10 | ⭐⭐⭐⭐ | ✅ With roofline |
| **Novel A: Aux Loss Interference** | CPU + analysis | 2–3 | ⭐⭐⭐⭐⭐ | 🔬 High-risk, high-reward |
| **Novel B: Token-Type Specialization** | CPU only | 2–3 | ⭐⭐⭐⭐ | 🔬 Medium risk |
| **Novel C: Budget Instability Cascade** | T4 (3–5h) | 3–4 | ⭐⭐⭐⭐⭐ | 🔬 High-risk, high-reward |

---

## What to Cut or De-prioritize

| Item | Recommendation |
|---|---|
| Full benchmark sweep (Weeks 5-6, all LongBench/RULER/InfiniteBench tasks) | Cut to 2 tasks only, once RQ1 tells you granularity actually matters. |
| 7 policies × 4 evictions = 16 combos in the paper | Report only 3-4 that tell the RQ story; put the rest in an appendix. |
| DeepSeek-MoE-16B as primary target | Stretch goal only. Do Qwen1.5-MoE (cheap, first) and Mixtral-8x7B (primary) properly. |
| Claiming "novel method" | Drop entirely. Frame as "reproducing and extending X for MoE-specific empirical characterization." |
| Triton kernel as the headline contribution | Keep it as RQ4 (systems validation), not the main story. |

---

## Paper Framing

### Recommended Title
> **"When Does Expert-Level KV Budgeting Matter? A Characterization of Entropy, Routing, and Hardware Regimes in Sparse MoE Inference"**

Alternative:
> **"Expert-Level KV Cache Allocation in Sparse MoE LLMs: An Empirical Study of Entropy, Routing Correlation, and Hardware Efficiency"**

### Abstract Template
```
Entropy-based KV cache budget allocation has been widely adopted for dense
transformers, but its behavior in sparse Mixture-of-Experts (MoE) models
remains uncharacterized. We present EKVA, an empirical study that answers
four open questions about expert-level KV budgeting in MoE inference.

First, we show that expert-level granularity [beats / fails to beat /
conditionally beats] layer-level granularity at equal total budget across
three MoE architectures with 8, 60, and 64 experts (RQ1).

Second, we discover that attention entropy and routing frequency are
[correlated / decoupled / anti-correlated] in load-balanced MoE models,
explaining why entropy-only allocation destabilizes reasoning on certain
architectures—a failure mode flagged but unexplained by prior work (RQ2).

Third, we show that entropy-derived budgets [transfer / degrade by X%]
across calibration domains, providing the first cross-domain robustness
characterization for MoE KV budgeting (RQ3).

Finally, we perform per-expert roofline analysis on Mixtral-8x7B,
demonstrating that individual experts occupy distinct compute/memory
regimes. We validate this hardware characterization with a Triton kernel
that achieves X× speedup on memory-bound experts and negligible overhead
on compute-bound experts (RQ4).
```

---

## Venue Targeting

| Venue | Bar | Fit |
|---|---|---|
| **ES-FoMo Workshop (co-located ICML/NeurIPS)** | Workshop: systems + empirical work, empirical is fine | ✅ Primary target |
| **ENLSP Workshop (NeurIPS)** | Efficient NLP + systems | ✅ Primary target |
| **MLSys** | Systems-first, benchmarks required | ✅ With kernel + roofline |
| **arXiv preprint** | Just post it | ✅ Do this regardless |
| **EMNLP Findings** | Empirical findings papers | ✅ Possible with RQ2 finding |

> **Do NOT target NeurIPS/ICML main track.** Workshop + arXiv is the right scope. Honest scope beats inflated claims every time.

---

## What to Do This Week

1. **Run RQ1 first, on your CPU simulator, no GPU needed.** Take your existing calibration output, compute layer-aggregated entropy budgets vs. expert-level entropy budgets at matched total budget, compare PPL under truncation. That's maybe 2–3 days of work with code you already have, and it tells you immediately whether the rest of this plan is worth building on top of, or whether you need to rethink further.

2. **Run RQ2 next (1–2 days, CPU only).** Compute entropy–routing correlation across all 3 models. This produces a citable quantitative finding with almost no compute cost.

3. **Read the PiKV paper and code properly, today, before touching Week 5 onward.** Not skim it — actually go through PERouter and BARouter's implementation. You need to know exactly what they do, because that becomes your related-work section either way.

4. **Wire the real KV truncation hook (Week 4).** Without this, all PPL numbers are placeholders. This is the gate for RQ1/RQ2 to become real benchmark results.

---

## The Key Insight

The mechanism (entropy → proportional budget) is not the innovation. The innovation is **answering questions nobody has answered about how that mechanism behaves in sparse MoE models with different architectures, different granularities, and different hardware regimes.** That's what makes this project worth doing, and that's what makes it defensible.

The load-balancing loss interference hypothesis (Novel Angle A) is the highest-leverage unexplored angle: if you can confirm that entropy variance is inversely related to auxiliary loss strength, you have a genuine discovery that explains a failure mode observed by 3+ independent papers and gives a principled criterion for when to use entropy-based KV budgeting.
