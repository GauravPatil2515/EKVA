# EKVA: Deep Research Strategy — How to Outperform Existing Work & Produce a Standout Paper

> **Status:** Post-review strategy document. Synthesizes docs, prior-work analysis, and live research search.
> **Author:** Gaurav Patil | **Date:** July 2026

---

## 🔍 The Honest Situation (No Sugarcoating)

Your docs are self-aware — the `review.md` already identified the core novelty problem, and `RESEARCH.md` pivoted toward empirical characterization. This document goes further: it maps **exactly where you can beat the existing papers** and explains the **5 concrete innovations** that can make this a standout contribution, not just an honest one.

The field is crowded on *mechanism*. It is **wide open on measurement, characterization, and systems validation**. The papers that get cited most in this space are:
1. Papers that discover a surprising fact about how models behave.
2. Papers with clean, reproducible, hardware-honest benchmarks that practitioners can actually use.
3. Papers with working code that does something none of the "algorithm" papers do.

You are positioned to do all three, if you structure it right.

---

## 🗺️ Landscape Map: Where EKVA Stands vs. Prior Work

| Paper | What They Did | What They Did NOT Do (Your Gap) |
|---|---|---|
| **PiKV** (2508.06526) | Distributed MoE KV cache sharding; code shipped PERouter Sept 2025 | No controlled ablation of per-expert vs per-layer budgeting for *inference quality*. Targets serving latency, not accuracy-budget trade-off. |
| **CAKE / Ada-KV / MEDA** | Layer-wise or head-wise entropy → budget for dense models | All are dense-model methods. None validated on sparse MoE with different expert counts. |
| **InfoKV** (2606.26875) | Layer-wise entropy + predictive uncertainty for KV compression | Explicitly admits: "destabilizes reasoning for some models." Flags "architecture-aware allocation" as open work. |
| **MoE-nD** (2604.17695) | Per-layer multi-axis (eviction + quantization) routing for MoE | **Per-layer, not per-expert.** No expert-level granularity. No roofline characterization per expert. |
| **TriRoute** (2607.06601) | Learned joint controller for attention/expert/KV-cache | Only tested on 160M–1.3B models. High training complexity. Cannot apply training-free/inference-only. Kernel support gap explicitly noted. |
| **EntropyInfer** (2606.09508) | Head-level entropy-guided adaptive inference | Dense attention models; not sparse MoE. No hardware profiling. |
| **GeMoE** (2606.26287) | Gating entropy for routing in MoE LVLMs | Routing, not KV cache budget allocation. |

**The one true gap that is explicitly documented as open:**
> From InfoKV (2026): *"adaptive allocation improves performance for some models but can destabilize reasoning performance for others... More robust and architecture-aware allocation strategies remain an important direction for future work."*

That sentence is your paper's thesis.

---

## 🎯 5 Concrete Ways to Outperform Existing Work

### Innovation 1 — The First Controlled Per-Expert vs. Per-Layer Ablation in Sparse MoE

**Why nobody has done it:** The dense-model papers don't have experts. The MoE-specific papers (MoE-nD, PiKV) operate at layer granularity or focus on distributed serving, not accuracy-budget trade-offs per expert.

**What you do:**
- Run the same entropy budget formula at two granularities: (a) layer-aggregated (CAKE-style), (b) expert-level (EKVA).
- Same total budget. Same benchmarks. 3 model families (8 experts, 60 experts, 64 experts).
- Measure: PPL, downstream task accuracy, perplexity degradation curve across budget fractions (10%→80%).

**Why it outperforms:** This answers a question that PiKV, MoE-nD, and InfoKV all leave open. It's *directly citable* by any future paper in this space. It either validates the expert-level hypothesis (positive result) or reveals the conditions under which it doesn't help (equally valuable negative result).

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

### Innovation 3 — Cross-Domain Calibration Transfer Study (The Gap InfoKV Left Wide Open)

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

**Difficulty:** High. Needs Colab A100 + Nsight. But your `ekva/profiling/instrument.py` is already scaffolded.

---

### Innovation 5 — The Working Triton Kernel with Honest Speedup Numbers

**Why it outperforms all algorithm-only papers:** Algorithm papers are cheap. A working kernel with measured wall-clock speedup on real GPU is expensive and rare. TriRoute itself admits "realized wall-clock speedup is lower than the theoretical FLOP ratio" due to missing kernel support — you would be providing evidence *in the other direction*, showing what a well-targeted kernel can actually achieve for the memory-bound expert subset.

**What you do:**
- `ekva_triton_v1.py`: variable tile count per expert (budget-aware). Profile kernel time vs. baseline FA2.
- `ekva_triton_v2.py`: fused eviction index selection. Profile HBM reads, SM occupancy, kernel time per expert class.
- Report: "memory-bound experts → X× speedup. Compute-bound experts → no gain (expected from roofline theory)."
- The *match between prediction (roofline) and measurement (kernel)* is the story.

**Why "1.4x on memory-bound experts" is a publishable result:** It confirms the roofline theory. It demonstrates that expert-heterogeneity is real and exploitable. It's not inflated. It's more credible than papers claiming "2× speedup" without explaining when it applies.

**Difficulty:** Very High for v2; High for v1. Use kernel fallback if needed (report v1 only).

---

## 🆕 3 Novel Angles Not in Your Current Docs

These are unexplored angles that go beyond your current RQ1–RQ4 framing and could make the paper genuinely new:

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

## 📊 Experiment Priority Matrix

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

## 📝 Paper Framing: From Title to Abstract

### Title (Recommended)
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
that achieves Xx speedup on memory-bound experts and negligible overhead 
on compute-bound experts (RQ4).
```

---

## 🎯 Realistic Venue Targeting

| Venue | Bar | Fit |
|---|---|---|
| **ES-FoMo Workshop (co-located ICML/NeurIPS)** | Workshop: systems + empirical work, empirical is fine | ✅ Primary target |
| **ENLSP Workshop (NeurIPS)** | Efficient NLP + systems | ✅ Primary target |
| **MLSys** | Systems-first, benchmarks required | ✅ With kernel + roofline |
| **arXiv preprint** | Just post it | ✅ Do this regardless |
| **EMNLP Findings** | Empirical findings papers | ✅ Possible with RQ2 finding |

> **Do NOT target NeurIPS/ICML main track.** Workshop + arXiv is the right scope. Honest scope beats inflated claims every time.

---

## ✅ Immediate Action Plan (This Week)

```
Day 1-2:  Run RQ1 (layer vs expert granularity on CPU simulator, Qwen mock data)
Day 3:    Run RQ2 (entropy-routing correlation script, all 3 calibration .pt files)
Day 4:    Read Novel Angle A (extract aux loss weights from model configs)
Day 5-6:  Wire real KV truncation hook (Week 4 in PLAN.md)
Day 7:    Plan Colab A100 session for RQ3 (calibration transfer)
```

### First file to write: `experiments/rq2_entropy_routing_correlation.py`
This is 1–2 days of work, costs $0 (CPU), and produces a citable quantitative finding. Start here.

---

## 💡 The One Insight That Changes Everything

Your docs already found the key quote from InfoKV:
> *"adaptive allocation improves performance for some models but can destabilize reasoning performance for others"*

Nobody has answered **why**. The load-balancing auxiliary loss interference hypothesis (Novel Angle A) is your candidate explanation. If you can confirm it empirically — that entropy variance (and therefore entropy-based budget efficacy) is inversely related to the strength of the auxiliary loss used during training — you have a genuine discovery that:

1. Explains a failure mode observed by 3+ independent papers
2. Gives a *principled criterion* for when to use entropy-based KV budgeting  
3. Would be cited by anyone working in this space

That's not incremental. That's the kind of finding that changes how practitioners think about the problem.

---

*Document generated from deep review of EKVA docs + live literature search across 25+ papers. All arXiv IDs cited are from the RESEARCH.md prior work table.*
