# EKVA Research Advisory
### Expert-Aware KV Budget Allocation for Sparse MoE Inference — Literature Audit, RQ Triage, and Paper Blueprint

*Compiled August 2026. Based on direct review of primary sources (arXiv, OpenReview, GitHub), not abstracts alone.*

---

## TL;DR

Your gap is real. Nobody has published expert-granularity KV budget allocation on real pretrained sparse-MoE checkpoints (Qwen1.5-MoE, Mixtral, DeepSeek-MoE), calibration-based, training-free. But it's a narrower gap than your doc suggests, one of your four cited "related works" isn't actually related, your closest real competitor (PiKV) is a moving open-source target you need to track, and RQ4 (the Triton kernel + roofline work) is your biggest scope-creep risk given an RTX 3050 and rationed A100 hours. RQ1 is your paper. Everything else should support it, not compete with it for your time.

---

## Step 1: Literature Landscape (2024–2026)

The KV cache compression field has split cleanly along **granularity axes**. Nobody's serious work spans two axes well — that's actually the throughline of the whole field right now, and it's your framing hook.

### Token/sequence axis (the oldest, most saturated)
**StreamingLLM, H2O, SnapKV, PyramidKV, Scissorhands.** All decide *which tokens* to keep, uniformly across heads/layers/experts. This is solved-enough territory; don't spend cycles here beyond using them as eviction-strategy baselines inside your simulator (which you're already doing — good).

### Head axis
**Ada-KV (NeurIPS 2025, Feng et al.)** — the first head-wise adaptive budget allocator. Diffuse-attention heads get more slots, concentrated-attention heads get fewer, derived from a theoretical L1 eviction-loss bound. Critically, the authors write it themselves: the method is "limited to within single layer," and extending across layers is explicit future work. Nobody's claimed the expert axis in that same paper or its lineage.

### Layer axis
**CAKE (2025)** — frames it as a "cake-slicing problem," allocates budget per layer using spatial (entropy) and temporal (variance) attention signals, with a cascading allocation mechanism as layers are processed.
**PyramidKV** — fixed pyramid shape (more budget to early layers, less to late layers), input-independent — CAKE and later LAVa both call this out as a weakness.
**LAVa (2025)** — hyperparameter-free, dynamic per-layer *and* per-head budget jointly, using an uncertainty-based score. Probably the most technically elegant of the layer/head-adaptive papers. Worth reading closely as a methodological template even though it's dense-model only.

### Multi-axis, but not what the name suggests
**MoE-nD (arXiv 2604.17695, April 2026)** — despite the name, this is **not** about sparse MoE expert routing. It uses "MoE" as an architectural metaphor: a router that picks, *per layer*, a tuple of (eviction ratio, K-bits, V-bits) via an offline-calibrated greedy solver. Tested on DeepSeek-R1-Distill-Qwen-7B — a dense model. Its own related-work section explicitly places itself against AdaKV (head) and PyramidKV (layer) as "no prior work routes [multiple axes jointly]" — it never mentions sparse-MoE expert budgets because that's not its problem. **You need one clear sentence in your related work distinguishing this**, because a reviewer skimming titles will assume overlap that doesn't exist. Don't let a name collision cost you a review round.

### Coupled, trained-from-scratch architectures
**TriRoute (arXiv 2607.06601, July 2026)** — genuinely couples attention mode, FFN expert choice, and KV bit-width, but as a jointly-trained controller on 160M–1.3B models trained from scratch, evaluated for Pareto-dominance over independently-tuned MoD/MoE/KV-quant. This is architecture research, not an inference-time calibration method for existing pretrained checkpoints. Different problem category from EKVA. Worth one line in related work; not a threat to your novelty.

### Entropy-based compression, dense models, with a cautionary result
**InfoKV (arXiv 2606.26875, June 2026)** — entropy-aware compression combining predictive uncertainty and layer-wise representation evolution. They ran the layer-adaptive-budget experiment you're implicitly betting on, and **rejected it**: excessively imbalanced layer budgets destabilized long-range reasoning, so they shipped uniform budgets as the default. This is a dense-model, layer-axis result. It may not transfer to expert-axis sparse-MoE — different mechanism, different failure surface — but you should treat it as a documented risk and test for it explicitly, not discover it in review.

### The actual MoE-aware systems competitor
**PiKV (arXiv 2508.06526, ICML 2025 ES-FoMo III, Liu et al.)** — a parallel/distributed KV serving system built specifically for sparse MoE. I pulled the full paper text, not just the abstract. Its three components:
- **PiKV Routing** — decides which *FFN* experts activate per token (standard MoE routing with various strategies, including an entropy-penalized load-balance variant). This is about which experts a token goes to, not how much KV budget an expert gets.
- **PiKV Compression** — applies existing schemes (LoRA, PyramidKV, SVD, distillation) to KV entries. No per-expert budget differentiation based on cross-expert entropy comparison is in the published methodology.
- **PiKV Scheduling** — per-page eviction scoring using existing formulas (H2O, AdaKV-style, DuoAttention), applied within an expert-sharded storage layout.

So the *published* PiKV is a serving-systems paper: expert-sharded storage + generic eviction/compression, not a principled "budget ∝ f(entropy, routing frequency, specialization)" allocator with theoretical grounding. **But it's a living open-source repo**, and as of September 2025 the GitHub added a Budget-Aware Router (BARouter) and Predictive-Entropy Router (PERouter) beyond what's in the paper. Check the current repo state before you submit — this is the one project that could close your gap out from under you if it keeps evolving in that direction. Cite the paper, but also cite/screenshot the repo state at time of submission, and if BARouter turns out to do something close to your core mechanism, you need a differentiation paragraph, not silence.

### What this means for your novelty claim
As of August 2026: **expert-granularity, calibration-based, training-free KV budget allocation on real pretrained sparse-MoE checkpoints is unclaimed.** Head axis (Ada-KV) explicitly stops at layer boundaries. Layer axis (CAKE, PyramidKV, LAVa) doesn't touch experts. The two "MoE" papers in your list either aren't about expert routing (MoE-nD) or aren't calibration-based on pretrained checkpoints (TriRoute). PiKV is MoE-native but solves a different problem (distributed serving, not principled budget theory) — for now.

This is a real, defensible, but *procedural* gap — nobody's run this specific experiment in this specific configuration — not a deep theoretical one. That's fine. It just means your paper's value has to come from rigor of execution and honest characterization (à la MoE-nD's own framing: "two null results, cleanly characterizing when this helps"), not from a conceptual breakthrough. Aim for that tone, not a bigger-than-it-is claim.

---

## Step 2: RQ Triage

### RQ1 (Granularity Ablation) — **KEEP. This is your paper.**
Per-expert vs. layer-aggregated budgeting at matched total budgets, across 8/60/64-expert architectures. This is the cleanest, most novel, most reviewer-legible claim you have. It's the direct analogue of "Ada-KV did heads, CAKE did layers, we do experts" — a horse race at an unclaimed granularity. Everything else in the paper should be built to support this result, not to compete with it for space or compute.

### RQ2 (Entropy vs. Routing Decoupling) — **DOWNGRADE from co-equal RQ to mechanistic sub-analysis.**
The hypothesis — load-balancing auxiliary loss creates anti-correlation between routing frequency and attention entropy, explaining why pure-entropy methods destabilize — is genuinely untested in the literature I found. Nobody's directly measured this at the expert level. That's interesting. But as a standalone RQ it's weak for three reasons: (1) it's explanatory, not a headline finding a reader can act on; (2) testing it properly wants access to training dynamics or a good inference-time proxy, which is more work than your compute budget probably wants to spend on a secondary point; (3) even if confirmed, it doesn't change what practitioners should *do* — RQ1's result does. Reframe it as Section 5 ("why entropy-only allocation fails") supporting the multi-signal design choice in RQ1, not Figure 1 material. This also de-risks the paper: if RQ2's correlation doesn't hold cleanly, you haven't lost a pillar, you've lost a paragraph.

### RQ3 (Calibration Transferability) — **KEEP, but connect it explicitly to the InfoKV risk.**
Cheap, standard due diligence — every serious KV paper in this space runs a generalization check (Ada-KV: question-aware vs question-agnostic; MoE-nD: 4-task LongBench subset plus AIME plus MATH-500 null results). Your version (text → code/math/long-QA) is the right shape. The one change I'd make: explicitly include a reasoning-heavy benchmark and frame part of RQ3 as "does expert-axis adaptive budgeting hit the same reasoning-destabilization failure mode InfoKV found at the layer axis?" That turns a routine generalization check into a finding a reviewer will actually remember.

### RQ4 (Roofline & Systems Validation) — **DOWNGRADE and DESCOPE. This is your biggest risk.**
Two separate problems here:

1. **Compute reality.** A custom Triton kernel (two variants!) plus Nsight Compute roofline modeling is a serious systems-engineering project on its own, and you're proposing to do it on an RTX 3050 with occasional A100 access. Kernel work has a nasty property: it eats weeks and either works or doesn't, with little partial credit. This is exactly the kind of scope creep that's bitten your other projects (per your Lethos-AI and CoT verification audits) — an ambitious secondary system built without protecting the core empirical claim first.

2. **Profiling reliability.** Consumer GPUs (RTX 3050) often lack the hardware performance counters datacenter GPUs (A100/H100) expose to Nsight Compute, so roofline numbers built primarily from consumer-GPU profiling invite reviewer skepticism if you don't say so upfront.

**Recommendation:** ship the *analytical* roofline model (FLOPs/bytes theoretical, which is cheap and correct regardless of hardware) as a motivating figure, keep only Triton kernel v1 (variable KV tile count per expert — the simpler one), report wall-clock decode latency on whatever hardware you actually have, and put v2 (fused eviction indexing) plus full Nsight-validated roofline in an explicit "future work" line. State the hardware limitation as a one-line limitation, not something to route around — reviewers reward that honesty far more than they reward a shaky claim.

---

## Step 3: Compute-Constrained Experimental Design

Given RTX 3050 local + rationed Colab A100, here's the sequencing that protects the paper's backbone first and treats everything else as marginal spend:

**Cheap and load-bearing (do first, do thoroughly):**
- Full RQ1 granularity ablation grid (uniform / layer-aggregated CAKE-style baseline you implement / ekva-multi-signal) on **Qwen1.5-MoE-A2.7B only** — it's the model that runs locally, so this is where you can afford to iterate. 3–4 budget levels × your existing benchmark set.
- Multi-signal ablation table (entropy-only / routing-freq-only / specialization-only / combined) — free, reuses calibration stats you've already computed for RQ1. This does double duty as your RQ2 support material.

**Moderate cost (do once, don't grid-search):**
- Reproduce a *reduced* version of the RQ1 sweep (fewer budget points, one benchmark, not the full grid) on Mixtral-8x7B and DeepSeek-MoE-16B via Colab A100 — this is what buys you the "8/60/64 experts" generalization claim, and it doesn't need to be exhaustive to be convincing. One clean point per model beats three noisy ones.
- RQ3 transferability, including the reasoning-benchmark check against the InfoKV failure mode.

**High cost, do last, descope aggressively:**
- Triton kernel v1 wall-clock benchmark, analytical roofline, honest limitations paragraph on consumer-GPU profiling. v2 and full Nsight validation → future work.

**Optional, only if RQ1 lands cleanly and time remains:**
- Online re-calibration cascade — periodically refreshing entropy/routing stats during long-context decoding rather than a single static calibration pass. This is genuinely the one add-on that would elevate the paper past "granularity ablation" into "here's a new mechanism," and it's cheap: no kernel work, just re-running your existing calibration hook on a schedule. If you have bandwidth for exactly one "breakthrough" experiment, this is the one — not the kernel.

---

## Step 4: Paper Blueprint

### Title (recommended, plus two alternates)
**Recommended:** *"Expert, Not Layer: Where Sparse-MoE KV Cache Budgets Should Actually Go"*
Alternates: *"EKVA: Expert-Granularity KV Cache Budgeting for Sparse Mixture-of-Experts Inference"* (safer, more literal — better if you want a searchable/citable acronym) · *"Beyond Layers and Heads: Expert-Aware KV Budget Allocation in Sparse MoE LLMs"*

### Abstract template
> KV cache budget allocation has been studied at the head level (Ada-KV) and the layer level (CAKE, PyramidKV, LAVa), but the expert axis of sparse Mixture-of-Experts models — where attention intensity, routing frequency, and specialization vary sharply across experts — remains unaddressed. We introduce EKVA, a calibration-based, training-free method that allocates per-expert KV budgets from a multi-signal score combining attention entropy, routing frequency, and specialization, under integer-exact memory and starvation constraints. Across three sparse-MoE architectures spanning 8, 60, and 64 experts (Mixtral-8x7B, Qwen1.5-MoE-A2.7B, DeepSeek-MoE-16B), we show expert-granularity allocation [outperforms / matches-with-lower-variance — fill after RQ1 results] layer-aggregated allocation at matched total budgets. We characterize when entropy-only allocation fails and why a multi-signal formulation is necessary, verify calibration transfers across text, code, and reasoning-heavy workloads, and demonstrate a Triton kernel realizing wall-clock decode speedups on memory-bound experts identified via roofline analysis.

*(Note the bracketed hedge — don't lock the abstract's central claim until RQ1 data is in hand. Write the rest now, fill that clause last.)*

### Figures (in narrative order, not RQ order)
- **Figure 1 — Motivation.** Heatmap of per-expert attention entropy and routing frequency (layer × expert) for one model, showing the non-uniformity your whole premise rests on. This is the "look how different experts actually are" figure — it should be the first thing a reader sees.
- **Figure 2 — Main result (RQ1).** Accuracy/task-score vs. total KV memory, three lines (uniform / layer-aggregated / EKVA), multi-panel across the three architectures. This is the figure the paper lives or dies on.
- **Figure 3 — Multi-signal ablation.** Component contribution table/bar chart: entropy-only, frequency-only, specialization-only, combined. Supports the RQ2 mechanism without needing RQ2 as a standalone claim.
- **Figure 4 — Transferability matrix (RQ3).** Calibrate-on-text, evaluate-on-{text, code, math, long-QA}, degradation heatmap, with the reasoning-failure-mode check called out explicitly.
- **Figure 5 — Systems payoff (RQ4, descoped).** Analytical roofline placing experts into memory-bound vs. compute-bound regimes, plus measured wall-clock speedup from the Triton v1 kernel on the memory-bound cluster.

### Prioritized checklist
1. **RQ1 core ablation on Qwen1.5-MoE** (uniform vs. your CAKE-reimplementation vs. ekva-multi-signal, 3–4 budgets, your existing benchmark set). This is the backbone — if it doesn't show a clear result, nothing downstream matters, so front-load it before touching kernels.
2. **Multi-signal ablation table** — free, same calibration run as step 1.
3. **Reduced RQ1 sweep on Mixtral and DeepSeek-MoE** (one clean config each, Colab A100) — buys the "3 architectures, 8/60/64 experts" generalization claim.
4. **RQ3 transferability**, including the explicit InfoKV-style reasoning-stability check.
5. **Triton v1 wall-clock benchmark + analytical roofline**, with an honest limitations paragraph on consumer-GPU profiling. Stop here unless everything above is already solid.
6. **(Stretch, only if time remains)** online re-calibration cascade experiment.
7. **Write-up and venue.** Target ES-FoMo or ENLSP workshop first — lower bar, faster turnaround, good fit for a result-and-characterization paper at your compute scale. Treat EMNLP Findings as a stretch target only if RQ1 and RQ3 both land cleanly; don't write the paper assuming Findings-tier scope from the start, or RQ4 scope creep will find its way back in.

---

## One more thing to flag before you write a line of the paper

Go re-check the PiKV GitHub repo's current state — not the ICML paper PDF — for what BARouter and PERouter actually compute. If either one turns out to allocate KV budget per expert as a function of entropy, your differentiation paragraph needs to be airtight and specific (what signal, what granularity, what's actually novel in your formula), not a generic "we address a different aspect" wave-off. That's the one open question in this whole assessment I couldn't close from the published paper text alone, and it's the one that matters most.
