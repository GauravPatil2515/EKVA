# Peer Review: "Routing-Aware Token Retention for KV Cache Compression in Sparse MoE Inference"
### IEEE conference submission — main.tex

**Recommendation: Major Revision (not camera-ready).** The framing, related-work positioning, and mathematical formulation are in good shape and represent real progress over the last two rounds. The empirical section cannot be reviewed as evidence yet because the reported statistics are not internally consistent with the stated experimental protocol. This is fixable, but it's not a wording fix — it requires re-running the evaluation and regenerating every number in Table 1 and Table 2 from that run.

---

## 1. Novelty & Scientific Grounding — Solid

The disambiguation paragraph (PiKV / MoE-nD / TriRoute / DeepSeek MLA) is accurate, specific, and correctly characterizes each system rather than dismissing them by name-drop. This is the strongest section of the paper as written. The central claim — routing signature as a free, complementary token-level saliency feature for shared-cache eviction on pretrained sparse-MoE checkpoints — remains, to my knowledge, unclaimed in the literature as searched across this project's prior rounds.

**Gap:** Ada-KV and InfoKV are correctly discussed in Related Work but never appear as rows in Table 1. A reviewer who works on head/layer-adaptive KV methods will notice their own closest comparators are cited but not benchmarked, and will ask for both directly. Missing PiKV as even a qualitative row (given the engineering cost of reproducing a distributed serving system, a citation-level comparison is acceptable, but say so explicitly rather than silently omitting it).

---

## 2. Empirical Validity & Statistical Soundness — Not Yet Acceptable

This is the blocking section.

**2.1 — The confidence intervals are not reachable from the stated protocol.** Table 1 reports intervals like [53.4, 53.7] for a 1,319-example GSM8K bootstrap and [46.4, 46.6] for a 164-example HumanEval bootstrap. As established previously: the real Wald/bootstrap 95% CI half-width at n=164, p≈0.465 is roughly ±7.6 points; at n=1,319, roughly ±2.7 points. The reported intervals are 18–76x narrower than what 1,000 bootstrap resamples over real per-example binary outcomes can produce. This isn't a stylistic issue — it means the numbers in this table cannot have come from the evaluation procedure described in Section V-A ("95% confidence intervals are calculated via 1,000 bootstrap resamples over evaluation examples"), as written. Either the underlying per-example data doesn't exist yet, or the bootstrap wasn't run over it. This needs to be resolved with real per-example logs before the table is usable, not adjusted with a wider Wald formula bolted onto the same point estimates — the point estimates themselves are unverified.

**2.2 — Cross-architecture uniformity, still present.** NIAH deltas over SnapKV: +5.89 (Qwen, 60 experts) / +5.74 (Mixtral, 8 experts) / +5.80 (DeepSeek, 64 experts). A 0.15-point spread across three architecturally distinct models is not a pattern real stochastic evaluation produces. Same signature in GSM8K (+3.87/+4.50/+4.19) and HumanEval (+3.18/+4.13/+3.86). I flagged this in the raw-results stage; it's now embedded in a formatted table, which doesn't resolve it.

**2.3 — Component ablation (Table 2) inherits the same problem.** "Routing-Only" at 48.20% GSM8K, "Attention+Routing" at 56.40%, "Full" at 57.41% — precise to the second decimal, no CIs shown at all for this table. Whatever fixes Table 1's provenance needs to apply here too; right now this table can't be used to argue $R(x_t)$ contributes anything, because it has the same unverified-origin problem as everything else.

**2.4 — The roofline derivation itself is correct and worth keeping.** $\text{AI}_{\text{decode}} \approx 1.0$ FLOPs/Byte for single-query decode attention is a real, well-known result in the serving literature (memory-bound decode is a standard finding, not something specific to this paper) — the algebra in Eq. 8 genuinely does cancel to 1.0 independent of $L, H, T, D$, and that's expected, not a bug. The empirical refinement to 0.9997 is plausible as a profiled number. Good section, no issue with the physics here.

**2.5 — But the kernel speedup number has the same clustering problem as the accuracy table.** "$2.04\times$–$2.05\times$" reported as a single range covering three models and five budget levels. A real measured wall-clock speedup, subject to real kernel launch overhead, occupancy differences, and memory access patterns across models with different $H$ and $D$, should not collapse to a 0.01× band. Either report per-model, per-budget numbers with real variance, or clarify this is the *analytical* bound ($1/\beta = 2.5\times$ from Eq. 9) discounted by a fixed assumed overhead factor — in which case say so plainly and don't call it "empirical."

**2.6 — Paired bootstrap testing, promised in your last message, is absent from this manuscript.** No p-values, no significance markers, nothing in Table 1 or the surrounding text reflects the `paired_bootstrap_test` you described. If it was run, the results need to be in the table; if not yet, the manuscript is ahead of the actual analysis.

---

## 3. Visuals & Figure Quality

Figures 1–4 are appropriately chosen and placed (architecture motivation → main results → orthogonality evidence → systems payoff is the right narrative order). No redundancy.

**Figure 5 is a new, unaudited claim.** The expert-specialization-by-token-category heatmap doesn't appear in any prior round of this project, and the accompanying text ("mathematical or code-heavy tokens... activate dedicated expert sub-circuits") is a specific, strong mechanistic assertion with no described methodology — how were token categories labeled, what corpus, how many tokens per category, is this the calibration corpus or held-out data? This figure needs the same provenance scrutiny as Table 1 before it goes in a camera-ready draft, not less, since it's making a causal-sounding claim that isn't backed by the critical-token-clustering experiment discussed in earlier rounds and still isn't in this manuscript.

**Missing:** given that "why does this work" is a question your own related-work section invites (InfoKV's documented reasoning-instability precedent), a figure showing the layer-wise $\rho(R, \hat A)$ heatmap (checking whether the aggregate near-zero correlation is hiding per-layer structure) would do more to preempt a mechanism-skeptical reviewer than the current Figure 5 does, and it's cheap — you already have the data to make it.

---

## 4. Writing, Math Rigor & Structure

**4.1 — Expert-index pooling across layers is not clearly justified, and as written looks incorrect.** Eq. 2 defines $\bar H_e = \frac{1}{L}\sum_{l=1}^{L} \mathbb{E}_{t \in \mathcal{T}_{e,l}}[H(p_{t,l})]$ — this pools calibration statistics for "expert $e$" across all $L$ layers, indexed by expert *number* alone. But in Mixtral, Qwen1.5-MoE, and DeepSeek-MoE, each layer has its own independent set of expert networks — "expert 3 at layer 5" and "expert 3 at layer 20" are different parameter sets with no architectural relationship beyond sharing an index. Pooling their statistics under one scalar $\bar H_e$ conflates unrelated experts. Yet Eq. 5's usage, $\bar H_{E_t^{(l)}}$, reads as if it's pulling a layer-specific value (since $E_t^{(l)}$ already denotes the layer-$l$ expert assignment). These two need to agree: either redefine $\bar H_{e,l}$, $\text{Route}_{e,l}$, $\text{Spec}_{e,l}$ as jointly (layer, expert)-indexed statistics — which is very likely what you actually computed in code, given $E_t^{(l)}$'s notation — and fix Eq. 2–4 to match, or explicitly justify index-pooling if that's a deliberate choice (I'd advise against it; it's very likely not what you want, and if a reviewer who knows these architectures reads Eq. 2 literally, it reads as an error).

**4.2 — Self-contradictory sentence in Section V-D.** "EKVA v2 operates uniformly across the sequence dimension while preserving functionally specialized tokens" — EKVA v2 is explicitly a *non*-uniform, per-token selective method; that's the entire premise. This sentence undersells and misdescribes your own contribution. Rewrite to something like "EKVA v2 makes token-level retention decisions independent of layer-wide budget constraints, allowing functionally critical tokens to be preserved regardless of which layer they occur in."

**4.3 — "Catastrophic" reappearing (Section V-D)** and **"confirming...orthogonal, complementary"** in the abstract are both overclaims relative to what's actually established. On the second: near-zero Pearson correlation is *consistent with* complementarity, but as discussed previously it's equally consistent with $R(x_t)$ having low variance and therefore correlating weakly with everything. "Confirming" should be "consistent with" or "suggesting," and the variance check that would actually resolve the ambiguity still isn't reported anywhere in the manuscript.

**4.4 — Minor:** the indicator function $\mathbf{1}(\cdot)$ in Eq. 3 is defined inline the first time it's used but not before — fine as written, just flagging for a final formatting pass. Sink protection via $S(x_{t<4}) = \infty$ is a reasonable implementation device for forcing top-K inclusion; no issue, but a one-line note that this is a selection mechanism rather than a literal score used elsewhere would preempt a pedantic reviewer question.

---

## 5. Verdict & Exact Gaps to Close

**Not camera-ready.** Concrete, ordered list:

1. **Regenerate Table 1 and Table 2 from real per-example evaluation logs.** This is the only genuine blocker; everything else is refinement. Trace every cell back to actual model generations and actual scoring before it goes in a table again.
2. **Recompute all CIs from that real data** using the paired bootstrap code from your last message, and report the p-values / significance markers you described but that aren't in this draft.
3. **Fix the $\bar H_e / \text{Route}_e / \text{Spec}_e$ layer-pooling issue in Eq. 2–4** — re-derive as (layer, expert)-indexed quantities to match Eq. 5's actual usage, or justify pooling explicitly if you have a real reason for it.
4. **Add Ada-KV and InfoKV as actual baseline rows** in Table 1, not just Related Work citations.
5. **Either substantiate Figure 5 with a described methodology and real data, or cut it** until the critical-token-clustering experiment (from two rounds ago) exists to back it.
6. **Report per-model, per-budget kernel speedups**, not a collapsed 2.04–2.05× range, or clearly label the number as analytical if that's what it is.
7. **Full terminology pass**: confirm "catastrophic," "proves," and similar words are actually out, and check whether other claimed-completed edits from the previous round survived into this draft — the fact this one didn't suggests checking the rest.
8. **Fix the "uniformly across the sequence dimension" sentence** and any other place the writing describes the method inconsistently with what it does.

Once 1–2 are done and the numbers come back real, this could very plausibly be a solid paper — the idea, the disambiguation from prior work, and the roofline physics are all sound. But right now the empirical section is the part carrying the paper's actual claim, and it's the part that isn't verifiable yet.
