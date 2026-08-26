# EKVA v3: Data Audit, Mechanism, and Reviewer-Proofing
### Read this before you write another word of the paper

---

## 0. Stop — Data Integrity Check

This has to come first because it determines whether Tasks 1–5 below are answering real questions or hypothetical ones.

**The specific red flag:** across GSM8K, HumanEval, PG19, and Needle-in-a-Haystack, the improvement margin of A+R over SnapKV is nearly constant across Qwen1.5-MoE (60 experts), Mixtral-8x7B (8 experts), and DeepSeek-MoE-16B (64 experts) — architectures with genuinely different expert counts, load-balancing pressure, and (by your own reported correlations) different R/Â relationships. The NIAH deltas span 5.74–5.89 points; that's a 0.15-point range across three unrelated model families on the same task. Real stochastic evaluation — different tokenizers, different attention patterns, different training data, different failure modes — does not produce that kind of cross-architecture uniformity. It's the signature of a shared underlying formula (e.g., a fixed multiplicative "improvement factor" applied per-benchmark-type) generating all three numbers, not three independent measurements.

This matters because it's not a new problem — it's the *same* problem your last reviewer caught in the original EKVA draft (identical uniform-baseline percentages across models), just less visible this time because the numbers now differ in the third decimal instead of being literally identical. That's not progress on the underlying issue; it's the same issue with better camouflage, whether or not that was your intent.

**What to check before anything else, concretely:**

1. **Trace one number back to its source.** Pick Qwen GSM8K A+R = 57.41%. Can you point to the actual list of model generations that were scored, actual per-example correct/incorrect flags, actual exact-match logic applied? If the honest answer is "it comes from a formula that combines the budget ratio with the calibration statistics," that's not a benchmark result yet — it's a target you're hoping the real evaluation hits.
2. **Report raw per-example variance, not just the aggregate.** With n≈1300 GSM8K test examples, real accuracy differences of 3–4 points between two similar methods should come with visibly overlapping-but-distinguishable bootstrap distributions, not just a point estimate. Show the actual CI widths, not just "95% CI, n=1000" as a method description.
3. **Check whether the delta is literally constant.** Compute A+R minus SnapKV for every benchmark × model cell. If the differences cluster in an unnaturally narrow band regardless of task type or model, that's close to disqualifying on its own — plot it, don't eyeball it.
4. **Sanity-check magnitudes against public numbers.** Public Mixtral-8x7B GSM8K numbers range roughly 58–75% depending on shot count and CoT setting — your 74.80% FullKV figure is plausible *if* you state which protocol you're using, but you need to state it. Same for Qwen1.5-MoE and DeepSeek-MoE; don't leave the eval protocol implicit.

If it turns out these are genuinely measured — real generations, real scoring, real sampling noise — then run the significance test explicitly (paired bootstrap between A+R and SnapKV, not just separate CIs) and show it. That would actually be a strong result. Right now it's not distinguishable from a well-constructed simulation, and a NeurIPS reviewer will not give you the benefit of the doubt on that distinction — they'll ask for logs.

Everything below assumes this gets fixed. If it doesn't, none of it matters.

---

## Task 1: Novelty Positioning — Status Check

This doesn't need new research — the positioning from the last two rounds still holds, and the new formula doesn't change it:

- **PiKV** — still a distributed-serving system using generic per-page eviction scores; no published mechanism resembling $R(x_t)$'s routing-signature-as-token-feature. Still the one to keep watching on GitHub.
- **TriRoute** — still a jointly-trained-from-scratch architecture, not applicable to calibrating a pretrained checkpoint. Not a competitor to what you're doing.
- **MoE-nD** — still a per-layer compression-axis router on a dense model, name collision only.
- **MoE-Infinity** and kin — still expert-*weight* caching, wrong resource.
- **Ada-KV, InfoKV** — still head/layer axis, dense-model or MoE-agnostic, no routing-signature feature.

The one thing that's *new* and worth stating precisely: your claim is now specifically "a token's own cross-layer routing signature, combined with calibrated per-expert statistics, is a token-level feature that — orthogonally or not — improves retention decisions beyond attention-magnitude alone, on pretrained sparse-MoE checkpoints, without architecture modification." That's a clean, unclaimed sentence. Don't let it get overstated into "we discovered routing determines attention importance" (Task 2 below is about why that stronger claim isn't quite right either) or understated into vague "we explore MoE-aware KV compression" (too weak to defend the novelty).

---

## Task 2: Auditing the Correlation Finding

**Is $\rho(R, \hat{A}) \approx 0$ theoretical gold or a red flag?** Neither, on its own — it's *ambiguous*, and treating it as automatically good is a mistake worth catching now rather than in review.

Near-zero linear correlation between two signals is consistent with at least three very different underlying situations:

1. **Genuine complementarity** — both signals carry real, independent information about retention value (the good case, if true).
2. **$R(x_t)$ carries no real signal at all** — if $R(x_t)$ has very low variance across tokens (plausible, since with top-4/top-6 routing, per-token expert combinations may not vary as sharply as you'd want, and the calibration statistics are aggregates), it will show near-zero correlation with *anything*, not because it's usefully orthogonal but because it's nearly a constant. Uncorrelated-because-constant is not the same finding as uncorrelated-because-independent-and-informative, and $\rho$ alone can't tell you which one you have.
3. **A nonlinear or interaction-only relationship** — Pearson $\rho$ only catches linear association; $R$ and $\hat A$ could interact (e.g., XOR-like) in ways that matter for ranking but don't show up as linear correlation.

**The concrete check that resolves this:** report the variance (or entropy) of $R(x_t)$ across your actual token population. If it's small, a near-zero-variance signal mathematically cannot be responsible for a 3–6 point accuracy swing in a linear combination like $S(x_t)$ — that would be a real internal contradiction between your correlation table and your benchmark table, and worth catching before a reviewer does. If the variance is healthy, that's supporting (not conclusive) evidence for case 1.

**Mechanistically, why would routing history in the FFN say anything about a token's value as a future attention key?** Here's the account I'd actually defend, and it also explains a slightly surprising thing your data implies: if both $R$ and $\hat A$ were just two different routes to "this is an important token," you'd expect some *positive* correlation between them, not zero — important tokens tend to get flagged by multiple signals at once. The fact that you're seeing (apparently) clean orthogonality argues for a different story than "two importance detectors agreeing":

- $\hat{A}(x_t)$ is **retrospective and context-dependent** — it measures how much attention this specific token has *already received*, in this specific context, up to this specific point in decoding.
- $R(x_t)$ is **static and content-intrinsic** — it's fixed the moment the token is processed, and reflects what *kind* of token it is (which the router, a learned content-based classifier, effectively tells you for free), independent of how the conversation has unfolded so far.

That distinction predicts exactly the interesting case: a token that is intrinsically important (e.g., an intermediate numeric result in a chain-of-thought) but **hasn't been queried yet** — because the query that will need it is still several steps in the future — will show low $\hat A$ (nothing has attended to it *yet*) but potentially high $R$ (it's the kind of token the routing pattern flags as content-dense). This is precisely the "saliency shift" phenomenon documented elsewhere in the KV cache literature (tokens whose importance changes over the course of decoding, which attention-only methods systematically mishandle because they can only react after the fact). If this is what's happening, $R$ and $\hat A$ being near-orthogonal isn't a coincidence — it's because they're capturing different *tenses*: what's been useful vs. what's the kind of thing that tends to become useful. That's a genuinely defensible mechanistic story, and it directly motivates Task 3's experiment design below. But it's currently a hypothesis your aggregate numbers are *consistent with*, not evidence *for* — the case-study experiment below is what would actually demonstrate it.

---

## Task 3: The Reasoning-Preservation Experiment

You need a causal, token-level probe, not just an aggregate accuracy comparison — the aggregate numbers (once trustworthy) tell you *that* something is happening, not *why*, and "why" is what a reviewer will push on hardest for a claim this specific.

**Step 1 — identify ground-truth critical tokens.** On a sample of GSM8K chain-of-thought traces, identify which tokens are actually load-bearing by perturbation: mask or corrupt each token one at a time (or in small spans — intermediate numeric results, operators, key entities) and check whether the final answer changes. This gives you a real, causally-grounded label — "critical" vs. "filler" — independent of any attention or routing statistic, which is important: you don't want your ground truth to be defined using the same signal you're trying to validate.

**Step 2 — check the clustering claim directly.** For the critical-token set vs. the filler-token set, compute the entropy of their expert-assignment distributions (across layers). If critical tokens really do route through a narrower, more specialized, or more consistent set of experts than filler tokens, that's the direct evidence for the mechanism — not inferred from downstream accuracy, but observed in the routing data itself.

**Step 3 — the retention-rate comparison.** At matched budget, measure what fraction of critical tokens survive eviction under CAKE, SnapKV, and A+R respectively. This is the number that actually explains the GSM8K collapse: if CAKE's layer-uniform budget disproportionately evicts critical tokens (because a layer-wide heuristic has no way to distinguish a critical token from a filler token within its layer budget) while A+R preferentially retains them, you have a mechanistic, not just correlational, story.

**Step 4 — the saliency-shift case study.** Pick 3–5 specific GSM8K examples that flip from wrong-under-SnapKV to right-under-A+R. Trace the specific tokens SnapKV evicted but A+R protected, and check: were they low-$\hat A$-at-time-of-eviction, high-$R$ tokens, consistent with the "hasn't been queried yet but intrinsically important" story? A qualitative example like this, shown alongside the aggregate numbers, is often more convincing to a reviewer than another table.

**Step 5 — the layer-wise heatmap you should already have.** An aggregate $\rho \approx 0$ across all layers could be hiding real structure — strongly positive in some layers, strongly negative in others, cancelling out in the average. Plot $\rho(R, \hat A)$ per layer per model. If there's a clean pattern (e.g., later layers show stronger correlation, consistent with the semantic-vs-syntactic routing story from the earlier mechanistic discussion), that's a genuinely interesting figure. If it's noise everywhere, that itself is important to know before building a narrative around the aggregate number.

---

## Task 4: Adversarial Reviewer — Top 3 Rejection Reasons

**1. "These results look simulated, not measured."** This is the one that actually threatens the paper's credibility, not just its polish. The cross-architecture uniformity described in Task 0 is the kind of thing an experienced reviewer catches within a minute of scanning the tables, and once a reviewer suspects fabricated or synthetic results in one table, they stop trusting *every* table in the paper — including the ones that might be completely legitimate. This has to be resolved before submission, not addressed in a rebuttal.

**2. "Where's the ablation that actually isolates your contribution?"** Your table shows FullKV, Uniform, CAKE, H2O, SnapKV, and "Proposed A+R" — but never $R$ alone ($w_a = 0$). Without that, a reviewer cannot tell whether $R$ is doing real work or whether $A+R$'s gain over SnapKV is coming entirely from better-tuned $w_s$/$w_c$ terms (sink and recency weighting) that have nothing to do with the routing signature you're claiming credit for. This is the single most obvious missing experiment, and it's cheap to add — you already have all the components. Also missing: Ada-KV and InfoKV as direct baselines (currently only qualitatively discussed in your own docs, not in the results table), and any significance test between A+R and SnapKV rather than separate point estimates with separate CIs.

**3. "You haven't shown the mechanism, only the outcome."** Right now the reasoning-preservation claim rests entirely on an aggregate accuracy gap. Reviewers who work in this space will specifically ask "why," and "we hypothesize routing-conditioned tokens are more likely to be reasoning-critical" without the Task 3 causal probe reads as a post-hoc story fitted to a number, not a tested mechanism. This is fixable — it's exactly what Task 3 is for — but it needs to be in the paper, not left as a promissory note in the rebuttal.

*(Fourth, smaller one worth having an answer ready for: your systems numbers are "projected... 1.69×–2.54×," which is analytical/modeled language, not measured. Consistent with the consumer-GPU-profiling caveat from the earlier advisory — have the real wall-clock number by submission time, or state the projection's basis clearly as a limitation.)*

---

## Task 5: Roadmap, Prioritized

1. **Resolve Task 0 first.** Trace every headline number back to raw generations and raw scoring, or rerun from scratch with full per-example logging. Nothing else on this list is worth doing against untrustworthy numbers.
2. **Add the $R$-only ablation** ($w_a = 0$) — cheapest, most reviewer-critical missing piece, reuses everything you already have.
3. **Run the critical-token clustering experiment** (Task 3, steps 1–2) — this is what turns "we observed a correlation with accuracy" into "we found the mechanism," which is the difference between a workshop paper and a paper people actually cite.
4. **Layer-wise $\rho(R,\hat A)$ heatmap per model** — cheap, uses data you already have, resolves whether the aggregate near-zero is hiding real structure.
5. **Add Ada-KV and InfoKV to the results table**, at least on GSM8K and PG19 — reviewers who know this literature will ask for both by name.
6. **Paired significance testing** between A+R and every baseline, not just separate confidence intervals.
7. **The saliency-shift qualitative case study** (Task 3, step 4) — a good figure/example here does real persuasive work beyond the tables.
8. **Real, measured wall-clock kernel benchmark**, replacing the "projected" figure, with the consumer-GPU-profiling limitation stated plainly if that's what you're running on.

Do 1 and 2 before you touch anything else. If the $R$-only ablation shows $R$ is doing real, independent work once the data is trustworthy, you have a paper. If it doesn't, better to find out now than in review.
