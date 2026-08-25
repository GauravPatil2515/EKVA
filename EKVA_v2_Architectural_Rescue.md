# EKVA v2: Architectural Rescue
### From "Experts Own KV Buckets" to "Routing History as a Saliency Signal" — Full Reformulation

*Supersedes the RQ framing in the first advisory doc. Read this one first if you're starting fresh.*

---

## The Verdict, Upfront

Critical Flaw A is real and it's fatal for the original formulation — not fixable by adjusting weights or constants. Self-attention in Mixtral, Qwen1.5-MoE, and DeepSeek-MoE is dense and shared; MoE routing happens only in the FFN block, downstream of attention. "Expert *i*'s KV cache" doesn't correspond to anything a query can restrict itself to without turning this into a novel sparse-attention architecture — which is a different, much bigger project than the one you've built.

The good news: your Task 2 instinct — pivot from *private buckets* to *routing dynamics as a signal informing saliency in the shared cache* — is exactly right, and it survives contact with the architecture. Most of your existing code (calibration hooks, the multi-signal formula's mathematical bones, the EMA recalibration cascade, the benchmark harness) is reusable with a reinterpretation, not a rewrite. The Triton kernel is the one piece that needs to change shape, and it gets *simpler*, not harder, as a result. That's the headline: this is a rescue, not a restart.

---

## TASK 1: Literature Positioning — Differentiation Matrix

| Method | Granularity | Resource | Mechanism | Applies to pretrained checkpoints? | Relation to EKVA v2 |
|---|---|---|---|---|---|
| StreamingLLM / H2O / SnapKV | Token | Attention KV | Attention-magnitude-based eviction | Yes | The baseline EKVA v2 augments |
| Ada-KV | Head, within-layer | Attention KV | Theoretical L1-bound-driven per-head budget | Yes | Orthogonal axis (head vs. our routing feature); authors flag cross-layer as open, nobody's touched cross-expert |
| CAKE / PyramidKV / LAVa | Layer (± head for LAVa) | Attention KV | Entropy/variance-driven per-layer budget | Yes | Orthogonal axis; InfoKV shows this axis has a real failure mode (see Task 3) |
| MoE-nD | Layer | Attention KV, multi-axis (eviction+quant) | "MoE" = routing metaphor for *compression-choice* selection, on a **dense** model | Yes (but not MoE-architecture-specific) | Name collision only — not actually about sparse-MoE expert routing |
| TriRoute | Token | Attention mode + FFN experts + KV bits | **Jointly trained from scratch**, 160M–1.3B models | **No** — architecture research, not inference-time calibration | Different problem category |
| PiKV | Token/page, expert-sharded storage | Attention KV | Distributed serving system: expert-sharded storage + generic per-page eviction (H2O/AdaKV/DuoAttention-style scores) | Yes | Closest MoE-native system; **does not** use routing-history as a saliency *feature* — it's about storage placement, not a novel score. Watch the GitHub repo (BARouter/PERouter additions) |
| MoE-Infinity, FlashMoE, HOBBIT, EdgeMoE | Expert (weights) | **FFN parameter weights**, not KV | Activation-aware expert weight prefetching/offloading (GPU↔CPU↔SSD) | Yes, but wrong resource entirely | Name collision only — "expert cache" here means weight cache |
| EPLB | Expert (placement) | Compute/parallelism | Redistributes/replicates experts across GPUs for load balance | Yes | Unrelated — systems/parallelism, not saliency |
| DeepSeek MLA | Per-token, uniform | Attention KV | Low-rank joint K/V compression into shared latent space, architecture-level | Requires pretraining (or retrofitting, e.g. X-EcoMLA) | **Orthogonal and stackable** — MLA shrinks every token's footprint equally; EKVA v2 decides *which* tokens to keep at all |
| Quest | Token, non-evicting | Attention KV | Query-aware dynamic block retrieval via key-value proximity estimate, retains full cache | Yes | Different mechanism (geometric proximity vs. routing signature); could inspire a query-conditioned EKVA variant (see Task 2, flagged as stretch) |
| InfoKV | Layer/token | Attention KV | Entropy-aware, dense-model, **tested and rejected** layer-adaptive budgets due to reasoning instability | Yes (dense only) | Direct cautionary precedent — test EKVA v2 against the same failure mode explicitly |

**Bottom line for related work:** three of your five original citations (MoE-nD, MoE-Infinity-adjacent systems, TriRoute) need an explicit "this sounds related but isn't" sentence each, or a reviewer will do it for you, less charitably. The genuinely open space — **routing signature as a token-level saliency feature, in a shared cache, on pretrained MoE checkpoints** — has no direct claimant as of this search.

---

## TASK 2: Core Reformulation

### 2.1 What actually exists per-token, for free

At layer *l*, the FFN router computes a top-*k* expert assignment for token *x_t*: this is a **required** computation (you need to know which experts to run the FFN through) — annotating each token with its routing decision costs nothing extra. Across *L* layers, token *x_t* accumulates a **routing signature**:

$$\mathcal{R}_t = \{ E_t^{(1)}, E_t^{(2)}, \dots, E_t^{(L)} \}$$

This is real, already-computed, per-token metadata. It is *not* a KV-cache partition — it's a cheap tag you can attach to each cached K/V entry, the same way you might tag a token with its position or recency.

### 2.2 What the calibration stats now mean

Keep your three offline-calibrated per-expert statistics — but reinterpret them as **descriptors of what kind of token gets routed to expert *e***, not as properties of a private cache:

- $\text{Route}_e$ — global routing frequency of expert *e* (unchanged).
- $\text{Spec}_e = 1 - \text{Evenness}_e$ — how narrow expert *e*'s semantic niche is (unchanged).
- $\bar{H}_e$ — reinterpreted: the average attention entropy exhibited **by tokens acting as queries**, restricted to timesteps where that querying token was itself routed to expert *e*. In plain terms: "when a token belongs to expert *e*'s niche, how focused vs. diffuse is its attention typically?" This is a legitimate, computable quantity — it just describes a population of tokens sharing a routing pattern, not a cache bucket.

### 2.3 The new saliency score

For a cached token *x_t*, define a retention score combining the attention-based signal you'd get from any SnapKV/H2O-style method with a **routing-conditioned** term built from *x_t*'s own routing signature:

$$R(x_t) = \frac{1}{L}\sum_{l=1}^{L} \bar{H}_{E_t^{(l)}} \cdot \log\!\left(1 + \text{Route}_{E_t^{(l)}}\right) \cdot \left(1 + \text{Spec}_{E_t^{(l)}}\right)$$

$$S(x_t) = w_a \cdot \hat{A}(x_t) + w_r \cdot R(x_t) + w_s \cdot \text{Sink}(x_t) + w_c \cdot \text{Recency}(x_t)$$

where $\hat{A}(x_t)$ is standard cumulative/windowed attention mass received by *x_t* (your H2O/SnapKV-style anchor signal — keep it, don't replace it), $\text{Sink}$ protects the first few tokens (StreamingLLM-style), $\text{Recency}$ is a decay term, and weights are tuned via a small calibration sweep. Eviction retains the top-*B* tokens by $S(x_t)$, per layer (since each layer already has its own independent KV cache in the standard architecture — this is a legitimate degree of freedom you were not exploiting incorrectly, unlike the expert-bucket idea).

**This is your entire paper's central question, cleanly stated:** does $R(x_t)$ — a signal you get essentially for free from routing decisions you're already computing — improve retention quality over $\hat{A}(x_t)$ alone, at matched budget? That's a real ablation (attention-only baseline vs. attention+routing), directly comparable to how Ada-KV demonstrated head-adaptivity beats uniform-within-layer. It is falsifiable, cheap to run, and doesn't require you to have invented anything architecturally implausible.

Be honest with yourself about the risk here: it's entirely possible $R(x_t)$ turns out to be **redundant** with $\hat{A}(x_t)$ — i.e., tokens that get high attention already tend to be the ones routed to specialized experts, so the routing signal adds nothing once you already have attention scores. That's a real, non-trivial empirical question, not a foregone conclusion either way. Run the ablation early (see Task 4 checklist) — if $R(x_t)$ alone (with $w_a=0$) is a weak predictor AND doesn't measurably improve $\hat A + R$ over $\hat A$ alone, that's your answer, and it's better to know in week one than in the rebuttal.

### 2.4 Static vs. dynamic vs. query-conditioned eviction

- **Static (post-prefill):** compute $S(x_t)$ once after prefill, evict down to budget, decode with the fixed retained set. Cheapest, matches most of the literature's default setting. Start here.
- **Dynamic (EMA-recalibrated):** update $\hat{A}(x_t)$ via a running EMA during decoding, recombine with the (fixed, since routing signature doesn't change once computed) $R(x_t)$ term periodically. **This is your existing recalibration cascade code, reused almost as-is** — it was originally built to address a different problem (domain-shift recalibration of expert stats), but it maps cleanly onto addressing "saliency shift" (the documented phenomenon where a token's importance changes mid-decode). Recontextualize, don't discard.
- **Query-conditioned retrieval-by-routing-similarity (stretch, don't build this into the core paper):** instead of evicting, retain everything and weight retrieval at each step by similarity between the current query's routing path and each cached token's stored signature — a routing-space analogue of what Quest does in key-value geometric space. Intellectually the most novel version of this idea. Flag it as a discussion-section/future-work item; building and evaluating it properly is more compute and engineering than your timeline supports alongside everything else.

### 2.5 Hardware execution model — and why your kernel plan should change

Because eviction is now **per-token, within each layer's own already-existing KV cache**, this maps directly onto standard PagedAttention/vLLM-style token-eviction infrastructure. You do not need — and should now discard — the "expert-sharded storage" and "variable KV tile count per expert" kernel design from the original plan, because there are no expert-owned shards to tile over anymore.

What you actually need is much smaller in scope: a fused kernel that computes $S(x_t)$ (attention term + a cheap routing-lookup term, since $R(x_t)$ per token is just a handful of table lookups and a small product — no heavy compute) and performs top-*B* selection/compaction, structurally similar to how AdaKV/SnapKV implement their custom eviction kernels. Per-token metadata cost is a few bytes (expert IDs per layer) against d-dimensional K/V vectors — negligible memory overhead.

This is good news for your compute budget: it's a smaller, more standard systems contribution than what you were planning, closer in scope to "extend an existing eviction kernel with one more cheap scoring term" than "invent a new attention memory layout." Keep the Triton v1 skeleton if it's structurally close to a top-k compaction kernel; expect to substantially rewrite the indexing logic since "per-expert tile" no longer exists as a concept.

---

## TASK 3: Mechanistic Hypotheses

Treat everything below as **your best current explanation for numbers you've measured**, not established fact — I don't have independent evidence for these specific correlations, and with 8–64 experts and noisy per-layer statistics, you need a real significance test before asserting a pattern like "16/32 negative layers" means something rather than sampling noise (a permutation test or Spearman correlation with FDR correction across layers would settle this cheaply — do this before writing a sentence about it in the paper).

**Why load-balancing loss plausibly drives the Qwen1.5-MoE correlation toward zero/negative ($r=-0.091$).** The auxiliary balance loss directly penalizes deviation from uniform routing frequency across experts. With 60 experts and top-4 activation, that's a strong, fine-grained pressure toward frequency uniformity — which mechanically squeezes out whatever "natural" relationship might otherwise exist between an expert's routing frequency and the attention-entropy character of the tokens it receives. If frequency is being forced toward $1/60$ regardless of semantic content, it stops carrying much information, and its correlation with anything else becomes dominated by noise. This is a direct, testable implication of the auxiliary-loss mechanism, not a guess about semantics.

**Why Mixtral is weaker and sign-mixed across layers ($r=0.426$, 16/32 negative).** Coarser granularity (8 experts, top-2) means the balance loss has less room to homogenize — each expert covers a broader slice, so some residual specialization signal can survive the balancing pressure. Sign-flipping across layers is consistent with the well-documented pattern that early layers tend to route on more syntactic/lexical features (which are inherently more uniformly distributed — hence higher, more balanced frequency, weaker informative signal) while later layers route more semantically (tighter, more concentrated specialization — hence a cleaner frequency-entropy relationship). This predicts your correlation should be more consistently positive in later layers than early ones — worth checking as a specific, falsifiable sub-claim rather than reporting the aggregate 16/32 split as a single number.

**Why DeepSeek-MoE shows a strong positive correlation ($r=0.791$).** This is the one I'd bet on being real rather than noise, because it follows directly from documented architectural design choices, not speculation. DeepSeekMoE's signature contributions are *fine-grained expert segmentation* (many small experts instead of few large ones) and *shared-expert isolation* (a small number of always-active experts that absorb generic/common patterns). The shared experts pulling out the "everyone needs this" traffic means the *routed* experts are left to specialize more cleanly — the balance loss still applies to the routed pool, but the shared experts have already siphoned off the traffic that would otherwise force routed-expert frequency toward uninformative uniformity. That structurally preserves a real frequency–entropy relationship among the routed experts: frequently-activated fine-grained experts likely still correspond to broadly-useful (higher-entropy) patterns, infrequent ones to narrow (lower-entropy) specializations, in a way the balance loss doesn't fully erase. This is a genuinely interesting, citable mechanistic story if it holds up under the significance test above — it's also a nice reason to keep DeepSeek-MoE in your architecture set even though it's the most expensive to run.

**On the reasoning-collapse claim (layer-level methods failing on GSM8K/MATH vs. expert-conditioned retention holding up).** This is plausible but currently unverified — treat it as your headline hypothesis to test, not a result to assume. The mechanistic story, if true, would go: layer-level budget cuts (CAKE/PyramidKV-style) compress a whole layer uniformly based on layer-wide attention statistics, with no regard for which *specific* tokens within that layer carry irreplaceable reasoning content (an intermediate computed value, a critical operator). A multi-step reasoning chain can be broken by losing a single load-bearing token even if the layer's aggregate budget looks generous. If routing-conditioned retention preferentially protects tokens that are repeatedly routed to rare/specialized experts (under the hypothesis that "hard," information-dense tokens cluster there), it would selectively protect exactly the tokens a layer-blind budget might drop. That's a coherent story — but it rests on the unverified premise that reasoning-critical tokens (numbers, intermediate results) actually do get routed to identifiable, low-frequency experts more than filler tokens do. **Check this premise directly** — e.g., manually inspect routing patterns on a handful of GSM8K traces before building the whole narrative around it — rather than backing into it from aggregate accuracy numbers alone. If the premise doesn't hold, contribution #4 in the blueprint below should be dropped or softened, not forced.

---

## TASK 4: Experimental Protocol

### Formal metrics (fixing the red flags directly)

- **Accuracy** — exact-match (GSM8K, MATH) or execution-pass@1 (HumanEval, MBPP). Report per-benchmark, never pooled into one "quality" number.
- **Perplexity** — $\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i \mid x_{<i})\right)$, on PG19/Proof-Pile.
- **Retained Quality Ratio** — $RQ = \frac{Q_{\text{compressed}}}{Q_{\text{full}}} \times 100$, where $Q$ is the *task-appropriate* primary metric above, computed from **actual model generations on actual benchmark data**, per benchmark. The fact that your simulator previously produced identical percentages (39.53%, 63.54%) across three architecturally distinct models is close to a statistical impossibility if these came from real evaluation runs with real sampling noise — that's the signature of an analytical formula standing in for an evaluation, not measurement. This needs a full rebuild against real generation + real scoring before any number from it goes in the paper.
- **TTFT / TPOT** — standard prefill/decode latency definitions, measured end-to-end, not on isolated kernel loops.
- **Memory footprint** — peak KV cache bytes at budget $B$, measured, not computed analytically only.
- **Confidence intervals** — bootstrap resampling (≥1000 resamples) over eval examples, report 95% CI on every headline number. Non-negotiable given the metric red flags already found once.

### Datasets
GSM8K, MATH (reasoning); HumanEval, MBPP (code, execution-scored); PG19, Proof-Pile (perplexity, long-document); Needle-in-a-Haystack (retrieval accuracy vs. depth/length); LongBench (multi-task, matches most baseline papers' own eval suites so your numbers are directly comparable).

### Baselines to beat
FullKV (upper bound), Uniform (no adaptivity — sanity floor), StreamingLLM, H2O, SnapKV, PyramidKV, Ada-KV, CAKE, InfoKV. PiKV: cite and qualitatively differentiate rather than fully reproduce — it's a distributed serving system, reproducing it faithfully is its own multi-week project and not central to your claim.

### Sensitivity sweeps
$B_{\min} \in \{16, 32, 64, 128\}$; calibration corpus size (does the routing-conditioned term need much calibration data, or does it saturate fast — this is a nice cheap robustness result); EMA window $W$; and critically, a **weight ablation** on $(w_a, w_r, w_s, w_c)$ — this is where you demonstrate $R(x_t)$'s marginal contribution directly, which is the paper's core claim.

---

## TASK 5: Revised Paper Blueprint

### Title
**Recommended:** *"Routing as a Signal: Expert-Path-Aware Token Retention for KV Cache Compression in Sparse MoE Inference"*
Alternates: *"What Your Expert Says About You: Routing-Conditioned KV Cache Eviction in Sparse MoE LLMs"* · *"Free Signal: Reusing MoE Routing Decisions for KV Cache Retention"* (leans into the "this info is already computed, costs nothing extra" framing, which is a genuinely strong selling point)

### Abstract template
> KV cache eviction methods decide which tokens to retain using attention-based importance, at the token (H2O, SnapKV), head (Ada-KV), or layer (CAKE, PyramidKV) granularity. In sparse Mixture-of-Experts LLMs, each token additionally carries a routing signature — the sequence of experts it activates across layers — that is computed for free during the forward pass but discarded by existing eviction methods. We show this signature is a informative, complementary signal for token retention: combined with calibrated per-expert statistics (routing frequency, specialization, attention entropy of a token's expert niche), it predicts retention value beyond what attention scores alone capture. We introduce a training-free, calibration-based retention score combining routing-conditioned and attention-based signals, evaluate it across three sparse-MoE architectures with markedly different load-balancing and expert-granularity designs (Mixtral-8x7B, Qwen1.5-MoE-A2.7B, DeepSeek-MoE-16B), and characterize when and why the routing signal helps — including [a preserved reasoning-task result, if TASK 3's premise holds up empirically] — alongside a lightweight kernel realizing the retention score at negligible overhead within existing token-eviction infrastructure.

*(Same rule as before: don't lock the bracketed claim until you've actually checked the reasoning-premise experiment from Task 3.)*

### Contribution list
1. Identify that a token's cross-layer routing signature — already computed, currently discarded — carries retention-relevant information not fully captured by attention-based importance alone, in sparse-MoE LLMs.
2. A training-free, calibration-based retention score combining routing-conditioned and attention-based signals, applicable to any pretrained sparse-MoE checkpoint without architecture modification.
3. A cross-architecture characterization (8/60/64 experts, three different load-balancing/shared-expert designs) explaining, mechanistically, why the routing signal's usefulness varies by architecture — grounded in documented design choices (fine-grained segmentation, shared-expert isolation), not just observed numbers.
4. *(Conditional on the Task 3 premise check)* Evidence that routing-conditioned retention narrows the disproportionate degradation attention-only and layer-uniform methods show on multi-step reasoning tasks at matched budget.
5. A PagedAttention-compatible kernel fusing routing-score lookup into existing top-k eviction, with measured (not purely analytical) wall-clock decode latency.

### Section outline
**Intro** — lead with the architecture diagram showing dense shared attention vs. sparse FFN routing, and the naive private-bucket idea failing against it, before introducing the real mechanism. Showing your own dead end briefly is good scientific storytelling and preempts the exact objection your reviewer would otherwise raise. **Related Work** — the Task 1 matrix, in prose. **Method** — the $S(x_t)$ formulation, static vs. EMA-dynamic. **Experiments** — the core ablation ($\hat A$ alone vs. $\hat A + R$) across three architectures at matched budget; sensitivity sweeps; the reasoning-task deep dive. **Systems** — kernel design and measured latency. **Limitations** — state plainly: doesn't apply to MoE-in-attention architectures (MoSA-style) where the original private-cache idea would actually have been valid instead; correlation claims carry significance-testing caveats; consumer-GPU profiling caveat carried over from the first advisory doc. **Conclusion.**

### Venue and rebuttal strategy
ES-FoMo/ENLSP workshop remains the right first target given the empirical-characterization shape of the paper. If contribution #4 holds up cleanly, it's a legitimate EMNLP/NAACL Findings case. The single rebuttal question to pre-empt in the draft itself, not just in your head: *"Isn't this just attention-based eviction with an extra feature bolted on?"* Answer it head-on with the weight-ablation result — show $R(x_t)$ alone is a mediocre predictor (attention still does most of the work) but that $\hat A + R$ beats $\hat A$ alone by a real, budget-matched margin. The claim is "cheap complementary signal," not "better signal" — smaller, more defensible, and it's the honest description of what you'll actually find either way.

---

## What Changed From the First Advisory Doc

- RQ1 (granularity ablation: per-expert vs. layer-aggregated) is **retired** — "per-expert budget" isn't a coherent object anymore. Its replacement is the $\hat A$ vs. $\hat A + R$ ablation above, which is the new backbone.
- RQ2 (entropy/frequency decoupling) **survives and gets promoted**, now grounded in real architectural mechanisms (load-balance loss strength, fine-grained + shared-expert design) rather than being a loose correlation to explain after the fact.
- RQ3 (transferability) **survives largely unchanged**, still worth tying to the InfoKV reasoning-instability precedent.
- RQ4 (kernel/roofline) **survives but shrinks further** — from "novel expert-sharded attention kernel" to "fused scoring term in a standard eviction kernel." This was already my top scope-creep concern in the first doc; the architecture fix happens to solve it as a side effect.
