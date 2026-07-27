Alright, I read through all four docs plus your README summary. Here's my honest take, no sugarcoating.

The one thing that could sink this

Before anything about feasibility - I searched for prior work, and this is the part you need to sit with before you write another line of code.

Your "key innovation" is entropy-driven, per-expert KV budget allocation for MoE models. I found PiKV (arXiv 2508.06526, accepted at ICML 2025 ES-FoMo workshop) - it's an open-source KV cache system built specifically for MoE, and as of September 2025 it shipped a PERouter (Predictive-Entropy) and a BARouter (Budget-Aware) router. That's your core idea, already implemented, already published, already open source with a GitHub repo people are actively adding features to.

On top of that:

There's an MDPI paper from July 2025 doing entropy-guided KV budget allocation (layer-wise, not expert-wise, but same mechanism)
A 2026 paper doing the identical entropy-proportional budget formula you'd use, just at the layer level instead of expert level - literally the same equation (budget ∝ entropy / sum of entropy)
TriRoute (mid-2026) unifies attention/expert/KV-cache routing under one learned controller, which is a superset of what you're proposing
MoE-nD (2026) does per-layer multi-axis KV cache routing for MoE

So the "not all experts are equally attention-intensive, let's budget KV proportional to entropy" idea isn't just plausible, it's already out there, in multiple forms, some specifically for MoE. That doesn't mean your project is worthless. But it does mean the framing in your README - "the core novelty is..." - is not accurate anymore, if it ever was. You'd be redoing something that exists, with less compute and no lab behind you.

I'd rather tell you this now than let you spend three months building toward a paper that gets desk-rejected for lack of novelty, or worse, gets scooped in front of you while you're still calibrating on Qwen1.5-MoE.

Is it feasible as a student, hardware-wise?

Actually yeah, this part is solid. Your two-track setup (CPU pipeline dev on the 3050, real runs on rented Colab A100) is the right call. Qwen1.5-MoE-A2.7B fitting in 6GB with offload is a smart first target. The 22 passing CPU tests tell me the plumbing works. That's genuinely good engineering discipline for a student project - a lot of people skip straight to "run it on the big model" and then debug blind.

Where it gets hard: Weeks 8-11 (roofline + Triton kernels) need real GPU time and Nsight Compute access, which means real money for Colab A100/H100 rental, and Triton kernel debugging is not a weekend skill - even people who've written CUDA before find Triton's autotuning and memory model annoying to debug. Your own docs already have a fallback ("report software-simulator results, describe the kernel as proposed") which tells me you already suspected this might not fully land. That's honest of past-you, keep that fallback.

What I'd actually check before writing more code

1. Read the PiKV paper and code properly, today, before touching Week 5 onward. Not skim it - actually go through PERouter and BARouter's implementation. You need to know exactly what they do, because that becomes your related-work section either way. If their entropy formula is basically yours, you have three options: differentiate hard, pivot the angle, or accept this becomes a smaller reproduction/extension study rather than a novel-method paper.

2. Where's your actual differentiation, if there is one? A few real candidates, looking at what you've built:

The roofline + hardware-motivated claim (Weeks 8-9) is less commonly paired with the entropy story in what I found. If entropy genuinely correlates with compute-bound vs memory-bound regime per expert, that's a more defensible, harder-to-scoop angle than "entropy → budget" alone.
Your Triton kernel with variable tile count per expert (v1/v2) is an actual systems contribution if it works and you benchmark it honestly - that's harder to copy-paste than a policy formula, and PiKV's readme doesn't mention a custom fused kernel like this.
Multi-signal (entropy + routing + specialization) is a reasonable extension if the pure-entropy version turns out to match prior work too closely.

3. Don't let this become a paper about "did entropy correlate with budget," because that question is already answered by three other papers. Make it about something PiKV and friends didn't test: does this hold across Mixtral vs DeepSeek-MoE with very different expert counts (8 vs 64)? Does the roofline story change per model family? That's a real empirical gap you could fill even if the core policy isn't novel.

Is it "significant" - real talk

As a novel-methods paper for a top venue: no, not as currently framed. The idea's been published. As a systems/engineering contribution (working Triton kernel + honest multi-model benchmark + roofline analysis), it's more defensible - kernels and clean empirical comparisons are undervalued by "is the algorithm novel" thinking but valued by people who actually read the appendix.

As a learning project and portfolio piece for you - genuinely strong. You'd come out the other side having touched calibration, MoE internals, Triton, Nsight profiling, and a real benchmark sweep across LongBench/RULER. Given where you're headed (agentic/GenAI roles, 10-20 LPA range), "I built and profiled a custom Triton kernel for KV cache allocation in MoE models" is a much stronger interview story than most student projects, scooped idea or not. Interviewers rarely check if your exact idea appeared on arXiv three months before you started.

My honest recommendation

Don't abandon it, but don't start Week 5 onward assuming the entropy-budget idea is yours to claim. Spend a day reading PiKV and the entropy-KV papers I found, then decide: pivot to the roofline+kernel angle as the real contribution, or explicitly position this as "reproducing and extending X for MoE-specific hardware profiling," which is a legitimate and honest framing for a student research project even if it's not "novel."

One more thing - given your existing interest in learned KV cache eviction policies (that direction-1 assessment we talked through before), this project and that research thread are basically the same territory. Worth deciding if EKVA is the vehicle for that broader interest or a distraction from it.

Want me to actually pull the PiKV paper and repo and do a proper side-by-side against your derive_kv_budget() / EKVAPolicy design, so you know exactly where you overlap and where you don't?

do research what we can do in this topic to get significant results , also to propely work and frame this as research project

Reframed research from novelty toward empirical characterization and systems validation
Reframed research from novelty toward empirical characterization and systems validation

Good, let's dig into this properly. I pulled up more papers on this specific niche, and there's a detail from the last search that actually matters a lot for you.

The PiKV nuance you need to know

Looking closer: PiKV's PERouter (Predictive-Entropy) and BARouter (Budget-Aware) showed up in the GitHub changelog on 09/06/2025 - but the actual arXiv paper (2508.06526) was published in August 2025, before those features landed. So there's a real chance PERouter is a shipped code feature that was never rigorously evaluated in a paper. Nobody's published numbers on "does entropy-based per-expert KV budgeting actually work, tested properly, with ablations, across multiple MoE architectures."

That's a meaningful distinction. "Someone wrote code for this" and "someone proved this works with a controlled study" are very different bars. The second one is still open. That's your opening, if you take it.

But - and this is the part I want you to sit with - layer-wise and head-wise attention-entropy KV budgeting is done to death. CAKE, Ada-KV, MEDA, the MDPI entropy paper, the Information-Aware KV paper - all of them do "compute attention entropy, allocate budget proportionally." The mechanism itself isn't the innovation anymore, hasn't been for a while. What's actually still open is whether that mechanism, applied at the expert granularity in a sparse MoE model, behaves differently than it does at the layer level, and whether it's reliable enough to trust in production. That second question is explicitly called out as unsolved - the Information-Aware KV paper's limitations section says adaptive allocation "improves performance for some models but can destabilize reasoning performance for others" and calls "more robust and architecture-aware allocation strategies" an open direction. That's from mid-2026. That's your gap, written down by someone else, waiting for someone to fill it.

Reframing the project: from "we invented X" to "we answered questions nobody's answered"

This is the mental shift that matters most. Right now your README reads like a novel-method paper ("the core novelty is entropy-driven per-expert allocation"). Given what's out there, that framing will get shredded by any reviewer who's read CAKE or PiKV. So drop it. Reframe around research questions instead. A paper built around honest, falsifiable RQs is harder to scoop and more defensible even when individual techniques already exist, because the contribution is the answer, not the idea.

Here's what I'd actually build the paper around, in priority order:

RQ1 - Does expert-level granularity beat layer-level granularity, at equal total budget, in MoE models?
This is dead simple to run with what you've already built. Take your simulator, run entropy-based budgeting at the layer level (like CAKE/MEDA do) versus at the expert level (your EKVA), same total budget, same models. If expert-level wins, you've shown MoE's structure genuinely needs finer granularity than dense-model methods assume - that's a real, citable finding. If it doesn't win, that's also a real finding (and an honest one - "expert-level complexity adds engineering cost without matching gains" is a useful negative result, not a failure). Either way you have something to say. This should be your first experiment, this week, before anything else - it's cheap (your CPU simulator can already do it) and it tells you whether the whole project's premise holds.

RQ2 - How does attention entropy relate to routing frequency, and does that relationship break the "high entropy = high budget" assumption?
Your multi-signal idea already gestures at this, sharpen it into the real question: in a trained MoE router (with its load-balancing auxiliary loss forcing roughly uniform expert usage), do frequently-routed experts also tend to be high-entropy? Or are they decoupled - some experts handle lots of tokens but with narrow/predictable attention, others handle few tokens but complex ones? Nobody's characterized this directly for KV budgeting purposes. This is a diagnostic contribution - you're not proposing a policy, you're mapping a relationship that determines whether entropy-only allocation is even the right signal. If they're anti-correlated in some models, that's a genuinely interesting result and directly explains why pure-entropy methods "destabilize" on some architectures (per that Information-Aware paper's admitted limitation).

RQ3 - Does calibration transfer, or is this brittle?
Almost every entropy-KV paper calibrates on one dataset and reports numbers on that same distribution, then maybe checks generalization as an afterthought. Explicitly test: calibrate entropy on WikiText, evaluate the derived budget on LongBench/RULER/code. Calibrate on code, test on QA. If your budgets hold up across domains, that's a real robustness claim nobody's made cleanly for MoE. If they don't, you document exactly how much they degrade and why - which is the "architecture-aware allocation" future work the field is asking for.

RQ4 - What does the hardware actually look like, and does your kernel realize the theoretical savings?
This is your least crowded, most defensible piece, because it's systems work, not an algorithm claim. Roofline classification of individual MoE experts (compute-bound vs memory-bound) isn't something I found anyone doing carefully. If you show experts split into distinct roofline regions and that split correlates with entropy, that's a hardware-grounded story literally nobody else has, and your Triton kernel becomes proof-of-concept rather than decoration. This is also the part where "I built a working fused kernel that gets real speedup on real GPU" is a much stronger portfolio artifact than any policy claim, because it can't be argued with - it either runs faster or it doesn't.

What "significant" should mean for you, concretely

Not "beats state of the art." You don't have the compute or team for that, and honestly neither does most of the field right now, everyone's iterating on 1-2 percentage point deltas. Significant for a student project means:

A clean ablation nobody else ran (RQ1, layer vs expert granularity) - cheap, fast, gives you a real result within days.
A diagnostic finding with an actual number attached (RQ2's correlation coefficient between entropy and routing frequency, across 3 models) - this is the kind of thing that gets cited even by papers proposing something completely different, because it's a fact about how MoE models behave, not a method.
A working kernel with a benchmarked speedup, honestly reported even if it's small (RQ4) - "we got 1.4x on the memory-bound experts, no gain on compute-bound ones, here's why" is a legitimate, useful result. Don't inflate it.
An honest robustness study (RQ3) that either supports or complicates the entropy-budgeting story - both outcomes are publishable, because the field has flagged this exact gap as open.
What to cut or de-prioritize

Weeks 5-6's full benchmark sweep across LongBench/RULER/InfiniteBench on Mixtral and DeepSeek-MoE-16B - that's a lot of Colab-A100-hours for marginal payoff if the real story is RQ1/RQ2. I'd run it on 2 tasks, not the full subset, once RQ1 tells you granularity actually matters. No point burning GPU credits benchmarking a method you haven't validated is worth benchmarking yet.

Seven allocation policies compared against four eviction strategies (16 combos) - fine for the simulator since it's free CPU compute, but don't try to report all 16 in the paper. Report the 3-4 that tell the RQ1-RQ4 story, put the rest in an appendix table.

DeepSeek-MoE-16B as primary target - it needs ~32GB and 8-bit tricks, that's expensive rented time for a third model. I'd do Qwen1.5-MoE (cheap, first) and Mixtral-8x7B (primary) properly, and treat DeepSeek as a "if we have leftover Colab credits" stretch goal, not a core deliverable. Two well-executed models beats three rushed ones.

Realistic target and honest framing

Given you're solo, undergrad, rented compute - a NeurIPS/ICML main-track paper isn't realistic and I'd be lying if I said otherwise. What is realistic: a solid arXiv preprint plus a workshop submission (ES-FoMo, ENLSP, or similar efficient-ML workshops take exactly this kind of empirical/systems work, and workshop bars are calibrated for exactly this scope). Frame the abstract honestly - "we present a controlled empirical study of..." not "we propose a novel method that achieves state-of-the-art..." Reviewers can smell the second one from a mile away when the technique isn't actually new, and it undermines work that's otherwise solid.

For your IEEE HardwAIre-style credibility and resume, honestly, "ran a rigorous ablation study that answers an open question flagged in a 2026 paper, plus a working Triton kernel with measured speedup" reads better to a hiring manager than an inflated novelty claim that falls apart under one follow-up question.

What I'd do literally this week

Run RQ1 first, on your CPU simulator, no GPU needed. Take your existing calibration output, compute layer-aggregated entropy budgets vs expert-level entropy budgets at matched total budget, compare PPL under truncation. That's maybe 2-3 days of work with code you already have, and it tells you immediately whether the rest of this plan is worth building on top of, or whether you need to rethink further.