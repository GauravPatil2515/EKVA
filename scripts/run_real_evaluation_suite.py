"""EKVA v2: Real Evaluation Suite — replaces the formula-generated results pipeline.

Everything in this file is measured from actual model forward passes: real HF model
weights, real GSM8K/HumanEval/PG19/NIAH data, real routing-hook capture, real KV
cache compaction, real generation, real scoring. No task score is ever computed
from a closed-form degradation formula.

This directly answers the Task 0 data-integrity audit in
EKVA_v3_Data_Audit_and_Mechanism.md:
  1. Every number here traces to a list of real per-example generations and
     real correct/incorrect (or real loss) values — see `raw_scores` in the output.
  2. Bootstrap CIs are computed by resampling those real per-example arrays, not by
     resampling noise added to a formula.
  3. A real per-layer CAKE baseline is included (entropy-weighted per-layer budget
     from real attention maps), not a coefficient copy of the other baselines.
  4. rho(R(x_t), A_hat(x_t)) and the variance of R(x_t) are computed from real
     routing signatures and real attention captured during real prefill, on a real
     text corpus — not from independently-seeded synthetic tensors.
  5. Paired bootstrap significance tests are run between A+R and every baseline.

Run with (on a GPU with enough VRAM for the target model — see ekva/models):
    python3 scripts/run_real_evaluation_suite.py --model qwen1.5-moe-a2.7b \
        --gsm8k-samples 200 --humaneval-samples 80 --pg19-docs 30 --niah-samples 40

Output: output/real_eval_<model>.json (per-model) — merge across models for the
paper table. Each cell reports {mean, ci_95, n} computed from `raw_scores`, plus a
top-level `protocol` block recording exactly what was run (dataset, split, sample
count, generation config, quantization) so the eval protocol is never implicit.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.retention.routing_signature import MoERoutingHook
from ekva.retention.saliency import (
    ExpertProfile,
    compute_routing_conditioned_score,
    combined_token_saliency,
)
from ekva.retention.eviction import select_topk_indices, compact_kv_tensor

BASELINES = ["FullKV", "Uniform", "CAKE", "H2O", "SnapKV", "R-only", "A+R (EKVA v2)"]
BUDGETS = [0.20, 0.40, 0.60, 0.80, 1.00]


# ---------------------------------------------------------------------------
# Statistics: real bootstrap CI + paired bootstrap significance test.
# ---------------------------------------------------------------------------

def bootstrap_ci(values: List[float], n_boot: int = 1000, ci: float = 0.95, seed: int = 0) -> Tuple[float, float, float]:
    """Bootstrap mean and 95% CI over REAL per-example values (not formula noise)."""
    rng = np.random.default_rng(seed)
    arr = np.array(values, dtype=np.float64)
    mean_val = float(np.mean(arr)) if len(arr) else 0.0
    if len(arr) <= 1:
        return round(mean_val, 2), round(mean_val, 2), round(mean_val, 2)
    boot_means = np.array([np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)])
    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_means, alpha * 100))
    high = float(np.percentile(boot_means, (1.0 - alpha) * 100))
    return round(mean_val, 2), round(low, 2), round(high, 2)


def paired_bootstrap_test(a_values: List[float], b_values: List[float], n_boot: int = 5000, seed: int = 0) -> Dict:
    """Paired bootstrap significance test for mean(a) > mean(b) on matched examples.

    a_values / b_values must be per-example scores for the SAME examples in the
    SAME order (e.g. A+R vs SnapKV on identical GSM8K questions), so within-example
    difficulty variance cancels out — this is the test the Task 0 audit asked for
    instead of comparing separate, unpaired confidence intervals.
    """
    a = np.array(a_values, dtype=np.float64)
    b = np.array(b_values, dtype=np.float64)
    if len(a) != len(b) or len(a) == 0:
        return {"diff_mean": None, "p_value": None, "n": 0}
    diff = a - b
    rng = np.random.default_rng(seed)
    n = len(diff)
    boot_diffs = np.array([np.mean(rng.choice(diff, size=n, replace=True)) for _ in range(n_boot)])
    observed = float(np.mean(diff))
    # Two-sided symmetric p-value
    p = min(float(np.mean(boot_diffs <= 0)), float(np.mean(boot_diffs >= 0))) * 2.0
    p = min(1.0, float(p))
    ci_low, ci_high = float(np.percentile(boot_diffs, 2.5)), float(np.percentile(boot_diffs, 97.5))
    return {
        "diff_mean": round(observed, 3),
        "diff_ci_95": [round(ci_low, 3), round(ci_high, 3)],
        "p_value": round(p, 4),
        "significant_at_0.05": bool(p < 0.05),
        "n": int(n),
    }


# ---------------------------------------------------------------------------
# Model loading.
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, use_4bit: Optional[bool] = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = get_model_spec(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(spec.hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "low_cpu_mem_usage": True}
    quant_used = "none"
    if device == "cuda":
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        want_4bit = use_4bit if use_4bit is not None else (total_vram_gb < 24.0)
        if want_4bit:
            try:
                import bitsandbytes  # noqa: F401
                load_kwargs["load_in_4bit"] = True
                load_kwargs["device_map"] = "auto"
                quant_used = "4bit-nf4"
            except ImportError:
                load_kwargs["torch_dtype"] = torch.float16
                load_kwargs["device_map"] = "auto"
                quant_used = "fp16"
        else:
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
            quant_used = "fp16"
    else:
        load_kwargs["torch_dtype"] = torch.float32
        quant_used = "fp32-cpu"

    model = AutoModelForCausalLM.from_pretrained(spec.hf_id, **load_kwargs)
    model.eval()
    return model, tokenizer, device, quant_used


# ---------------------------------------------------------------------------
# Real calibration: expert profiles + rho(R, A_hat) from real forward passes.
# ---------------------------------------------------------------------------

def build_real_calibration(model, tokenizer, device, calib_texts: List[str], max_len: int = 512) -> Tuple[Dict[int, ExpertProfile], float, float]:
    """Runs real prefill over a calibration corpus and derives:
      - expert_profiles: real per-expert (entropy, routing_freq, specialization)
      - corr: real Pearson rho(R(x_t), A_hat(x_t)) pooled over all tokens seen
      - r_variance: real variance of R(x_t) across the token population (Task 2 check)
    """
    routing_counts: Dict[int, int] = {}
    routing_entropy_sum: Dict[int, float] = {}
    routing_entropy_n: Dict[int, int] = {}
    all_r, all_a = [], []

    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            with MoERoutingHook(model) as hook:
                out = model(input_ids, output_attentions=True)
                try:
                    sig = hook.get_signature()
                except RuntimeError:
                    continue

        # Real per-token attention-received score from the last layer (matches
        # the A_hat used at eviction time in evict_and_generate below).
        if out.attentions is not None and len(out.attentions) > 0:
            last_attn = out.attentions[-1][0]  # (H, T, T)
            a_hat = last_attn.mean(dim=0).mean(dim=0).detach().cpu().numpy()  # (T,)
        else:
            continue

        # Real per-expert routing frequency + real per-expert attention-entropy
        # of the tokens that were routed there (needed for ExpertProfile.entropy,
        # which is defined as the average attention-entropy of tokens the expert sees).
        idx = sig.expert_indices[0]  # (T, L, K)
        T, L, K = idx.shape
        # token-level attention entropy over the (real) attention distribution
        # this token RECEIVED as a key, used as a proxy for "how spread out is
        # this token's importance" — consistent with entropy's role elsewhere.
        probs = last_attn.mean(dim=0)  # (T_q, T_k) averaged over heads
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        tok_entropy = (-(probs * (probs.clamp(min=1e-12)).log()).sum(dim=-1)).detach().cpu().numpy()  # (T_q,)

        idx_np = idx.numpy()
        for t in range(T):
            ent_t = float(tok_entropy[min(t, len(tok_entropy) - 1)])
            for l in range(L):
                for k in range(K):
                    e = int(idx_np[t, l, k])
                    routing_counts[e] = routing_counts.get(e, 0) + 1
                    routing_entropy_sum[e] = routing_entropy_sum.get(e, 0.0) + ent_t
                    routing_entropy_n[e] = routing_entropy_n.get(e, 0) + 1

        # Real R(x_t) using whatever profiles we have accumulated SO FAR is
        # circular; instead we compute rho in a second pass below once profiles
        # are finalized. Here we only stash real A_hat per text for that pass.
        all_a.append(a_hat)

    num_experts = get_model_spec_num_experts(model)
    profiles: Dict[int, ExpertProfile] = {}
    for e in range(num_experts):
        freq = float(routing_counts.get(e, 0))
        ent = routing_entropy_sum.get(e, 0.0) / max(1, routing_entropy_n.get(e, 1))
        # Specialization: how concentrated this expert's routing volume is
        # relative to a uniform baseline across all seen experts.
        total = max(1, sum(routing_counts.values()))
        p_e = freq / total
        uniform_p = 1.0 / max(1, len(routing_counts))
        specialization = float(np.clip((p_e - uniform_p) / max(uniform_p, 1e-8), 0.0, 1.0)) if routing_counts else 0.0
        profiles[e] = ExpertProfile(entropy=max(0.1, ent), routing_freq=max(1.0, freq), specialization=specialization)

    # Second pass: real R(x_t) vs real A_hat(x_t), pooled.
    for text, a_hat in zip(calib_texts, all_a):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            with MoERoutingHook(model) as hook:
                model(input_ids)
                try:
                    sig = hook.get_signature()
                except RuntimeError:
                    continue
        r = compute_routing_conditioned_score(sig, profiles, num_experts=num_experts)[0].detach().cpu().numpy()
        n = min(len(r), len(a_hat))
        all_r.append(r[:n])
        # a_hat already trimmed to same text; re-slice defensively
    all_r_flat = np.concatenate(all_r) if all_r else np.array([0.0])
    all_a_flat = np.concatenate([a[: len(r)] for a, r in zip(all_a, all_r)]) if all_r else np.array([0.0])

    if len(all_r_flat) > 1 and np.std(all_r_flat) > 1e-8 and np.std(all_a_flat) > 1e-8:
        corr = float(np.corrcoef(all_r_flat, all_a_flat)[0, 1])
    else:
        corr = 0.0
    r_variance = float(np.var(all_r_flat))

    return profiles, corr, r_variance


def compute_layerwise_correlation(model, tokenizer, profiles, num_experts, device, calib_texts: List[str], max_len: int = 512) -> List[float]:
    """Real per-layer rho(R_l(x_t), A_hat_l(x_t)), addressing Task 3 step 5 / Task 5
    item 4: an aggregate rho near 0 can hide layer-specific structure (some layers
    strongly positive, some strongly negative, cancelling in the average). Unlike
    build_real_calibration's pooled rho, this keeps each layer's routing score and
    that SAME layer's real attention separate instead of averaging across layers.
    """
    num_layers = None
    per_layer_r: List[List[np.ndarray]] = []
    per_layer_a: List[List[np.ndarray]] = []

    for text in calib_texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = enc.input_ids.to(device)
        if input_ids.shape[1] < 4:
            continue
        with torch.no_grad():
            with MoERoutingHook(model) as hook:
                out = model(input_ids, output_attentions=True)
                try:
                    sig = hook.get_signature()
                except RuntimeError:
                    continue
        if out.attentions is None or len(out.attentions) == 0:
            continue

        idx = sig.expert_indices[0]  # (T, L, K)
        T, L, K = idx.shape
        if num_layers is None:
            num_layers = L
            per_layer_r = [[] for _ in range(L)]
            per_layer_a = [[] for _ in range(L)]

        score_table = torch.zeros(num_experts, dtype=torch.float32)
        for e, prof in profiles.items():
            if e < num_experts:
                score_table[e] = prof.score_multiplier

        for l in range(min(L, len(out.attentions))):
            layer_r = score_table[idx[:, l, :].clamp(0, num_experts - 1)].mean(dim=-1).numpy()  # (T,)
            layer_attn = out.attentions[l][0]  # (H, T, T)
            layer_a = layer_attn.mean(dim=0).mean(dim=0).detach().cpu().numpy()  # (T,)
            n = min(len(layer_r), len(layer_a))
            per_layer_r[l].append(layer_r[:n])
            per_layer_a[l].append(layer_a[:n])

    if num_layers is None:
        return []

    layer_rhos = []
    for l in range(num_layers):
        r_flat = np.concatenate(per_layer_r[l]) if per_layer_r[l] else np.array([0.0])
        a_flat = np.concatenate(per_layer_a[l]) if per_layer_a[l] else np.array([0.0])
        if len(r_flat) > 1 and np.std(r_flat) > 1e-8 and np.std(a_flat) > 1e-8:
            layer_rhos.append(round(float(np.corrcoef(r_flat, a_flat)[0, 1]), 4))
        else:
            layer_rhos.append(0.0)
    return layer_rhos


def get_model_spec_num_experts(model) -> int:
    cfg = model.config
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        if hasattr(cfg, attr):
            return int(getattr(cfg, attr))
    raise AttributeError("Could not determine num_experts from model.config")


# ---------------------------------------------------------------------------
# Real per-layer CAKE-style eviction (differs per layer, unlike the other
# baselines which share one retained-index set across all layers).
# ---------------------------------------------------------------------------

def cake_layer_budgets(attentions: Tuple[torch.Tensor, ...], total_budget: int, min_per_layer: int = 4) -> List[int]:
    """Allocates a real per-layer token budget from real attention entropy.

    Layers whose attention is more diffuse (higher entropy -> long-range,
    less locally-redundant dependencies) get a larger share of the budget;
    layers with sharply peaked attention get a smaller share, since a small
    top-k already captures most of their mass. This is the standard
    layer-adaptive CAKE intuition, computed from real attention, not copied
    from another baseline's formula.
    """
    num_layers = len(attentions)
    entropies = []
    for layer_attn in attentions:
        a = layer_attn[0].mean(dim=0)  # (T_q, T_k) averaged over heads
        a = a / a.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        ent = (-(a * a.clamp(min=1e-12).log()).sum(dim=-1)).mean().item()
        entropies.append(max(1e-6, ent))
    entropies = np.array(entropies)
    weights = entropies / entropies.sum()
    raw = weights * total_budget * num_layers
    budgets = np.maximum(min_per_layer, raw.round()).astype(int)
    return budgets.tolist()


def compute_evicted_kv(
    model: nn.Module,
    input_ids: torch.Tensor,
    budget_fraction: float,
    policy_name: str,
    expert_profiles: Dict[int, ExpertProfile],
    num_experts: int,
    device: str = "cuda",
):
    """Real prefill + real routing/attention capture + real eviction decision.

    Returns (past_key_values, was_evicted: bool). Shared by generation-based
    tasks (GSM8K/HumanEval/NIAH) and the perplexity task, so eviction logic
    lives in exactly one place.
    """
    seq_len = input_ids.shape[1]
    budget = max(4, int(budget_fraction * seq_len))

    with torch.no_grad():
        with MoERoutingHook(model) as hook:
            outputs = model(input_ids, use_cache=True, output_attentions=True)
            past_key_values = outputs.past_key_values
            try:
                sig = hook.get_signature()
            except RuntimeError:
                sig = None

    if budget >= seq_len or policy_name == "FullKV" or budget_fraction >= 0.999:
        return past_key_values, False

    attentions = outputs.attentions  # tuple over layers: (1, H, T, T)
    last_layer_attn = attentions[-1][0]
    attn_received = last_layer_attn.mean(dim=0).mean(dim=0).unsqueeze(0)  # (1, T)

    if sig is not None and expert_profiles:
        r_score = compute_routing_conditioned_score(sig, expert_profiles, num_experts=num_experts)
    else:
        r_score = torch.zeros((1, seq_len), device=device)

    per_layer_indices: Optional[List[torch.Tensor]] = None

    if policy_name == "Uniform":
        step = max(1, seq_len // budget)
        retained_indices = torch.arange(0, seq_len, step, device=device)[:budget].unsqueeze(0)
    elif policy_name == "CAKE":
        layer_budgets = cake_layer_budgets(attentions, budget)
        per_layer_indices = []
        for l_idx, layer_attn in enumerate(attentions):
            layer_a = layer_attn[0].mean(dim=0).mean(dim=0).unsqueeze(0)  # (1, T)
            b_l = layer_budgets[l_idx]
            idx_l = select_topk_indices(layer_a, budget=b_l, protect_sink_tokens=4)
            per_layer_indices.append(idx_l)
        retained_indices = per_layer_indices[-1]
    else:
        if policy_name == "H2O":
            w_a, w_r, w_s, w_c = 1.0, 0.0, 0.0, 0.0
        elif policy_name == "SnapKV":
            w_a, w_r, w_s, w_c = 1.0, 0.0, 0.05, 0.05
        elif policy_name == "R-only":
            w_a, w_r, w_s, w_c = 0.0, 1.0, 0.05, 0.05
        else:  # A+R (EKVA v2)
            w_a, w_r, w_s, w_c = 0.60, 0.30, 0.05, 0.05

        saliency = combined_token_saliency(
            attn_scores=attn_received.to(device), routing_scores=r_score.to(device),
            num_sink_tokens=4, w_a=w_a, w_r=w_r, w_s=w_s, w_c=w_c,
        )
        retained_indices = select_topk_indices(saliency, budget=budget, protect_sink_tokens=4)

    def _compact(pkv):
        if hasattr(pkv, "key_cache") and hasattr(pkv, "value_cache"):
            for l_idx in range(len(pkv.key_cache)):
                idx_l = per_layer_indices[l_idx] if per_layer_indices is not None else retained_indices
                pkv.key_cache[l_idx] = compact_kv_tensor(pkv.key_cache[l_idx], idx_l)
                pkv.value_cache[l_idx] = compact_kv_tensor(pkv.value_cache[l_idx], idx_l)
            return pkv
        else:
            out = []
            for l_idx, (k, v) in enumerate(pkv):
                idx_l = per_layer_indices[l_idx] if per_layer_indices is not None else retained_indices
                out.append((compact_kv_tensor(k, idx_l), compact_kv_tensor(v, idx_l)))
            return tuple(out)

    return _compact(past_key_values), True


def evict_and_generate(
    model: nn.Module,
    tokenizer,
    prompt: str,
    budget_fraction: float,
    policy_name: str,
    expert_profiles: Dict[int, ExpertProfile],
    num_experts: int,
    max_new_tokens: int = 128,
    device: str = "cuda",
) -> str:
    """Real prefill, real routing/attention capture, real KV compaction, real decode."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids

    compacted_pkv, _evicted = compute_evicted_kv(
        model, input_ids, budget_fraction, policy_name, expert_profiles, num_experts, device,
    )

    # Manual per-token decode loop for every policy (including FullKV), rather than
    # handing `generate()` a pre-populated cache: mixing a cache already covering
    # the whole prompt with `generate(input_ids=<full prompt>, past_key_values=...)`
    # double-counts cache positions in current transformers versions. Using the
    # same loop for every policy also keeps the comparison apples-to-apples.
    curr_token = input_ids[:, -1:]
    generated_ids = []
    for _ in range(max_new_tokens):
        with torch.no_grad():
            step_out = model(curr_token, past_key_values=compacted_pkv, use_cache=True)
            logits = step_out.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            compacted_pkv = step_out.past_key_values
            token_id = next_token.item()
            if token_id == tokenizer.eos_token_id:
                break
            generated_ids.append(token_id)
            curr_token = next_token

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Real task loaders + scorers.
# ---------------------------------------------------------------------------

import re


def extract_gsm8k_answer(text: str) -> Optional[str]:
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "").strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text)
    return numbers[-1].strip() if numbers else None


def load_gsm8k(n: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split=f"test[:{n}]")
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


def load_humaneval(n: int) -> List[Dict]:
    from datasets import load_dataset
    ds = load_dataset("openai_humaneval", split=f"test[:{n}]")
    return [dict(r) for r in ds]


def run_humaneval_test(completion_code: str, test_code: str, entry_point: str, timeout: float = 10.0) -> bool:
    """Executes generated code + the task's real unit tests in an isolated subprocess."""
    program = completion_code + "\n" + test_code + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, path], capture_output=True, timeout=timeout, text=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        os.unlink(path)


def load_pg19_or_fallback(n: int) -> Tuple[List[str], str]:
    from datasets import load_dataset
    try:
        ds = load_dataset("pg19", split="test", streaming=True)
        texts = []
        for i, r in enumerate(ds):
            if i >= n:
                break
            texts.append(r["text"][:4000])
        if texts:
            return texts, "pg19/test"
    except Exception:
        pass
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=f"test[:{n*20}]")
    texts, buf = [], ""
    for r in ds:
        buf += r["text"]
        if len(buf) > 3000:
            texts.append(buf[:4000])
            buf = ""
        if len(texts) >= n:
            break
    return texts, "wikitext-103-raw-v1/test (pg19 unavailable)"


def build_niah_samples(filler_texts: List[str], tokenizer, n: int, seq_len: int = 1500) -> List[Dict]:
    import random
    rng = random.Random(1234)
    samples = []
    for i in range(n):
        filler = filler_texts[i % len(filler_texts)]
        needle_num = rng.randint(100000, 999999)
        needle = f"The special magic number for this task is {needle_num}."
        toks = tokenizer(filler, truncation=True, max_length=seq_len).input_ids
        depth = rng.uniform(0.1, 0.9)
        cut = int(len(toks) * depth)
        prefix = tokenizer.decode(toks[:cut], skip_special_tokens=True)
        suffix = tokenizer.decode(toks[cut:], skip_special_tokens=True)
        context = f"{prefix}\n{needle}\n{suffix}"
        prompt = f"{context}\n\nQuestion: What is the special magic number mentioned in the text above? Answer with only the number.\nAnswer:"
        samples.append({"prompt": prompt, "answer": str(needle_num)})
    return samples


# ---------------------------------------------------------------------------
# Main evaluation loop.
# ---------------------------------------------------------------------------

def run_task(task: str, examples, model, tokenizer, device, profiles, num_experts, budgets, policies, max_new_tokens) -> Dict:
    """Runs one task across all budgets/policies, returns real per-example raw_scores."""
    results = {}
    for b in budgets:
        b_key = f"{int(b*100)}%"
        results[b_key] = {}
        for policy in policies:
            if b >= 0.999 and policy != "FullKV":
                continue
            if policy == "FullKV" and b < 0.999:
                continue
            raw_scores = []
            for ex in examples:
                if task == "GSM8K":
                    prompt = (
                        "Solve the following mathematical reasoning problem step by step. "
                        "Put your final numeric answer after '#### '.\n\n"
                        f"Question: {ex['question']}\n\nAnswer:"
                    )
                    gen = evict_and_generate(model, tokenizer, prompt, b, policy, profiles, num_experts, max_new_tokens, device)
                    gold = extract_gsm8k_answer(ex["answer"])
                    pred = extract_gsm8k_answer(gen)
                    raw_scores.append(1.0 if (pred == gold and gold is not None) else 0.0)
                elif task == "HumanEval":
                    prompt = ex["prompt"]
                    gen = evict_and_generate(model, tokenizer, prompt, b, policy, profiles, num_experts, max_new_tokens, device)
                    passed = run_humaneval_test(prompt + gen, ex["test"], ex["entry_point"])
                    raw_scores.append(1.0 if passed else 0.0)
                elif task == "NIAH":
                    gen = evict_and_generate(model, tokenizer, ex["prompt"], b, policy, profiles, num_experts, 16, device)
                    raw_scores.append(1.0 if ex["answer"] in gen else 0.0)
                elif task == "PG19_PPL":
                    raw_scores.append(compute_ppl_with_eviction(model, tokenizer, ex, b, policy, profiles, num_experts, device))
            mean_v, ci_l, ci_h = bootstrap_ci(raw_scores)
            results[b_key][policy] = {"mean": mean_v, "ci_95": [ci_l, ci_h], "n": len(raw_scores), "raw_scores": raw_scores}
    return results


def compute_ppl_with_eviction(model, tokenizer, text, budget_fraction, policy_name, profiles, num_experts, device) -> float:
    """Real perplexity under KV eviction.

    Evicts the prefix's KV cache under the given policy/budget, then scores the
    real cross-entropy loss of the continuation conditioned on the (possibly
    compacted) prefix cache. This makes PG19 actually sensitive to which policy
    evicted which tokens, unlike a flat full-context loss.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc.input_ids.to(device)
    if input_ids.shape[1] < 32:
        return float("nan")
    split = input_ids.shape[1] // 2
    prefix, cont = input_ids[:, :split], input_ids[:, split:]

    compacted_pkv, _ = compute_evicted_kv(
        model, prefix, budget_fraction, policy_name, profiles, num_experts, device,
    )
    with torch.no_grad():
        out = model(cont, past_key_values=compacted_pkv, labels=cont)
        return float(torch.exp(out.loss).item())


def main():
    parser = argparse.ArgumentParser(description="EKVA v2 Real Evaluation Suite")
    parser.add_argument("--model", default="qwen1.5-moe-a2.7b")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--4bit", dest="use_4bit", action="store_true", default=None)
    parser.add_argument("--gsm8k-samples", type=int, default=1319)
    parser.add_argument("--humaneval-samples", type=int, default=164)
    parser.add_argument("--pg19-docs", type=int, default=30)
    parser.add_argument("--niah-samples", type=int, default=40)
    parser.add_argument("--calib-samples", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.model} ...")
    model, tokenizer, device, quant_used = load_model_and_tokenizer(args.model, args.use_4bit)
    num_experts = get_model_spec_num_experts(model)

    print("Building real calibration profiles + real rho(R, A_hat) ...")
    calib_texts, calib_source = load_pg19_or_fallback(args.calib_samples)
    profiles, corr, r_variance = build_real_calibration(model, tokenizer, device, calib_texts)
    print(f"  real rho(R, A_hat) = {corr:.4f}   var(R) = {r_variance:.6f}   (n_texts={len(calib_texts)}, source={calib_source})")

    print("Computing real per-layer rho(R_l, A_hat_l) ...")
    layer_rhos = compute_layerwise_correlation(model, tokenizer, profiles, num_experts, device, calib_texts)
    print(f"  per-layer rho: {layer_rhos}")

    print("Loading real task datasets ...")
    gsm8k = load_gsm8k(args.gsm8k_samples)
    humaneval = load_humaneval(args.humaneval_samples)
    pg19_texts, pg19_source = load_pg19_or_fallback(args.pg19_docs)
    niah = build_niah_samples(pg19_texts, tokenizer, args.niah_samples)

    tasks_results = {}
    for task_name, examples in [("GSM8K", gsm8k), ("HumanEval", humaneval), ("PG19_PPL", pg19_texts), ("NIAH", niah)]:
        print(f"\nRunning {task_name} ({len(examples)} real examples) across {BUDGETS} x {BASELINES} ...")
        t0 = time.time()
        tasks_results[task_name] = run_task(
            task_name, examples, model, tokenizer, device, profiles, num_experts,
            BUDGETS, BASELINES, args.max_new_tokens,
        )
        print(f"  done in {time.time()-t0:.1f}s")

    # Paired significance: A+R vs every other baseline, per task per budget.
    significance = {}
    for task_name, task_res in tasks_results.items():
        significance[task_name] = {}
        for b_key, policies_res in task_res.items():
            if "A+R (EKVA v2)" not in policies_res:
                continue
            ar_raw = policies_res["A+R (EKVA v2)"]["raw_scores"]
            significance[task_name][b_key] = {}
            for other in policies_res:
                if other == "A+R (EKVA v2)":
                    continue
                significance[task_name][b_key][f"A+R vs {other}"] = paired_bootstrap_test(ar_raw, policies_res[other]["raw_scores"])

    output = {
        "model": args.model,
        "protocol": {
            "quantization": quant_used,
            "device": device,
            "gsm8k": {"source": "gsm8k/main/test", "n": len(gsm8k)},
            "humaneval": {"source": "openai_humaneval/test", "n": len(humaneval)},
            "pg19_ppl": {"source": pg19_source, "n": len(pg19_texts)},
            "niah": {"source": f"synthetic needle in {pg19_source}", "n": len(niah)},
            "calibration": {"source": calib_source, "n": len(calib_texts)},
            "generation": {"do_sample": False, "max_new_tokens": args.max_new_tokens},
            "budgets": BUDGETS,
            "baselines": BASELINES,
        },
        "correlation_rho": round(corr, 4),
        "R_variance": round(r_variance, 6),
        "layerwise_correlation_rho": layer_rhos,
        "tasks": tasks_results,
        "significance_vs_A+R": significance,
    }

    out_file = os.path.join(args.out_dir, f"real_eval_{args.model}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved real evaluation results to: {out_file}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(max(6, len(layer_rhos) * 0.35), 3.5))
        ax.bar(range(len(layer_rhos)), layer_rhos, color=["#d95f02" if v >= 0 else "#1f77b4" for v in layer_rhos])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(r"$\rho(R_l, \hat{A}_l)$")
        ax.set_title(f"Real per-layer rho(R, A_hat) — {args.model}")
        fig_path = os.path.join(args.out_dir, f"real_layerwise_rho_{args.model}.png")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=200)
        plt.close()
        print(f"Saved per-layer rho figure to: {fig_path}")
    except Exception as e:
        print(f"(skipped per-layer rho figure: {e})")


if __name__ == "__main__":
    main()
