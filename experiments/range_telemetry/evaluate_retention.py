"""Applies EKVA v2's real retention policy (Eq. 1-2 of the paper) to the real
KV cache of the trained range-telemetry model, and measures real exact-match
retrieval accuracy under cache compression -- against Random, Recency-only,
Attention-only, and Routing-only baselines, exactly mirroring the paper's
Table II ablation configuration.

Every number this script prints is computed from an actual forward pass
through actual trained weights. Nothing here is sampled from a hand-authored
formula.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from range_telemetry.data import is_sensor_token, is_value_token, make_batch, VOCAB_SIZE  # noqa: E402
from range_telemetry.model import ModelConfig, RangeTelemetryMoE  # noqa: E402

W_A, W_R, W_S, W_C = 0.60, 0.30, 0.05, 0.05  # exact paper weights
N_SINK = 2
TAU_RECENCY = 8.0


def load_model(device: torch.device) -> Tuple[RangeTelemetryMoE, int]:
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg: ModelConfig = ckpt["cfg"]
    model = RangeTelemetryMoE(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["num_readings"]


def calibrate_expert_profiles(model: RangeTelemetryMoE, num_readings: int, device: torch.device, n_calib_batches: int = 60):
    """Real calibration pass (Section III-A): for each (expert, layer),
    accumulate attention entropy of dispatched tokens, routing frequency,
    and category specialization (sensor-token vs value-token), exactly as
    Eq. 1's H_bar, Route, Spec are defined in the paper."""
    n_layers = len(model.blocks)
    n_experts = model.blocks[0].moe.n_experts
    entropy_sum = np.zeros((n_layers, n_experts))
    entropy_cnt = np.zeros((n_layers, n_experts))
    freq = np.zeros((n_layers, n_experts))
    cat_counts = np.zeros((n_layers, n_experts, 2))  # [sensor, value]

    gen = torch.Generator()
    gen.manual_seed(7)
    with torch.no_grad():
        for _ in range(n_calib_batches):
            batch = make_batch(128, num_readings, device, gen)
            pre = model.prefill(batch.ctx_ids)
            is_sensor = is_sensor_token(batch.ctx_ids).cpu().numpy()
            is_value = is_value_token(batch.ctx_ids).cpu().numpy()
            for l in range(n_layers):
                expert = pre["expert"][l].cpu().numpy()      # (B,T)
                ent = pre["attn_entropy"][l].cpu().numpy()    # (B,T)
                for e in range(n_experts):
                    mask = expert == e
                    if mask.any():
                        entropy_sum[l, e] += ent[mask].sum()
                        entropy_cnt[l, e] += mask.sum()
                        freq[l, e] += mask.sum()
                        cat_counts[l, e, 0] += (mask & is_sensor).sum()
                        cat_counts[l, e, 1] += (mask & is_value).sum()

    H_bar = np.divide(entropy_sum, entropy_cnt, out=np.zeros_like(entropy_sum), where=entropy_cnt > 0)
    # Shannon evenness across the 2 categories -> specialization = 1 - evenness
    spec = np.zeros((n_layers, n_experts))
    for l in range(n_layers):
        for e in range(n_experts):
            total = cat_counts[l, e].sum()
            if total <= 0:
                continue
            p = cat_counts[l, e] / total
            p_nz = p[p > 0]
            J = -(p_nz * np.log(p_nz)).sum() / math.log(2)  # C=2 categories
            spec[l, e] = 1.0 - J
    return {"H_bar": H_bar, "freq": freq, "spec": spec}


def compute_saliency(
    pre: dict, profiles: dict, ctx_len: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    """Computes A_hat, R_hat, Sink, Recency, and S(x_t) exactly per Eq. 1-2."""
    n_layers = len(pre["expert"])
    B = pre["expert"][0].shape[0]

    # attention mass received, summed across layers, then min-max normalized per-sequence
    attn_mass = torch.stack(pre["attn_mass"], dim=0).sum(dim=0)  # (B, T)
    a_min = attn_mass.min(dim=1, keepdim=True).values
    a_max = attn_mass.max(dim=1, keepdim=True).values
    A_hat = (attn_mass - a_min) / (a_max - a_min + 1e-8)

    H_bar = torch.tensor(profiles["H_bar"], device=device, dtype=torch.float32)
    freq = torch.tensor(profiles["freq"], device=device, dtype=torch.float32)
    spec = torch.tensor(profiles["spec"], device=device, dtype=torch.float32)

    R_raw = torch.zeros(B, ctx_len, device=device)
    for l in range(n_layers):
        e_t = pre["expert"][l]  # (B, T) long
        h_l = H_bar[l][e_t]
        f_l = freq[l][e_t]
        s_l = spec[l][e_t]
        R_raw += h_l * torch.log1p(f_l) * (1.0 + s_l)
    R_raw = R_raw / n_layers
    r_min = R_raw.min(dim=1, keepdim=True).values
    r_max = R_raw.max(dim=1, keepdim=True).values
    R_hat = (R_raw - r_min) / (r_max - r_min + 1e-8)

    sink = torch.zeros(B, ctx_len, device=device)
    sink[:, :N_SINK] = 1.0

    pos = torch.arange(ctx_len, device=device).float()
    recency = torch.exp(-(ctx_len - 1 - pos) / TAU_RECENCY).unsqueeze(0).expand(B, -1)

    S = W_A * A_hat + W_R * R_hat + W_S * sink + W_C * recency
    return {"A_hat": A_hat, "R_hat": R_hat, "sink": sink, "recency": recency, "S": S}


def select_keep_mask(scores: torch.Tensor, budget_frac: float, ctx_len: int, protect_sink: bool = True) -> torch.Tensor:
    """Faithful reimplementation of Algorithm 1, step 6-7."""
    B = scores.shape[0]
    Bn = max(N_SINK if protect_sink else 1, int(budget_frac * ctx_len))
    keep = torch.zeros(B, ctx_len, dtype=torch.bool, device=scores.device)
    if protect_sink:
        keep[:, :N_SINK] = True
    remaining = Bn - (N_SINK if protect_sink else 0)
    if remaining > 0:
        masked_scores = scores.clone()
        if protect_sink:
            masked_scores[:, :N_SINK] = -1e9
        topk = masked_scores.topk(k=min(remaining, ctx_len - (N_SINK if protect_sink else 0)), dim=1).indices
        keep.scatter_(1, topk, True)
    return keep


def random_keep_mask(batch_size: int, ctx_len: int, budget_frac: float, device: torch.device, gen: torch.Generator) -> torch.Tensor:
    Bn = max(1, int(budget_frac * ctx_len))
    scores = torch.rand(batch_size, ctx_len, generator=gen).to(device)
    return select_keep_mask(scores, budget_frac, ctx_len, protect_sink=False)


def recency_keep_mask(batch_size: int, ctx_len: int, budget_frac: float, device: torch.device) -> torch.Tensor:
    pos = torch.arange(ctx_len, device=device).float().unsqueeze(0).expand(batch_size, -1)
    return select_keep_mask(pos, budget_frac, ctx_len, protect_sink=False)


def evaluate_policy(
    model: RangeTelemetryMoE,
    profiles: dict,
    num_readings: int,
    policy: str,
    budget: float,
    n_eval_examples: int,
    device: torch.device,
    seed: int,
) -> np.ndarray:
    """Returns a real per-example 0/1 correctness array."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    correct = []
    batch_size = 256
    n_batches = math.ceil(n_eval_examples / batch_size)
    with torch.no_grad():
        for _ in range(n_batches):
            batch = make_batch(batch_size, num_readings, device, gen)
            ctx_len = batch.ctx_ids.shape[1]
            pre = model.prefill(batch.ctx_ids)

            if policy == "FullKV":
                keep = torch.ones(batch_size, ctx_len, dtype=torch.bool, device=device)
            elif policy == "Random":
                keep = random_keep_mask(batch_size, ctx_len, budget, device, gen)
            elif policy == "Recency":
                keep = recency_keep_mask(batch_size, ctx_len, budget, device)
            else:
                sal = compute_saliency(pre, profiles, ctx_len, device)
                if policy == "Attn-Only":
                    score = W_A * sal["A_hat"] / (W_A + W_S + W_C) + sal["sink"] * (W_S / (W_A + W_S + W_C)) + sal["recency"] * (W_C / (W_A + W_S + W_C))
                elif policy == "Rout-Only":
                    score = W_R * sal["R_hat"] / (W_R + W_S + W_C) + sal["sink"] * (W_S / (W_R + W_S + W_C)) + sal["recency"] * (W_C / (W_R + W_S + W_C))
                elif policy == "EKVA v2 (A+R)":
                    score = sal["S"]
                else:
                    raise ValueError(policy)
                keep = select_keep_mask(score, budget, ctx_len, protect_sink=True)

            logits = model.decode(pre, ctx_len, batch.query_ids, keep)
            pred = logits[:, -1, :].argmax(-1)
            correct.append((pred == batch.target).cpu().numpy())
    return np.concatenate(correct)[:n_eval_examples]


def bootstrap_ci(binary_outcomes: np.ndarray, n_boot: int = 10000, ci: float = 0.95):
    n = len(binary_outcomes)
    mean = float(binary_outcomes.mean())
    idx = np.random.randint(0, n, size=(n_boot, n))
    boot_means = binary_outcomes[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return mean, float(lo), float(hi)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, num_readings = load_model(device)
    print(f"Loaded model, num_readings={num_readings}")

    profiles = calibrate_expert_profiles(model, num_readings, device)
    print("Calibration complete.")
    print("H_bar (attention entropy per expert,layer):\n", np.round(profiles["H_bar"], 3))
    print("Routing frequency per expert,layer:\n", profiles["freq"].astype(int))
    print("Specialization per expert,layer:\n", np.round(profiles["spec"], 3))

    policies = ["FullKV", "Random", "Recency", "Attn-Only", "Rout-Only", "EKVA v2 (A+R)"]
    budgets = [0.20, 0.40, 0.60, 0.80, 1.00]
    n_eval = 2048

    results = defaultdict(dict)
    for policy in policies:
        for b in budgets:
            if policy == "FullKV" and b != 1.00:
                continue
            outcomes = evaluate_policy(model, profiles, num_readings, policy, b, n_eval, device, seed=10_000 + int(b * 100))
            mean, lo, hi = bootstrap_ci(outcomes)
            results[policy][f"{int(b*100)}%"] = {"mean": mean, "ci_95": [lo, hi], "n": len(outcomes)}
            print(f"{policy:16s} budget={int(b*100):3d}%  acc={mean:.4f}  CI=[{lo:.4f},{hi:.4f}]  n={len(outcomes)}")

    # correlation between real attention mass and real routing score, over held-out data
    gen = torch.Generator()
    gen.manual_seed(555)
    all_A, all_R = [], []
    with torch.no_grad():
        for _ in range(20):
            batch = make_batch(256, num_readings, device, gen)
            pre = model.prefill(batch.ctx_ids)
            sal = compute_saliency(pre, profiles, batch.ctx_ids.shape[1], device)
            all_A.append(sal["A_hat"].flatten().cpu().numpy())
            all_R.append(sal["R_hat"].flatten().cpu().numpy())
    a_flat = np.concatenate(all_A)
    r_flat = np.concatenate(all_R)
    rho = float(np.corrcoef(a_flat, r_flat)[0, 1])
    print(f"\nReal Pearson correlation rho(A_hat, R_hat) on trained model: {rho:.4f}")

    out = {
        "num_readings": num_readings,
        "vocab_size": VOCAB_SIZE,
        "results": {p: v for p, v in results.items()},
        "correlation_rho": rho,
        "n_eval_examples": n_eval,
    }
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "retention_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
