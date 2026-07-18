"""Evaluation harness for the EKVA simulator (Weeks 4-6, 12).

Provides:
  compute_perplexity        — average PPL over prompts (needs transformers)
  run_policy_eviction_grid  — all (policy x eviction x budget-fraction) cells
  format_results_table      — human-readable table for the terminal / logs

The grid function is model-agnostic: it takes a `score_fn` callback that, given
(policy_name, eviction, budget_fraction, budgets), returns a quality metric
(e.g. PPL or benchmark EM/F1). This keeps the harness usable for both the
software simulator and the real hook (Week 4+).
"""
import math
from typing import Callable, Dict, List, Optional

import torch

from ekva.budget.policies import POLICY_REGISTRY, get_policy
from ekva.simulator.eviction import EVICTION_REGISTRY


def compute_perplexity(model, tokenizer, prompts: List[str], device) -> float:
    """Average perplexity over prompts using the model's default (full) KV."""
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            outputs = model(**inputs, labels=input_ids)
            loss = outputs.loss
            n = input_ids.shape[1]
            total_nll += loss.item() * n
            total_tokens += n
    return math.exp(total_nll / max(total_tokens, 1))


def run_policy_eviction_grid(
    num_experts: int,
    total_budget: int,
    policy_names: Optional[List[str]] = None,
    eviction_names: Optional[List[str]] = None,
    budget_fractions: Optional[List[float]] = None,
    entropy_map: Optional[Dict] = None,
    score_fn: Optional[Callable] = None,
) -> Dict[str, Dict]:
    """Sweep the (policy x eviction x budget_fraction) experiment matrix.

    Args:
        score_fn: callable(policy_name, eviction, fraction, budgets_dict) -> float
            Quality metric (lower PPL = better, or higher EM/F1 = better depending
            on sign convention in the caller). If None, only the budgets are
            returned (dry run, useful for sanity checks without a model).

    Returns:
        Nested dict keyed by "policy|eviction|frac" -> {budgets, metric?}.
    """
    policy_names = policy_names or list(POLICY_REGISTRY.keys())
    eviction_names = eviction_names or list(EVICTION_REGISTRY.keys())
    budget_fractions = budget_fractions or [1.0]

    results: Dict[str, Dict] = {}
    for pname in policy_names:
        policy = get_policy(pname)
        for evict in eviction_names:
            for frac in budget_fractions:
                frac_budget = int(total_budget * frac)
                try:
                    budgets = policy.allocate(
                        num_experts=num_experts, total_budget=frac_budget,
                        entropy_map=entropy_map,
                    )
                except Exception as e:  # policy needs entropy_map it doesn't have
                    budgets = {i: max(64, frac_budget // num_experts) for i in range(num_experts)}
                    policy_err = str(e)
                else:
                    policy_err = None
                key = f"{pname}|{evict}|{int(frac * 100)}%"
                entry = {"budgets": budgets, "policy_error": policy_err}
                if score_fn is not None:
                    entry["metric"] = score_fn(pname, evict, frac, budgets)
                results[key] = entry
    return results


def format_results_table(results: Dict[str, Dict], metric_key: str = "metric") -> str:
    header = f"{'Policy|Eviction|Frac':<32} | {'Sum Budget':>10} | {'Metric':>10}"
    sep = "-" * len(header)
    lines = [sep, header, sep]
    for key, r in sorted(results.items()):
        s = sum(r["budgets"].values())
        metric = r.get(metric_key, float("nan"))
        lines.append(f"{key:<32} | {s:>10} | {metric:>10.4f}")
    lines.append(sep)
    return "\n".join(lines)
