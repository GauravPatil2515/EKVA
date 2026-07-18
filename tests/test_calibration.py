"""Unit tests for calibration + budget derivation (Phase 1)."""
import math
from typing import Dict

import torch

from ekva.calibration.entropy import ExpertStats, calibrate_expert_entropy
from ekva.budget.derive import derive_kv_budget
from ekva.calibration.signals import specialization_score


def test_expert_stats_entropy_and_finalize():
    num_layers = 3
    stats = ExpertStats(num_layers=num_layers)
    attn_probs = torch.full((1, 2, 2, 4), 0.25)
    stats.update_entropy(layer_idx=0, attn_probs=attn_probs)
    stats.update_entropy(layer_idx=1, attn_probs=attn_probs)
    out = stats.finalize()
    avg = out["avg_entropy"]
    expected = math.log(4.0)
    assert avg.shape[0] == num_layers
    assert torch.isclose(avg[0], torch.tensor(expected, dtype=torch.float64), atol=1e-5)
    assert torch.isclose(avg[1], torch.tensor(expected, dtype=torch.float64), atol=1e-5)


def _fake_entropy_map(num_experts=8, total_budget=2048, min_per_expert=64):
    emap: Dict[int, Dict[str, torch.Tensor]] = {}
    for eid in range(num_experts):
        emap[eid] = {
            "avg_entropy": torch.full((4,), 1.0 + 0.1 * eid),
            "routing_count": torch.tensor(100 + 10 * eid),
        }
    return emap


def test_derive_kv_budget_proportional():
    emap = _fake_entropy_map()
    budgets = derive_kv_budget(emap, total_budget=2048, strategy="proportional")
    assert budgets.shape[0] == 8
    assert budgets.sum().item() == 2048
    assert torch.all(budgets >= 64)


def test_derive_kv_budget_multisignal():
    emap = _fake_entropy_map()
    token_types = {e: torch.randint(0, 8, (int(emap[e]["routing_count"].item()) + 1,)) for e in emap}
    spec = specialization_score(token_types, 8)
    budgets = derive_kv_budget(emap, total_budget=2048, strategy="multi_signal", specialization=spec)
    assert budgets.sum().item() == 2048
    assert torch.all(budgets >= 64)


def test_derive_kv_budget_min_budget_guard():
    emap = _fake_entropy_map(num_experts=8)
    try:
        derive_kv_budget(emap, total_budget=100, min_per_expert=64)
        assert False, "should raise"
    except ValueError:
        pass


def test_calibrate_expert_entropy_mock():
    # Lightweight in-process mock model (no transformers needed).
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))
    from generate_mock_calibration import _MockModel, _Tok
    model = _MockModel(layers=4, n_experts=8)
    prompts = ["Explain attention.", "Summarize LLMs."]
    emap = calibrate_expert_entropy(model=model, tokenizer=_Tok(), calibration_prompts=prompts, num_experts=8)
    assert set(emap.keys()) == set(range(8))
    assert emap[0]["avg_entropy"].shape[0] == 4
