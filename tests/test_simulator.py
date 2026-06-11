"""Unit tests for Phase 2 simulator: ExpertKVBuffer and budget allocation policies."""
import torch
import pytest

from ekva.simulator.kv_buffer import ExpertKVBuffer
from ekva.simulator.policies import UniformPolicy, RandomPolicy, EKVAPolicy


# ── ExpertKVBuffer Tests ─────────────────────────────────────────────────────

def test_kv_buffer_fill_below_capacity():
    buf = ExpertKVBuffer(budget=8, head_dim=4, num_heads=2, eviction="recency", dtype=torch.float32)
    k = torch.randn(4, 2, 4)  # 4 tokens
    v = torch.randn(4, 2, 4)
    buf.update(k, v)
    assert buf.size == 4
    k_out, v_out = buf.get()
    assert k_out.shape == (4, 2, 4)


def test_kv_buffer_eviction_recency():
    """Buffer with budget=4 filled with 6 tokens should keep 4 newest."""
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="recency", dtype=torch.float32)
    for i in range(6):
        k = torch.full((1, 1, 2), float(i))
        v = torch.full((1, 1, 2), float(i))
        buf.update(k, v)
    assert buf.size == 4


def test_kv_buffer_eviction_random():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="random", dtype=torch.float32)
    k = torch.randn(6, 1, 2)
    v = torch.randn(6, 1, 2)
    buf.update(k, v)
    assert buf.size == 4


def test_kv_buffer_reset():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="recency", dtype=torch.float32)
    k = torch.randn(4, 1, 2)
    v = torch.randn(4, 1, 2)
    buf.update(k, v)
    assert buf.size == 4
    buf.reset()
    assert buf.size == 0
    k_out, v_out = buf.get()
    assert k_out is None


# ── Policy Tests ─────────────────────────────────────────────────────────────

def test_uniform_policy():
    policy = UniformPolicy()
    budgets = policy.allocate(num_experts=8, total_budget=2048, min_per_expert=64)
    assert len(budgets) == 8
    for eid, b in budgets.items():
        assert b >= 64


def test_random_policy_sum_and_min():
    policy = RandomPolicy()
    budgets = policy.allocate(num_experts=8, total_budget=2048, min_per_expert=64)
    assert sum(budgets.values()) == 2048
    assert all(b >= 64 for b in budgets.values())


def test_ekva_policy_requires_entropy_map():
    policy = EKVAPolicy()
    with pytest.raises(ValueError):
        policy.allocate(num_experts=8, total_budget=2048, entropy_map=None)


def test_ekva_policy_with_entropy_map():
    entropy_map = {
        i: {
            "avg_entropy": torch.full((4,), 1.0 + 0.1 * i),
            "routing_count": torch.tensor(100 + 10 * i),
        }
        for i in range(8)
    }
    policy = EKVAPolicy()
    budgets = policy.allocate(
        num_experts=8,
        total_budget=2048,
        entropy_map=entropy_map,
        min_per_expert=64,
    )
    assert sum(budgets.values()) == 2048
    assert all(b >= 64 for b in budgets.values())


def test_kv_buffer_eviction_attention():
    """Attention-based eviction must keep buffer size <= budget without crashing."""
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=2, eviction="attention", dtype=torch.float32)
    for i in range(6):
        k = torch.randn(1, 2, 2)
        v = torch.randn(1, 2, 2)
        cur_size = max(min(i, 4), 1)
        attn_w = torch.rand(1, 2, cur_size)
        buf.update(k, v, attn_weights=attn_w)
    assert buf.size == 4


def test_random_policy_tight_budget():
    """RandomPolicy must terminate when min_per_expert × num_experts == total_budget."""
    policy = RandomPolicy()
    # 8 × 256 = 2048: correction loop has zero slack, must still terminate
    budgets = policy.allocate(num_experts=8, total_budget=2048, min_per_expert=256)
    assert all(b >= 256 for b in budgets.values())
