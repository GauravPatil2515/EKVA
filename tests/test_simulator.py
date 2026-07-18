"""Unit tests for the Phase 2 software simulator (buffer + eviction + hook API)."""
import torch

from ekva.simulator.kv_buffer import ExpertKVBuffer
from ekva.simulator.eviction import EVICTION_REGISTRY, get_eviction, validate_eviction
from ekva.simulator.hook import EKVACacheHook


def test_kv_buffer_fill_below_capacity():
    buf = ExpertKVBuffer(budget=8, head_dim=4, num_heads=2, eviction="recency", dtype=torch.float32)
    k = torch.randn(4, 2, 4)
    v = torch.randn(4, 2, 4)
    buf.update(k, v)
    assert buf.size == 4
    ko, vo = buf.get()
    assert ko.shape == (4, 2, 4)


def test_kv_buffer_eviction_recency():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="recency", dtype=torch.float32)
    for i in range(6):
        buf.update(torch.full((1, 1, 2), float(i)), torch.full((1, 1, 2), float(i)))
    assert buf.size == 4


def test_kv_buffer_eviction_random():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="random", dtype=torch.float32)
    buf.update(torch.randn(6, 1, 2), torch.randn(6, 1, 2))
    assert buf.size == 4


def test_kv_buffer_eviction_hybrid():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=2, eviction="hybrid", dtype=torch.float32)
    for i in range(6):
        buf.update(torch.randn(1, 2, 2), torch.randn(1, 2, 2), attn_weights=torch.rand(1, 2, max(min(i, 4), 1)))
    assert buf.size == 4


def test_kv_buffer_reset():
    buf = ExpertKVBuffer(budget=4, head_dim=2, num_heads=1, eviction="recency", dtype=torch.float32)
    buf.update(torch.randn(4, 1, 2), torch.randn(4, 1, 2))
    buf.reset()
    assert buf.size == 0 and buf.get() == (None, None)


def test_eviction_registry():
    assert set(EVICTION_REGISTRY) == {"recency", "attention", "random", "hybrid"}
    assert get_eviction("attention") == "attention"
    assert validate_eviction("nope") is False


def test_hook_api_no_model():
    # EKVACacheHook needs a nn.Module; just check it constructs buffers for a stub.
    class Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
    hook = EKVACacheHook(Stub(), {0: 64, 1: 128}, num_experts=2, eviction="attention")
    assert set(hook.buffers) == {0, 1}
    assert hook.buffers[0].budget == 64 and hook.buffers[1].budget == 128
