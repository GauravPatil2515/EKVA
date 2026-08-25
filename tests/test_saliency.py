"""Unit tests for EKVA v2 Saliency and Eviction Engine."""
import pytest
import torch
import torch.nn as nn
from ekva.retention.routing_signature import RoutingSignature, MoERoutingHook
from ekva.retention.saliency import (
    ExpertProfile,
    compute_routing_conditioned_score,
    compute_recency_score,
    compute_sink_score,
    combined_token_saliency,
)
from ekva.retention.eviction import (
    select_topk_indices,
    compact_kv_tensor,
    evict_shared_kv_cache,
)


def test_routing_score_nonnegative_and_bounded():
    B, T, L, K = 2, 128, 16, 4
    expert_indices = torch.randint(0, 8, (B, T, L, K))
    sig = RoutingSignature(expert_indices=expert_indices)

    profiles = {
        e: ExpertProfile(entropy=float(e + 1), routing_freq=float(e * 100), specialization=float(e) / 8.0)
        for e in range(8)
    }

    r = compute_routing_conditioned_score(sig, profiles, num_experts=8)
    assert r.shape == (B, T)
    assert (r >= 0.0).all()
    assert (r <= 1.0 + 1e-6).all()


def test_combined_saliency_preserves_sink_tokens():
    B, T = 2, 64
    attn = torch.rand(B, T)
    routing = torch.rand(B, T)

    saliency = combined_token_saliency(
        attn_scores=attn,
        routing_scores=routing,
        num_sink_tokens=4,
        w_a=0.6,
        w_r=0.3,
        w_s=0.05,
        w_c=0.05,
    )

    # Sink tokens must be infinite / maximum
    assert (saliency[:, :4] == float("inf")).all()
    assert (saliency[:, 4:] < float("inf")).all()


def test_eviction_produces_exact_budget_shape_and_order():
    B, H, T, D = 2, 4, 128, 64
    k = torch.randn(B, H, T, D)
    v = torch.randn(B, H, T, D)
    saliency = torch.rand(B, T)
    saliency[:, :4] = float("inf") # sink

    budget = 32
    comp_k, comp_v, indices = evict_shared_kv_cache(
        key_cache=k,
        value_cache=v,
        saliency_scores=saliency,
        budget=budget,
        protect_sink_tokens=4,
    )

    assert comp_k.shape == (B, H, budget, D)
    assert comp_v.shape == (B, H, budget, D)
    assert indices.shape == (B, budget)

    # Indices must be strictly ascending (chronological order)
    for b in range(B):
        diffs = indices[b, 1:] - indices[b, :-1]
        assert (diffs > 0).all()
        # First 4 tokens must be 0, 1, 2, 3
        assert list(indices[b, :4].tolist()) == [0, 1, 2, 3]


def test_routing_hook_with_dummy_model():
    class DummyRouter(nn.Module):
        def forward(self, x):
            # Return logits for 8 experts
            B, T, _ = x.shape
            return torch.randn(B, T, 8)

    class DummyMoELayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate = DummyRouter()

        def forward(self, x):
            logits = self.gate(x)
            return logits

    class DummyMoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([DummyMoELayer() for _ in range(4)])

        def forward(self, x):
            for l in self.layers:
                x = l(x)
            return x

    model = DummyMoEModel()
    dummy_input = torch.randn(2, 32, 16)

    with MoERoutingHook(model, model_family="generic_moe") as hook:
        _ = model(dummy_input)
        sig = hook.get_signature()

    assert sig.expert_indices.shape == (2, 32, 4, 4) # 4 layers, top-4
