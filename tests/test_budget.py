"""Unit tests for budget allocation policies (the full policy matrix)."""
import torch

from ekva.budget.policies import (
    UniformPolicy, RandomPolicy, EKVAPolicy, EKVAMultiSignalPolicy,
    EKVAEntropyOnlyPolicy, EKVARoutingOnlyPolicy, EKVASpecializationOnlyPolicy,
    CakeLayerAggregatedPolicy, SnapKVStylePolicy, PyramidKVStylePolicy, DynamicKVStylePolicy, get_policy,
)

EMAP = {
    i: {"avg_entropy": torch.full((4,), 1.0 + 0.1 * i),
        "routing_count": torch.tensor(100 + 10 * i)}
    for i in range(8)
}


def test_uniform_policy():
    b = UniformPolicy().allocate(num_experts=8, total_budget=2048, min_per_expert=64)
    assert len(b) == 8 and all(v >= 64 for v in b.values())


def test_random_policy_sum_and_min():
    b = RandomPolicy().allocate(num_experts=8, total_budget=2048, min_per_expert=64)
    assert sum(b.values()) == 2048 and all(v >= 64 for v in b.values())


def test_ekva_policy_requires_entropy_map():
    import pytest
    with pytest.raises(ValueError):
        EKVAPolicy().allocate(num_experts=8, total_budget=2048, entropy_map=None)


def test_ekva_policy_with_entropy_map():
    b = EKVAPolicy().allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, min_per_expert=64)
    assert sum(b.values()) == 2048 and all(v >= 64 for v in b.values())


def test_multisignal_policy():
    tok_types = {e: torch.randint(0, 8, (100,)) for e in range(8)}
    from ekva.calibration.signals import specialization_score
    spec = specialization_score(tok_types, 8)
    b = EKVAMultiSignalPolicy().allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, specialization=spec)
    assert sum(b.values()) == 2048 and all(v >= 64 for v in b.values())


def test_pyramidkv_policy_sums():
    b = PyramidKVStylePolicy().allocate(num_experts=8, total_budget=2048, entropy_map=EMAP)
    assert sum(b.values()) == 2048 and all(v >= 64 for v in b.values())


def test_snkvk_dynamickv_policies():
    for P in (SnapKVStylePolicy, DynamicKVStylePolicy):
        b = P().allocate(num_experts=8, total_budget=2048)
        assert sum(b.values()) == 2048


def test_policy_registry_get():
    assert isinstance(get_policy("ekva"), EKVAPolicy)
    assert isinstance(get_policy("uniform"), UniformPolicy)
    assert isinstance(get_policy("cake_layer_aggregated"), CakeLayerAggregatedPolicy)
    assert isinstance(get_policy("ekva_entropy_only"), EKVAEntropyOnlyPolicy)
    assert isinstance(get_policy("ekva_routing_only"), EKVARoutingOnlyPolicy)
    assert isinstance(get_policy("ekva_specialization_only"), EKVASpecializationOnlyPolicy)


def test_cake_layer_aggregated_policy():
    b = get_policy("cake_layer_aggregated").allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, min_per_expert=64)
    assert sum(b.values()) == 2048 and all(v >= 64 for v in b.values())


def test_ablation_policies():
    tok_types = {e: torch.randint(0, 8, (100,)) for e in range(8)}
    from ekva.calibration.signals import specialization_score
    spec = specialization_score(tok_types, 8)

    b_ent = get_policy("ekva_entropy_only").allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, min_per_expert=64)
    assert sum(b_ent.values()) == 2048 and all(v >= 64 for v in b_ent.values())

    b_route = get_policy("ekva_routing_only").allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, min_per_expert=64)
    assert sum(b_route.values()) == 2048 and all(v >= 64 for v in b_route.values())

    b_spec = get_policy("ekva_specialization_only").allocate(num_experts=8, total_budget=2048, entropy_map=EMAP, specialization=spec, min_per_expert=64)
    assert sum(b_spec.values()) == 2048 and all(v >= 64 for v in b_spec.values())

