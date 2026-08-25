"""Correctness tests for the Triton kernel (Weeks 10-11).

Runs on machines with `triton` + CUDA (Colab A100). Excluded from collection
elsewhere via conftest.py on CPU-only boxes.
"""
import torch

from ekva.kernel.reference_flashattn2 import flash_attention_2_reference
from ekva.kernel.ekva_triton_v1 import ekva_attention_v1


def _oracle(q, k, v, scale):
    s = torch.einsum("bhqd,bhkd->bhqk", q * scale, k)
    p = torch.softmax(s, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v)


def test_kernel_v1_matches_reference_at_full_budget():
    B, H, N, D = 1, 4, 256, 64
    q = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    scale = D ** -0.5
    ref = _oracle(q, k, v, scale)
    out = ekva_attention_v1(q, k, v, kv_budget=N, scale=scale)
    assert (out - ref).abs().max().item() < 1e-2


def test_triton_compact_kv_cache_matches_pytorch():
    from ekva.kernel.ekva_eviction_v2 import triton_compact_kv_cache
    from ekva.retention.eviction import compact_kv_tensor

    B, H, T, D = 2, 4, 256, 64
    budget = 128
    k = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, T, D, device="cuda", dtype=torch.float16)

    # Random unique indices per batch/head
    idx_list = []
    for _ in range(B * H):
        perm = torch.randperm(T, device="cuda")[:budget]
        sorted_perm, _ = torch.sort(perm)
        idx_list.append(sorted_perm)
    indices = torch.stack(idx_list, dim=0).view(B, H, budget)

    ref_k = compact_kv_tensor(k, indices)
    ref_v = compact_kv_tensor(v, indices)

    tri_k, tri_v = triton_compact_kv_cache(k, v, indices)

    assert (tri_k - ref_k).abs().max().item() == 0.0
    assert (tri_v - ref_v).abs().max().item() == 0.0

