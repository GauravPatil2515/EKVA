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
