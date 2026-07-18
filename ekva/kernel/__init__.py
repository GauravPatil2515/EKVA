"""Triton FlashAttention-2 kernel variants for EKVA (Weeks 10-11).

  reference_flashattn2.py — standard FA2 tiling (baseline, variable-free)
  ekva_triton_v1.py        — variable KV tile count per expert (Week 10)
  ekva_triton_v2.py        — v1 + fused important-token index selection (Week 11)

These require `triton` and a CUDA GPU; they are NOT imported by the core
pipeline and are exercised only in experiments/week10_kernel_v1.py and
experiments/week11_kernel_v2.py.
"""
from ekva.kernel.reference_flashattn2 import flash_attention_2_reference
from ekva.kernel.ekva_triton_v1 import ekva_attention_v1
from ekva.kernel.ekva_triton_v2 import ekva_attention_v2

__all__ = [
    "flash_attention_2_reference",
    "ekva_attention_v1",
    "ekva_attention_v2",
]
