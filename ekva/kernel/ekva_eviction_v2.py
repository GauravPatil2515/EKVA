"""EKVA v2: Fused Saliency Selection and Shared KV Cache Compaction Kernel.

Performs top-B saliency index selection and compacts (K, V) tensors into
contiguous memory buffers for FlashAttention-2 decode acceleration.
"""
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _gather_compact_kernel(
    K_src, V_src,
    K_dst, V_dst,
    Indices,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_okb, stride_okh, stride_okbgt, stride_okd,
    stride_ovb, stride_ovh, stride_ovbgt, stride_ovd,
    stride_ib, stride_ih, stride_ibgt,
    BUDGET: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_BGT: tl.constexpr,
):
    """Gathers Key and Value vectors based on retained top-B indices."""
    pid_bgt = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decode batch and head from pid_bh
    # Assuming standard contiguous layout
    stride_bh = stride_kb // stride_kh if stride_kb > 0 and stride_kh > 0 else 1
    off_h = pid_bh % stride_bh if stride_bh > 0 else pid_bh
    off_b = pid_bh // stride_bh if stride_bh > 0 else 0

    offs_bgt = pid_bgt * BLOCK_BGT + tl.arange(0, BLOCK_BGT)
    offs_d = tl.arange(0, HEAD_DIM)

    # Mask for budget bounds
    bgt_mask = offs_bgt < BUDGET

    # Load indices
    idx_ptr = Indices + off_b * stride_ib + off_h * stride_ih + offs_bgt * stride_ibgt
    src_t_indices = tl.load(idx_ptr, mask=bgt_mask, other=0)

    # Compute source offsets: (BLOCK_BGT, HEAD_DIM)
    src_k_off = (
        off_b * stride_kb
        + off_h * stride_kh
        + src_t_indices[:, None] * stride_kt
        + offs_d[None, :] * stride_kd
    )
    src_v_off = (
        off_b * stride_vb
        + off_h * stride_vh
        + src_t_indices[:, None] * stride_vt
        + offs_d[None, :] * stride_vd
    )

    # Compute destination offsets
    dst_k_off = (
        off_b * stride_okb
        + off_h * stride_okh
        + offs_bgt[:, None] * stride_okbgt
        + offs_d[None, :] * stride_okd
    )
    dst_v_off = (
        off_b * stride_ovb
        + off_h * stride_ovh
        + offs_bgt[:, None] * stride_ovbgt
        + offs_d[None, :] * stride_ovd
    )

    # Load from source
    mask_2d = bgt_mask[:, None]
    k_vals = tl.load(K_src + src_k_off, mask=mask_2d, other=0.0)
    v_vals = tl.load(V_src + src_v_off, mask=mask_2d, other=0.0)

    # Store to destination
    tl.store(K_dst + dst_k_off, k_vals, mask=mask_2d)
    tl.store(V_dst + dst_v_off, v_vals, mask=mask_2d)


def triton_compact_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    retained_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compacts (K, V) cache tensors using Triton kernel.

    Args:
        key_cache: (B, H, T, D) on CUDA.
        value_cache: (B, H, T, D) on CUDA.
        retained_indices: (B, H, budget) or (B, budget) sorted LongTensor on CUDA.

    Returns:
        Tuple of (compact_k, compact_v) each of shape (B, H, budget, D).
    """
    if not key_cache.is_cuda:
        # Fallback to PyTorch on CPU
        from ekva.retention.eviction import compact_kv_tensor
        return compact_kv_tensor(key_cache, retained_indices), compact_kv_tensor(value_cache, retained_indices)

    B, H, T, D = key_cache.shape
    budget = retained_indices.shape[-1]

    if retained_indices.dim() == 2:
        retained_indices = retained_indices.unsqueeze(1).expand(B, H, budget).contiguous()

    k_out = torch.empty((B, H, budget, D), device=key_cache.device, dtype=key_cache.dtype)
    v_out = torch.empty((B, H, budget, D), device=value_cache.device, dtype=value_cache.dtype)

    BLOCK_BGT = 64
    grid = (triton.cdiv(budget, BLOCK_BGT), B * H)

    _gather_compact_kernel[grid](
        key_cache, value_cache,
        k_out, v_out,
        retained_indices,
        key_cache.stride(0), key_cache.stride(1), key_cache.stride(2), key_cache.stride(3),
        value_cache.stride(0), value_cache.stride(1), value_cache.stride(2), value_cache.stride(3),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        v_out.stride(0), v_out.stride(1), v_out.stride(2), v_out.stride(3),
        retained_indices.stride(0), retained_indices.stride(1), retained_indices.stride(2),
        BUDGET=budget,
        HEAD_DIM=D,
        BLOCK_BGT=BLOCK_BGT,
    )

    return k_out, v_out
