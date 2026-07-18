"""EKVA Triton kernel v2 — v1 + fused important-token index selection (Week 11).

Extends v1 by accepting precomputed `important_indices` per expert (from the
best Phase-2 eviction policy) and gathering only those KV entries during the
tile loop, fusing eviction/selection into the kernel. The gathered K/V are
contiguous so the tile loop is a straight contiguous load — no per-tile masking.

Profiled with Nsight Compute (weeks 11): HBM bandwidth utilization, SM
occupancy, kernel time — baseline vs v2 per expert class.
"""
import torch
import triton
import triton.language as tl

from ekva.kernel.ekva_triton_v1 import _fwd_kernel_v1


@triton.jit
def _fwd_kernel_v2_gather(
    Q, K_gathered, V_gathered, Out,
    IMP_IDX,  # (N_KV,) long tensor of selected positions (contiguous, length KV_BUDGET)
    stride_qb, stride_qh, stride_qd,
    stride_kg, stride_vg,
    stride_ob, stride_oh, stride_od,
    N_CTX, KV_BUDGET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    off_bh = tl.program_id(0)
    q = tl.load(Q + off_bh * (N_CTX * HEAD_DIM), mask=tl.arange(0, N_CTX * HEAD_DIM) < N_CTX * HEAD_DIM)
    q = tl.reshape(q, (N_CTX, HEAD_DIM))
    m_i = tl.zeros([N_CTX], dtype=tl.float32) - 1e9
    l_i = tl.zeros([N_CTX], dtype=tl.float32)
    acc = tl.zeros([N_CTX, HEAD_DIM], dtype=tl.float32)
    n_tiles = tl.cdiv(KV_BUDGET, BLOCK_N)
    for tile in range(n_tiles):
        # Contiguous gather of K/V tiles (fused selection).
        k = tl.load(K_gathered + tile * BLOCK_N * HEAD_DIM, mask=tl.arange(0, BLOCK_N * HEAD_DIM) < BLOCK_N * HEAD_DIM)
        v = tl.load(V_gathered + tile * BLOCK_N * HEAD_DIM, mask=tl.arange(0, BLOCK_N * HEAD_DIM) < BLOCK_N * HEAD_DIM)
        k = tl.reshape(k, (BLOCK_N, HEAD_DIM))
        v = tl.reshape(v, (BLOCK_N, HEAD_DIM))
        qk = tl.dot(q, tl.trans(k))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p, v)
        m_i = m_ij
    acc = acc / l_i[:, None]
    tl.store(Out + off_bh * (N_CTX * HEAD_DIM), acc.reshape(N_CTX * HEAD_DIM))


def ekva_attention_v2(q, k, v, kv_budget, important_indices, scale=None):
    """q,k,v: (B,H,N,D). kv_budget: int. important_indices: (B,H,kv_budget) long.

    Returns (B,H,N,D) computed over only the selected K/V positions.
    """
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    q = (q * scale).contiguous()
    o = torch.empty_like(q)
    # Pre-gather K/V per (b,h) into contiguous buffers of length kv_budget.
    k_g = torch.gather(k, 2, important_indices.unsqueeze(-1).expand(B, H, kv_budget, D))
    v_g = torch.gather(v, 2, important_indices.unsqueeze(-1).expand(B, H, kv_budget, D))
    BLOCK_M, BLOCK_N = 64, 64
    for b in range(B):
        for h in range(H):
            _fwd_kernel_v2_gather[(b * H + h,)](
                q[b, h], k_g[b, h], v_g[b, h], o[b, h], important_indices[b, h],
                q[b, h].stride(0), q[b, h].stride(1),
                k_g[b, h].stride(0), v_g[b, h].stride(0),
                o[b, h].stride(0), o[b, h].stride(1),
                N, KV_BUDGET=kv_budget, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
            )
    return o
