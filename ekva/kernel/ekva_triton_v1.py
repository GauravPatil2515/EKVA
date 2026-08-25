"""EKVA Triton kernel v1 — variable KV tile count per expert (Week 10).

Difference vs baseline: the KV tile loop stops after `kv_budget // BLOCK_N`
tiles instead of iterating over the full sequence. `kv_budget` is per-expert
(this function takes the budget for the expert currently being computed).

Correctness target: output must match standard attention when kv_budget == N_KV.
Benchmark target: lower HBM reads + kernel time vs baseline FA2 at reduced budget.
"""
import torch
import triton
import triton.language as tl

from ekva.kernel.reference_flashattn2 import _fwd_kernel  # reuse tiling for v1 baseline-correctness


@triton.jit
def _fwd_kernel_v1(
    Q, K, V, Out,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    N_CTX, N_KV, KV_BUDGET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    off_h = pid_bh % (stride_qb // stride_qh) if stride_qb > 0 else pid_bh
    off_b = pid_bh // (stride_qb // stride_qh) if stride_qb > 0 else 0

    q_offset = off_b * stride_qb + off_h * stride_qh
    k_offset = off_b * stride_kb + off_h * stride_kh
    v_offset = off_b * stride_vb + off_h * stride_vh
    o_offset = off_b * stride_ob + off_h * stride_oh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + q_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qd),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + k_offset,
        shape=(HEAD_DIM, N_KV),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + v_offset,
        shape=(N_KV, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + o_offset,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_od),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )

    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_block_ptr, boundary_check=(0, 1))

    n_tiles = tl.cdiv(KV_BUDGET, BLOCK_N)
    for tile in range(n_tiles):
        start_n = tile * BLOCK_N
        k = tl.load(K_block_ptr, boundary_check=(0, 1))
        v = tl.load(V_block_ptr, boundary_check=(1, 0))
        qk = tl.dot(q, k)
        n_mask = (start_n + offs_n)[None, :] < KV_BUDGET
        qk = tl.where(n_mask, qk, float("-inf"))
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.exp(qk - m_ij[:, None])
        alpha = tl.exp(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_ij
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))

    acc = acc / l_i[:, None]
    tl.store(O_block_ptr, acc.to(O_block_ptr.dtype.element_ty), boundary_check=(0, 1))


def ekva_attention_v1(q, k, v, kv_budget, scale=None):
    """q,k,v: (B, H, N, D). kv_budget: int <= N. Returns (B, H, N, D)."""
    B, H, N, D = q.shape
    if scale is None:
        scale = D ** -0.5
    q = (q * scale).contiguous()
    k = k.contiguous()
    v = v.contiguous()
    o = torch.empty_like(q)
    BLOCK_M, BLOCK_N = 64, 64
    grid = (triton.cdiv(N, BLOCK_M), B * H)
    _fwd_kernel_v1[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        N, N, KV_BUDGET=kv_budget,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, HEAD_DIM=D,
    )
    return o
