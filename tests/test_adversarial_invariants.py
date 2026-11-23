"""Adversarial invariant test suite for production KV eviction hardening.

Verifies:
1. RoPE & attention equivalence: keep_ratio=1.0 matches uncompacted KV cache identically.
2. Variable-length & padded batches: padding tokens are never indexed.
3. Multi-turn / periodic decode-time compaction: cache remains valid across cyclic eviction.
4. GQA head alignment: pooling from query heads down to KV heads preserves shapes and order.
5. Prefix-pinning: first N tokens (system prompt / RAG context) are immutably retained.
6. FP8 quantized cache compatibility: handles float8 formats cleanly.
"""
import pytest
import torch
from ekva.retention.saliency import combined_token_saliency, reduce_gqa_heads
from ekva.retention.eviction import select_topk_indices, compact_kv_tensor, evict_shared_kv_cache


def test_prefix_pinning_invariance():
    """Verify that protected_prefix_len guarantees first N tokens remain strictly in place."""
    B, T = 2, 128
    protected_prefix = 32
    budget = 64
    
    # Random attention and routing scores
    attn_scores = torch.rand(B, T)
    # Even if prefix has lowest possible score
    attn_scores[:, :protected_prefix] = 0.001
    
    saliency = combined_token_saliency(
        attn_scores=attn_scores,
        routing_scores=None,
        num_sink_tokens=4,
        protected_prefix_len=protected_prefix,
    )
    
    indices = select_topk_indices(
        saliency_scores=saliency,
        budget=budget,
        protected_prefix_len=protected_prefix,
    )
    
    assert indices.shape == (B, budget)
    # The first 32 indices must be exactly 0, 1, 2, ..., 31 for all batches
    for b in range(B):
        prefix_indices = indices[b, :protected_prefix].tolist()
        expected = list(range(protected_prefix))
        assert prefix_indices == expected, f"Prefix pinning violated: {prefix_indices} != {expected}"


def test_padded_batches_mask_exclusion():
    """Verify that padding tokens (attention_mask == 0) are never selected in top-B."""
    B, T = 2, 64
    budget = 20
    
    # Sequence 0 has length 40 (24 padding tokens on the right)
    # Sequence 1 has length 50 (14 padding tokens on the right)
    attention_mask = torch.ones(B, T)
    attention_mask[0, 40:] = 0
    attention_mask[1, 50:] = 0
    
    attn_scores = torch.rand(B, T)
    saliency = combined_token_saliency(
        attn_scores=attn_scores,
        routing_scores=None,
        attention_mask=attention_mask,
    )
    
    indices = select_topk_indices(saliency, budget=budget)
    
    # Assert no index >= 40 is in batch 0, and no index >= 50 is in batch 1
    assert (indices[0] < 40).all(), "Batch 0 selected padding tokens!"
    assert (indices[1] < 50).all(), "Batch 1 selected padding tokens!"


def test_gqa_head_alignment():
    """Verify GQA reduction pools Query heads cleanly down to KV heads."""
    B, T = 2, 32
    H_q, H_kv = 32, 4  # Standard GQA setup (8 query heads per KV head)
    
    attn_scores = torch.rand(B, H_q, T)
    pooled = reduce_gqa_heads(attn_scores, num_kv_heads=H_kv)
    
    assert pooled.shape == (B, H_kv, T)
    
    # Check that each KV head is indeed the mean of its 8 corresponding query heads
    for h in range(H_kv):
        expected_mean = attn_scores[:, h*8:(h+1)*8, :].mean(dim=1)
        assert torch.allclose(pooled[:, h, :], expected_mean, atol=1e-6)


def test_rope_attention_equivalence_full_budget():
    """Verify that keep_ratio=1.0 produces bit-identical KV tensors without altering ordering."""
    B, H, T, D = 2, 8, 128, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    key = torch.randn(B, H, T, D, device=device)
    val = torch.randn(B, H, T, D, device=device)
    saliency = torch.rand(B, T, device=device)
    
    comp_k, comp_v, kept_idx = evict_shared_kv_cache(
        key_cache=key,
        value_cache=val,
        saliency_scores=saliency,
        budget=T,
    )
    
    assert torch.equal(comp_k, key), "Compacted Key differs from original under 100% budget!"
    assert torch.equal(comp_v, val), "Compacted Value differs from original under 100% budget!"
    assert kept_idx.shape == (B, T)
    for b in range(B):
        assert kept_idx[b].tolist() == list(range(T))


def test_decode_time_periodic_compaction():
    """Simulates multi-turn / periodic decode compaction across multiple rounds."""
    B, H, D = 1, 4, 32
    max_capacity = 64
    evict_budget = 48
    prefix_len = 16
    
    # Start with initial 64 tokens
    curr_t = max_capacity
    key = torch.randn(B, H, curr_t, D)
    val = torch.randn(B, H, curr_t, D)
    
    # Run 4 eviction cycles
    for cycle in range(4):
        saliency = torch.rand(B, curr_t)
        comp_k, comp_v, kept_idx = evict_shared_kv_cache(
            key_cache=key,
            value_cache=val,
            saliency_scores=saliency,
            budget=evict_budget,
            protected_prefix_len=prefix_len,
        )
        assert comp_k.shape == (B, H, evict_budget, D)
        # Prefix must stay intact
        assert kept_idx[0, :prefix_len].tolist() == list(range(prefix_len))
        
        # Simulate decode generation: append 16 new tokens
        new_k = torch.randn(B, H, 16, D)
        new_v = torch.randn(B, H, 16, D)
        key = torch.cat([comp_k, new_k], dim=2)
        val = torch.cat([comp_v, new_v], dim=2)
        curr_t = key.shape[2]
        assert curr_t == max_capacity


def test_fp8_dtype_preservation():
    """Verify that compact_kv_tensor operates seamlessly with FP8 tensors."""
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("PyTorch build does not support float8_e4m3fn")
    
    B, H, T, D = 2, 4, 64, 32
    budget = 32
    
    fp8_tensor = torch.randn(B, H, T, D).to(torch.float8_e4m3fn)
    indices = torch.arange(budget).unsqueeze(0).expand(B, budget)
    
    compacted = compact_kv_tensor(fp8_tensor, indices)
    assert compacted.shape == (B, H, budget, D)
    assert compacted.dtype == torch.float8_e4m3fn
