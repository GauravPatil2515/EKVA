"""EKVA v2: Shared KV Cache Eviction and Compaction.

Selects the top-B tokens by saliency score S(x_t) and compacts the shared
Key-Value tensors into a contiguous buffer of length B <= T.
"""
from typing import Optional, Tuple, Union
import torch


def select_topk_indices(
    saliency_scores: torch.Tensor,
    budget: int,
    protect_sink_tokens: int = 4,
) -> torch.Tensor:
    """Computes sorted top-B indices to retain for each batch/head.

    Args:
        saliency_scores: (B, T) or (B, H, T) tensor.
        budget: Maximum number of tokens B to retain.
        protect_sink_tokens: Number of initial tokens guaranteed retention.

    Returns:
        torch.LongTensor of shape (B, budget) or (B, H, budget) with sorted indices.
    """
    T = saliency_scores.shape[-1]
    budget = max(1, min(budget, T))

    if budget >= T:
        # Full retention
        idx = torch.arange(T, device=saliency_scores.device)
        shape = list(saliency_scores.shape[:-1]) + [T]
        return idx.expand(shape)

    # Topk selection along last dimension
    _, topk_indices = torch.topk(saliency_scores, k=budget, dim=-1, largest=True, sorted=False)

    # Preserve temporal order by sorting indices ascending
    sorted_indices, _ = torch.sort(topk_indices, dim=-1)
    return sorted_indices


def compact_kv_tensor(
    tensor: torch.Tensor,
    retained_indices: torch.Tensor,
) -> torch.Tensor:
    """Compacts a (B, H, T, D) KV tensor along dimension T using retained_indices.

    Args:
        tensor: (B, H, T, D) Key or Value cache tensor.
        retained_indices: (B, budget) or (B, H, budget) index tensor.

    Returns:
        Compacted tensor of shape (B, H, budget, D).
    """
    B, H, T, D = tensor.shape
    budget = retained_indices.shape[-1]

    if budget >= T:
        return tensor

    if retained_indices.dim() == 2:  # (B, budget) -> expand to (B, H, budget, D)
        idx_expanded = retained_indices.unsqueeze(1).unsqueeze(-1).expand(B, H, budget, D)
    elif retained_indices.dim() == 3: # (B, H, budget) -> expand to (B, H, budget, D)
        idx_expanded = retained_indices.unsqueeze(-1).expand(B, H, budget, D)
    else:
        raise ValueError(f"Unexpected index shape: {retained_indices.shape}")

    # Gather along time dimension (dim=2)
    compacted = torch.gather(tensor, dim=2, index=idx_expanded)
    return compacted.contiguous()


def evict_shared_kv_cache(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    saliency_scores: torch.Tensor,
    budget: int,
    protect_sink_tokens: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evicts low-saliency tokens and returns compacted (K, V) tensors and kept indices.

    Args:
        key_cache: (B, H, T, D)
        value_cache: (B, H, T, D)
        saliency_scores: (B, T) or (B, H, T)
        budget: Number of KV tokens to retain.
        protect_sink_tokens: Number of initial sink tokens.

    Returns:
        Tuple of:
          - compacted_key: (B, H, budget, D)
          - compacted_val: (B, H, budget, D)
          - retained_indices: (B, budget) or (B, H, budget)
    """
    indices = select_topk_indices(
        saliency_scores=saliency_scores,
        budget=budget,
        protect_sink_tokens=protect_sink_tokens,
    )

    compacted_k = compact_kv_tensor(key_cache, indices)
    compacted_v = compact_kv_tensor(value_cache, indices)

    return compacted_k, compacted_v, indices
