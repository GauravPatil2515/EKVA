"""Per-expert KV cache buffer with pluggable eviction policies.

This is a pure-Python/PyTorch simulator used in Phase 2 to validate that
adaptive per-expert budgets improve quality over uniform budgets before any
kernel work is attempted.
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


class ExpertKVBuffer:
    """Fixed-capacity KV cache buffer for a single expert.

    When the buffer is full, an eviction policy decides which token(s) to drop
    before inserting the new KV pair.

    Args:
        budget: maximum number of tokens to retain.
        head_dim: dimensionality of each K or V head vector.
        num_heads: number of attention heads for this expert.
        eviction: one of "recency" | "attention" | "random".
        dtype: tensor dtype (default float16).
        device: torch device.
    """

    def __init__(
        self,
        budget: int,
        head_dim: int,
        num_heads: int,
        eviction: str = "recency",
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        self.budget = budget
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eviction = eviction
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        # (budget, num_heads, head_dim)
        self._k: Optional[Tensor] = None
        self._v: Optional[Tensor] = None
        # Cumulative attention scores per token position (used by "attention" eviction)
        self._attn_scores: Optional[Tensor] = None
        self._size: int = 0

    @property
    def size(self) -> int:
        return self._size

    def is_full(self) -> bool:
        return self._size >= self.budget

    def reset(self) -> None:
        self._k = None
        self._v = None
        self._attn_scores = None
        self._size = 0

    def update(
        self,
        new_k: Tensor,
        new_v: Tensor,
        attn_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Insert new_k / new_v into the buffer, evicting if necessary.

        Args:
            new_k: (seq_len, num_heads, head_dim) new keys.
            new_v: (seq_len, num_heads, head_dim) new values.
            attn_weights: (seq_len, num_heads, current_buf_size) optional attention
                          weights used to update importance scores of cached tokens.

        Returns:
            (k_cache, v_cache) — the full current buffer contents as tensors.
        """
        new_k = new_k.to(self.dtype).to(self.device)
        new_v = new_v.to(self.dtype).to(self.device)
        seq_len = new_k.shape[0]

        for t in range(seq_len):
            k_t = new_k[t].unsqueeze(0)  # (1, num_heads, head_dim)
            v_t = new_v[t].unsqueeze(0)

            if attn_weights is not None:
                # attn_weights[t]: (num_heads, current_buf_size)
                score_t = attn_weights[t].mean(dim=0)  # (current_buf_size,)
            else:
                score_t = None

            self._insert(k_t, v_t, score_t)

        return self._k, self._v

    def _insert(self, k_t: Tensor, v_t: Tensor, score_update: Optional[Tensor]) -> None:
        """Insert a single token's KV into the buffer, evicting if at capacity."""
        # Update attention scores for existing tokens
        if score_update is not None and self._attn_scores is not None:
            n = min(self._size, len(score_update))
            self._attn_scores[:n] += score_update[:n]

        if self._size < self.budget:
            # Buffer not full yet — append
            if self._k is None:
                self._k = torch.zeros(self.budget, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                self._v = torch.zeros_like(self._k)
                self._attn_scores = torch.zeros(self.budget, dtype=torch.float32, device=self.device)

            self._k[self._size] = k_t[0]
            self._v[self._size] = v_t[0]
            self._attn_scores[self._size] = 0.0
            self._size += 1
        else:
            # Buffer full — evict one token
            evict_idx = self._evict_index()
            self._k[evict_idx] = k_t[0]
            self._v[evict_idx] = v_t[0]
            self._attn_scores[evict_idx] = 0.0

    def _evict_index(self) -> int:
        """Return the index of the token to evict based on eviction strategy."""
        if self.eviction == "recency":
            return 0  # evict oldest (index 0, FIFO)
        elif self.eviction == "attention":
            # Evict token with lowest cumulative attention score
            return int(self._attn_scores[: self._size].argmin().item())
        elif self.eviction == "random":
            return int(torch.randint(0, self._size, (1,)).item())
        else:
            raise ValueError(f"Unknown eviction strategy: {self.eviction}")

    def get(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Return current (k_cache, v_cache) views of filled entries."""
        if self._k is None:
            return None, None
        return self._k[: self._size], self._v[: self._size]
