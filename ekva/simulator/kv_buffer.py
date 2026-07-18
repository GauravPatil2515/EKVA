"""Per-expert KV cache buffer with pluggable eviction policies.

Pure-Python/PyTorch simulator used to validate that adaptive per-expert budgets
improve quality over uniform budgets *before* any kernel work is attempted.

Eviction strategies (set via `eviction=`): "recency" (FIFO), "attention"
(lowest cumulative attention score), "random", "hybrid" (recency + attention).
"""
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor

SUPPORTED_EVICTION = ("recency", "attention", "random", "hybrid")


class ExpertKVBuffer:
    def __init__(
        self,
        budget: int,
        head_dim: int,
        num_heads: int,
        eviction: str = "recency",
        hybrid_attn_weight: float = 0.5,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        if eviction not in SUPPORTED_EVICTION:
            raise ValueError(f"eviction must be one of {SUPPORTED_EVICTION}, got {eviction!r}")
        self.budget = budget
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eviction = eviction
        self.hybrid_attn_weight = hybrid_attn_weight
        self.dtype = dtype
        self.device = device or torch.device("cpu")

        self._k: Optional[Tensor] = None
        self._v: Optional[Tensor] = None
        self._attn_scores: Optional[Tensor] = None
        self._age: Optional[Tensor] = None
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
        self._age = None
        self._size = 0

    def update(
        self, new_k: Tensor, new_v: Tensor, attn_weights: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        new_k = new_k.to(self.dtype).to(self.device)
        new_v = new_v.to(self.dtype).to(self.device)
        seq_len = new_k.shape[0]
        for t in range(seq_len):
            k_t = new_k[t].unsqueeze(0)
            v_t = new_v[t].unsqueeze(0)
            score_t = attn_weights[t].mean(dim=0) if attn_weights is not None else None
            self._insert(k_t, v_t, score_t)
        return self._k, self._v

    def _insert(self, k_t: Tensor, v_t: Tensor, score_update: Optional[Tensor]) -> None:
        if score_update is not None and self._attn_scores is not None:
            n = min(self._size, len(score_update))
            self._attn_scores[:n] += score_update[:n]

        if self._size < self.budget:
            if self._k is None:
                self._k = torch.zeros(self.budget, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                self._v = torch.zeros_like(self._k)
                self._attn_scores = torch.zeros(self.budget, dtype=torch.float32, device=self.device)
                self._age = torch.zeros(self.budget, dtype=torch.float32, device=self.device)
            self._k[self._size] = k_t[0]
            self._v[self._size] = v_t[0]
            self._attn_scores[self._size] = 0.0
            self._age[self._size] = 0.0
            self._size += 1
        else:
            evict_idx = self._evict_index()
            self._k[evict_idx] = k_t[0]
            self._v[evict_idx] = v_t[0]
            self._attn_scores[evict_idx] = 0.0
            self._age[evict_idx] = 0.0

    def _evict_index(self) -> int:
        if self.eviction == "recency":
            return 0  # FIFO: oldest is at index 0
        if self.eviction == "attention":
            return int(self._attn_scores[: self._size].argmin().item())
        if self.eviction == "random":
            return int(torch.randint(0, self._size, (1,)).item())
        if self.eviction == "hybrid":
            # Combine normalized attention score and age; evict the worst.
            a = self._attn_scores[: self._size]
            age = self._age[: self._size]
            a_n = a / (a.max() + 1e-9)
            age_n = age / (age.max() + 1e-9)
            cost = self.hybrid_attn_weight * (1.0 - a_n) + (1.0 - self.hybrid_attn_weight) * age_n
            return int(cost.argmax().item())
        raise ValueError(f"Unknown eviction strategy: {self.eviction}")

    def get(self) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        if self._k is None:
            return None, None
        return self._k[: self._size], self._v[: self._size]
