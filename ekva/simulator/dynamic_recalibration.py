"""Dynamic Online Re-Calibration Cascade (Novel Mechanism).

Periodically updates per-expert attention entropy and routing statistics
during long-context generation/decoding, dynamically adjusting KV-cache budgets
across experts to handle topic shifts, distribution changes, and multi-turn drift.
"""
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from ekva.budget.derive import derive_kv_budget
from ekva.simulator.kv_buffer import ExpertKVBuffer


class DynamicKVRecalibrationManager:
    """Manages online, streaming recalibration of per-expert KV budgets during inference.

    Args:
        num_experts: total number of experts in the MoE layer.
        total_budget: total token budget across all experts.
        min_per_expert: minimum floor per expert.
        recalibration_interval: number of tokens between dynamic budget updates.
        ema_alpha: exponential moving average weight for historical stats [0, 1].
        head_dim: dimension of each head.
        num_heads: number of attention heads.
        eviction: eviction strategy ('recency', 'attention', 'random', 'hybrid').
        device: torch device.
    """

    def __init__(
        self,
        num_experts: int,
        total_budget: int,
        min_per_expert: int = 64,
        recalibration_interval: int = 256,
        ema_alpha: float = 0.7,
        head_dim: int = 128,
        num_heads: int = 1,
        eviction: str = "hybrid",
        device: Optional[torch.device] = None,
    ):
        self.num_experts = num_experts
        self.total_budget = total_budget
        self.min_per_expert = min_per_expert
        self.recalibration_interval = recalibration_interval
        self.ema_alpha = ema_alpha
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.eviction = eviction
        self.device = device or torch.device("cpu")

        # Running statistics
        self.token_counter = 0
        self.recalibration_count = 0
        self.running_entropy = torch.ones(num_experts, dtype=torch.float32, device=self.device)
        self.running_routing = torch.ones(num_experts, dtype=torch.float32, device=self.device)
        self.current_window_entropy = torch.zeros(num_experts, dtype=torch.float32, device=self.device)
        self.current_window_routing = torch.zeros(num_experts, dtype=torch.float32, device=self.device)
        self.current_window_counts = torch.zeros(num_experts, dtype=torch.int64, device=self.device)

        # Initial uniform allocation
        init_budget = max(min_per_expert, total_budget // num_experts)
        self.current_budgets: Dict[int, int] = {i: init_budget for i in range(num_experts)}

        # Initialize expert KV buffers
        self.buffers: Dict[int, ExpertKVBuffer] = {
            eid: ExpertKVBuffer(
                budget=self.current_budgets[eid],
                head_dim=self.head_dim,
                num_heads=self.num_heads,
                eviction=self.eviction,
                device=self.device,
            )
            for eid in range(num_experts)
        }

        self.history_budgets: List[Dict[int, int]] = [dict(self.current_budgets)]

    def initialize_from_calibration(self, entropy_map: Dict[int, Dict[str, torch.Tensor]]) -> None:
        """Prime the running statistics with offline Phase-1 calibration data if available."""
        for eid, stats in entropy_map.items():
            if eid < self.num_experts:
                self.running_entropy[eid] = stats["avg_entropy"].mean().item()
                self.running_routing[eid] = float(stats["routing_count"].item())

        self._recompute_budgets()

    def record_step(
        self,
        expert_id: int,
        k: Tensor,
        v: Tensor,
        attn_probs: Optional[Tensor] = None,
    ) -> Tuple[Optional[Tensor], Optional[Tensor]]:
        """Record token routed to expert_id, store KV, and trigger online recalibration if interval reached."""
        self.token_counter += 1
        self.current_window_routing[expert_id] += 1.0

        attn_for_buf = None
        if attn_probs is not None:
            eps = 1e-12
            p = attn_probs.clamp_min(eps)
            ent = -(p * p.log()).sum(dim=-1).mean().item()
            self.current_window_entropy[expert_id] += ent
            self.current_window_counts[expert_id] += 1

            if attn_probs.dim() == 4:
                # (batch, heads, q_len, k_len) -> (q_len, heads, k_len)
                attn_for_buf = attn_probs.mean(dim=0).transpose(0, 1)
            elif attn_probs.dim() == 3:
                attn_for_buf = attn_probs
            elif attn_probs.dim() == 2:
                attn_for_buf = attn_probs.unsqueeze(1)

        # Store in buffer
        buf = self.buffers[expert_id]
        k_buf, v_buf = buf.update(k, v, attn_weights=attn_for_buf)

        # Check if periodic recalibration interval is triggered
        if self.token_counter % self.recalibration_interval == 0:
            self._trigger_recalibration()

        return k_buf, v_buf

    def _trigger_recalibration(self) -> None:
        """Execute dynamic recalibration cascade."""
        self.recalibration_count += 1

        for eid in range(self.num_experts):
            if self.current_window_counts[eid] > 0:
                win_ent = self.current_window_entropy[eid] / self.current_window_counts[eid].float()
            else:
                win_ent = self.running_entropy[eid]

            win_route = self.current_window_routing[eid]

            # Exponential Moving Average update
            self.running_entropy[eid] = (
                self.ema_alpha * self.running_entropy[eid] + (1.0 - self.ema_alpha) * win_ent
            )
            self.running_routing[eid] = (
                self.ema_alpha * self.running_routing[eid] + (1.0 - self.ema_alpha) * win_route
            )

        # Reset current window accumulators
        self.current_window_entropy.zero_()
        self.current_window_routing.zero_()
        self.current_window_counts.zero_()

        # Recompute budgets and resize buffers
        self._recompute_budgets()

    def _recompute_budgets(self) -> None:
        """Derive new budgets from current EMA statistics and adjust buffer capacities."""
        entropy_map = {
            eid: {
                "avg_entropy": self.running_entropy[eid].unsqueeze(0),
                "routing_count": self.running_routing[eid].long(),
            }
            for eid in range(self.num_experts)
        }

        new_budget_tensor = derive_kv_budget(
            entropy_map=entropy_map,
            total_budget=self.total_budget,
            min_per_expert=self.min_per_expert,
            strategy="proportional",
        )

        for eid in range(self.num_experts):
            new_b = int(new_budget_tensor[eid].item())
            self.current_budgets[eid] = new_b
            self._resize_buffer(eid, new_b)

        self.history_budgets.append(dict(self.current_budgets))

    def _resize_buffer(self, expert_id: int, new_budget: int) -> None:
        """Adjust buffer capacity. If shrinking, evict excess tokens; if growing, expand tensor allocation."""
        buf = self.buffers[expert_id]
        buf.budget = new_budget
        if buf._k is not None:
            old_size = min(buf._size, new_budget)
            new_k = torch.zeros(new_budget, buf.num_heads, buf.head_dim, dtype=buf.dtype, device=buf.device)
            new_v = torch.zeros_like(new_k)
            new_attn = torch.zeros(new_budget, dtype=torch.float32, device=buf.device)
            new_age = torch.zeros(new_budget, dtype=torch.float32, device=buf.device)

            if old_size > 0:
                new_k[:old_size] = buf._k[:old_size]
                new_v[:old_size] = buf._v[:old_size]
                if buf._attn_scores is not None:
                    new_attn[:old_size] = buf._attn_scores[:old_size]
                if buf._age is not None:
                    new_age[:old_size] = buf._age[:old_size]

            buf._k = new_k
            buf._v = new_v
            buf._attn_scores = new_attn
            buf._age = new_age
            buf._size = old_size

    def get_summary(self) -> Dict:
        """Return operational summary of the dynamic cascade."""
        return {
            "total_tokens_processed": self.token_counter,
            "recalibrations_performed": self.recalibration_count,
            "final_budgets": dict(self.current_budgets),
            "history_budgets": self.history_budgets,
            "running_entropy": self.running_entropy.tolist(),
            "running_routing": self.running_routing.tolist(),
        }
