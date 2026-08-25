"""EKVA v2: Expert-Conditioned Token Saliency Engine.

Computes the token-level retention score S(x_t) by combining:
1. Standard attention-magnitude/sink scores (A_hat, Sink, Recency).
2. Routing-conditioned semantic niche score R(x_t) derived from cross-layer MoE routing.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import math
import torch
from ekva.retention.routing_signature import RoutingSignature


@dataclass
class ExpertProfile:
    """Calibrated properties of an MoE expert."""
    entropy: float            # H_bar_e (avg attention entropy of tokens routed here)
    routing_freq: float       # Route_e (global routing volume)
    specialization: float     # Spec_e = 1 - Evenness_e (semantic selectivity)

    @property
    def score_multiplier(self) -> float:
        """Core multi-signal term: H_bar_e * log(1 + Route_e) * (1 + Spec_e)."""
        route_term = math.log1p(max(0.0, self.routing_freq))
        spec_term = 1.0 + max(0.0, min(1.0, self.specialization))
        return float(self.entropy * route_term * spec_term)


def compute_routing_conditioned_score(
    signature: RoutingSignature,
    expert_profiles: Dict[int, ExpertProfile],
    num_experts: Optional[int] = None,
) -> torch.Tensor:
    """Compute R(x_t) across all tokens in the sequence.

    Args:
        signature: RoutingSignature with shape (batch_size, seq_len, num_layers, top_k).
        expert_profiles: Mapping expert_id -> ExpertProfile.
        num_experts: Total experts count (if None, inferred from profiles).

    Returns:
        Tensor of shape (batch_size, seq_len) with normalized R(x_t) scores in [0, 1].
    """
    indices = signature.expert_indices  # (B, T, L, K)
    weights = signature.routing_weights # (B, T, L, K) or None
    B, T, L, K = indices.shape

    # Pre-build lookup table for expert score multipliers
    max_eid = max(expert_profiles.keys()) if expert_profiles else 0
    if num_experts is not None:
        max_eid = max(max_eid, num_experts - 1)
    
    score_table = torch.zeros(max_eid + 1, dtype=torch.float32, device=indices.device)
    for eid, prof in expert_profiles.items():
        if eid <= max_eid:
            score_table[eid] = prof.score_multiplier

    # Vectorized lookup: (B, T, L, K) -> score values
    # Clamp indices to valid table range
    clamped_indices = indices.clamp(0, max_eid)
    token_expert_scores = score_table[clamped_indices] # (B, T, L, K)

    if weights is not None:
        # Weight by routing probabilities
        w_norm = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        layer_scores = (token_expert_scores * w_norm.to(token_expert_scores.device)).sum(dim=-1) # (B, T, L)
    else:
        # Simple average across top_k
        layer_scores = token_expert_scores.mean(dim=-1) # (B, T, L)

    # Average across all L transformer layers: (B, T)
    r_scores = layer_scores.mean(dim=-1)

    # Normalize per-batch to [0, 1] range
    min_val = r_scores.min(dim=-1, keepdim=True).values
    max_val = r_scores.max(dim=-1, keepdim=True).values
    denom = (max_val - min_val).clamp(min=1e-8)
    r_normalized = (r_scores - min_val) / denom

    return r_normalized


def compute_recency_score(seq_len: int, tau: float = 64.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Computes exponential recency decay score for tokens 0..seq_len-1."""
    positions = torch.arange(seq_len, dtype=torch.float32, device=device)
    dist_from_end = (seq_len - 1) - positions
    return torch.exp(-dist_from_end / max(1.0, tau))


def compute_sink_score(seq_len: int, num_sink_tokens: int = 4, device: Optional[torch.device] = None) -> torch.Tensor:
    """Computes binary indicator for initial attention sink tokens."""
    scores = torch.zeros(seq_len, dtype=torch.float32, device=device)
    scores[:min(seq_len, num_sink_tokens)] = 1.0
    return scores


def combined_token_saliency(
    attn_scores: torch.Tensor,               # (B, T) or (B, H, T)
    routing_scores: Optional[torch.Tensor],  # (B, T)
    num_sink_tokens: int = 4,
    recency_tau: float = 64.0,
    w_a: float = 0.60,
    w_r: float = 0.30,
    w_s: float = 0.05,
    w_c: float = 0.05,
) -> torch.Tensor:
    """Computes unified token retention saliency S(x_t).

    S(x_t) = w_a * A_hat(x_t) + w_r * R(x_t) + w_s * Sink(x_t) + w_c * Recency(x_t)

    Returns:
        Tensor of shape identical to attn_scores with combined retention saliency.
    """
    device = attn_scores.device
    T = attn_scores.shape[-1]

    # Normalize attention scores to [0, 1]
    a_min = attn_scores.min(dim=-1, keepdim=True).values
    a_max = attn_scores.max(dim=-1, keepdim=True).values
    a_norm = (attn_scores - a_min) / (a_max - a_min).clamp(min=1e-8)

    sink = compute_sink_score(T, num_sink_tokens=num_sink_tokens, device=device)
    recency = compute_recency_score(T, tau=recency_tau, device=device)

    # Broadcast 1D components to match attn_scores shape
    while sink.dim() < attn_scores.dim():
        sink = sink.unsqueeze(0)
        recency = recency.unsqueeze(0)

    # Saliency combination
    saliency = w_a * a_norm + w_s * sink + w_c * recency

    if routing_scores is not None and w_r > 0.0:
        r_norm = routing_scores
        while r_norm.dim() < attn_scores.dim():
            r_norm = r_norm.unsqueeze(1)
        saliency = saliency + w_r * r_norm.to(device)

    # Re-enforce explicit sink tokens to maximum saliency
    if num_sink_tokens > 0:
        if saliency.dim() == 2:
            saliency[:, :num_sink_tokens] = float("inf")
        elif saliency.dim() == 3:
            saliency[:, :, :num_sink_tokens] = float("inf")

    return saliency
