"""EKVA v2: Expert-Conditioned Saliency and Shared KV Cache Retention."""
from ekva.retention.routing_signature import RoutingSignature, MoERoutingHook
from ekva.retention.saliency import (
    ExpertProfile,
    compute_routing_conditioned_score,
    compute_recency_score,
    compute_sink_score,
    combined_token_saliency,
)
from ekva.retention.eviction import (
    select_topk_indices,
    compact_kv_tensor,
    evict_shared_kv_cache,
)

__all__ = [
    "RoutingSignature",
    "MoERoutingHook",
    "ExpertProfile",
    "compute_routing_conditioned_score",
    "compute_recency_score",
    "compute_sink_score",
    "combined_token_saliency",
    "select_topk_indices",
    "compact_kv_tensor",
    "evict_shared_kv_cache",
]
