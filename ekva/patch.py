"""Drop-in HuggingFace patcher for EKVA v2.

Usage:
    from transformers import AutoModelForCausalLM
    from ekva import patch_moe_ekva

    model = AutoModelForCausalLM.from_pretrained(...)
    model = patch_moe_ekva(
        model,
        budget_ratio=0.40,
        protected_prefix_len=256,
        periodic_decode_step=512,
        use_triton=True,
    )
"""
from typing import Optional, List, Tuple, Dict, Any
import torch
import torch.nn as nn

from ekva.retention.routing_signature import MoERoutingHook, RoutingSignature
from ekva.retention.saliency import combined_token_saliency
from ekva.retention.eviction import evict_shared_kv_cache, select_topk_indices
from ekva.kernel.ekva_eviction_v2 import triton_compact_kv_cache


class EKVAPatchedModel(nn.Module):
    """Wraps a causal MoE model with non-invasive routing hooks and automatic KV cache compaction."""
    def __init__(
        self,
        base_model: nn.Module,
        budget_ratio: float = 0.40,
        protected_prefix_len: int = 128,
        num_sink_tokens: int = 4,
        periodic_decode_step: int = 512,
        use_triton: bool = True,
        routing_weight: float = 0.30,
        attention_weight: float = 0.60,
    ):
        super().__init__()
        self.base_model = base_model
        self.budget_ratio = budget_ratio
        self.protected_prefix_len = protected_prefix_len
        self.num_sink_tokens = num_sink_tokens
        self.periodic_decode_step = periodic_decode_step
        self.use_triton = use_triton
        self.routing_weight = routing_weight
        self.attention_weight = attention_weight

        # Attach non-invasive routing hook
        self.hook = MoERoutingHook(self.base_model)
        self._decode_step_counter = 0

    def compact_cache(
        self,
        past_key_values: Any,
        attn_scores: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Any:
        """Compacts all layers in past_key_values according to EKVA token saliency."""
        if past_key_values is None or len(past_key_values) == 0:
            return past_key_values

        # Check sequence length from first layer's key tensor
        # PastKeyValues can be tuple of (K, V) or DynamicCache
        is_dynamic_cache = hasattr(past_key_values, "key_cache")
        if is_dynamic_cache:
            k0 = past_key_values.key_cache[0]
        else:
            k0 = past_key_values[0][0]

        B, H, T, D = k0.shape
        budget = int(T * self.budget_ratio)
        if budget >= T:
            return past_key_values

        # Fetch harvested routing signatures
        sig = self.hook.get_routing_signatures()
        routing_scores = None
        if sig is not None and self.routing_weight > 0.0:
            from ekva.retention.saliency import compute_routing_conditioned_score
            routing_scores = compute_routing_conditioned_score(sig, expert_profiles={})

        if attn_scores is None:
            # Fallback to uniform / recency / routing if attention is FlashAttention-native
            attn_scores = torch.ones(B, T, device=k0.device)

        # Compute saliency
        saliency = combined_token_saliency(
            attn_scores=attn_scores,
            routing_scores=routing_scores,
            num_sink_tokens=self.num_sink_tokens,
            protected_prefix_len=self.protected_prefix_len,
            attention_mask=attention_mask,
            w_a=self.attention_weight,
            w_r=self.routing_weight,
        )

        indices = select_topk_indices(
            saliency_scores=saliency,
            budget=budget,
            protect_sink_tokens=self.num_sink_tokens,
            protected_prefix_len=self.protected_prefix_len,
        )

        if is_dynamic_cache:
            for l in range(len(past_key_values.key_cache)):
                k = past_key_values.key_cache[l]
                v = past_key_values.value_cache[l]
                if self.use_triton and k.is_cuda:
                    ck, cv = triton_compact_kv_cache(k, v, indices)
                else:
                    from ekva.retention.eviction import compact_kv_tensor
                    ck = compact_kv_tensor(k, indices)
                    cv = compact_kv_tensor(v, indices)
                past_key_values.key_cache[l] = ck
                past_key_values.value_cache[l] = cv
            return past_key_values
        else:
            compacted_layers = []
            for k, v in past_key_values:
                if self.use_triton and k.is_cuda:
                    ck, cv = triton_compact_kv_cache(k, v, indices)
                else:
                    from ekva.retention.eviction import compact_kv_tensor
                    ck = compact_kv_tensor(k, indices)
                    cv = compact_kv_tensor(v, indices)
                compacted_layers.append((ck, cv))
            return tuple(compacted_layers)

    def forward(self, *args, **kwargs):
        # Extract use_cache
        use_cache = kwargs.get("use_cache", True)
        past_key_values = kwargs.get("past_key_values", None)

        self._decode_step_counter += 1

        # Periodic decode-time compaction
        if (
            past_key_values is not None
            and self.periodic_decode_step > 0
            and (self._decode_step_counter % self.periodic_decode_step == 0)
        ):
            kwargs["past_key_values"] = self.compact_cache(
                past_key_values,
                attention_mask=kwargs.get("attention_mask", None),
            )

        outputs = self.base_model(*args, **kwargs)

        # Prefill compaction: when processing prompt sequence (T > budget_ratio threshold)
        if kwargs.get("past_key_values", None) is None and hasattr(outputs, "past_key_values") and outputs.past_key_values is not None:
            # Check length of generated KV cache
            is_dc = hasattr(outputs.past_key_values, "key_cache")
            k0 = outputs.past_key_values.key_cache[0] if is_dc else outputs.past_key_values[0][0]
            if k0.shape[2] > self.protected_prefix_len * 2:
                outputs.past_key_values = self.compact_cache(
                    outputs.past_key_values,
                    attn_scores=outputs.attentions[-1] if hasattr(outputs, "attentions") and outputs.attentions else None,
                    attention_mask=kwargs.get("attention_mask", None),
                )

        return outputs

    def generate(self, *args, **kwargs):
        return self.base_model.generate(*args, **kwargs)


def patch_moe_ekva(
    model: nn.Module,
    budget_ratio: float = 0.40,
    protected_prefix_len: int = 128,
    num_sink_tokens: int = 4,
    periodic_decode_step: int = 512,
    use_triton: bool = True,
    routing_weight: float = 0.30,
    attention_weight: float = 0.60,
) -> EKVAPatchedModel:
    """Wraps any HuggingFace MoE causal model with EKVA v2 routing-aware token eviction.

    Args:
        model: HuggingFace model instance (e.g. Qwen1.5-MoE, Mixtral, DeepSeek).
        budget_ratio: Fraction of KV cache to retain (e.g., 0.40 = 40% budget).
        protected_prefix_len: Number of initial prefix tokens to pin as immutable.
        num_sink_tokens: Dedicated initial sink tokens guaranteed retention.
        periodic_decode_step: How often to compact during long decode runs (0 = prefill only).
        use_triton: Whether to use fused Triton kernel on GPU.
        routing_weight: Multi-signal weight for routing niche score.
        attention_weight: Multi-signal weight for attention mass.

    Returns:
        EKVAPatchedModel instance with active routing hooks and automated compaction.
    """
    return EKVAPatchedModel(
        base_model=model,
        budget_ratio=budget_ratio,
        protected_prefix_len=protected_prefix_len,
        num_sink_tokens=num_sink_tokens,
        periodic_decode_step=periodic_decode_step,
        use_triton=use_triton,
        routing_weight=routing_weight,
        attention_weight=attention_weight,
    )
