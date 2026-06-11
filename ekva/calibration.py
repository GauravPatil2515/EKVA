from importlib import import_module
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class ExpertStats:
    """Container for per-expert statistics collected during calibration.

    Stores running sums so we can compute averages at the end without holding
    all intermediate tensors in memory.
    """

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        # Sum of entropies per layer for this expert
        self.entropy_sum = torch.zeros(num_layers, dtype=torch.float64)
        # Number of attention samples accumulated per layer
        self.count = torch.zeros(num_layers, dtype=torch.int64)
        # Routing count (how many tokens selected this expert across layers)
        self.routing_count = 0

    def update_entropy(self, layer_idx: int, attn_probs: torch.Tensor) -> None:
        """Update running entropy statistics for a given layer.

        attn_probs: (batch, heads, query_len, key_len)
        """
        with torch.no_grad():
            # Avoid log(0) by adding small epsilon
            eps = 1e-12
            p = attn_probs.clamp_min(eps)
            entropy = -(p * p.log()).sum(dim=-1)  # sum over key_len
            # Mean over batch, heads, query_len
            entropy_mean = entropy.mean().double()

            self.entropy_sum[layer_idx] += entropy_mean
            self.count[layer_idx] += 1

    def update_routing(self, num_tokens: int) -> None:
        self.routing_count += int(num_tokens)

    def finalize(self) -> Dict[str, torch.Tensor]:
        """Return average entropy per layer for this expert."""
        avg_entropy = torch.zeros_like(self.entropy_sum)
        mask = self.count > 0
        avg_entropy[mask] = self.entropy_sum[mask] / self.count[mask].double()
        return {
            "avg_entropy": avg_entropy,
            "routing_count": torch.tensor(self.routing_count, dtype=torch.int64),
        }


def _get_moe_layers(model: nn.Module) -> List[nn.Module]:
    """Best-effort extraction of MoE layers from Mixtral / DeepSeek-style models.

    This function is intentionally conservative: you can customize it for your
    specific HF model implementations by adjusting the class name checks.
    """
    moe_layers: List[nn.Module] = []
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "mixtureofexperts" in class_name or "moe" in class_name:
            moe_layers.append(module)
    return moe_layers


def _entropy_from_logits(attn_logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Utility to convert attention logits to probabilities and compute entropy.

    This is useful if you hook *before* softmax in the attention implementation.
    """
    if mask is not None:
        attn_logits = attn_logits + mask
    attn_probs = attn_logits.softmax(dim=-1)
    eps = 1e-12
    p = attn_probs.clamp_min(eps)
    entropy = -(p * p.log()).sum(dim=-1)
    return entropy


@torch.no_grad()
def calibrate_expert_entropy(
    model: nn.Module,
    tokenizer,
    calibration_prompts: List[str],
    num_experts: int,
    max_new_tokens: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Run a lightweight calibration pass to estimate per-expert attention entropy.

    Args:
        model: HF-style MoE model (e.g., Mixtral-8x7B, DeepSeek-V2) in eval mode.
        tokenizer: corresponding tokenizer.
        calibration_prompts: small list of prompts representative of deployment.
        num_experts: number of experts per MoE layer (for initialization).
        max_new_tokens: optional generation length; set >0 to capture decode-time stats.
        device: optional device override.

    Returns:
        Dict mapping expert_id -> {"avg_entropy": Tensor[L], "routing_count": Tensor[]}
        where L is the number of transformer layers.
    """
    model.eval()
    if device is not None:
        model.to(device)

    # Attempt to locate MoE layers
    moe_layers = _get_moe_layers(model)
    if not moe_layers:
        raise RuntimeError("No MoE layers found. Customize _get_moe_layers for your model.")

    num_layers = len(moe_layers)

    # One ExpertStats per expert id
    expert_stats: Dict[int, ExpertStats] = {
        expert_id: ExpertStats(num_layers=num_layers) for expert_id in range(num_experts)
    }

    # Hooks to capture attention probabilities and routing decisions per expert
    handles = []

    def make_moe_forward_hook(layer_idx: int):
        def hook(module, args, output):
            """Assumes output contains per-expert attention or routing info.

            This is intentionally abstract because HF implementations differ.
            You will likely need to adapt this to your MoE block signature.
            """
            # Example sketch for Mixtral-style block:
            # output might be (hidden_states, router_logits, attn_outputs, ...)
            # You can inspect `output` in a debug run and adapt.
            if not isinstance(output, tuple):
                return

            # Routing: top-k expert indices per token
            # router_logits: (batch, seq_len, num_experts)
            router_logits = None
            attn_probs_per_expert: Optional[Tuple[torch.Tensor, ...]] = None

            for tensor in output:
                if isinstance(tensor, torch.Tensor) and tensor.dim() == 3 and tensor.size(-1) == num_experts:
                    router_logits = tensor
                # If the implementation exposes per-expert attention probs as a tuple/list,
                # you can pattern match here.

            if router_logits is None:
                return

            # Compute routing decisions and counts
            # (batch, seq_len, top_k)
            top_experts = router_logits.topk(k=2, dim=-1).indices  # example: top-2 routing
            bsz, seqlen, topk = top_experts.shape

            # Count how many tokens each expert gets
            for b in range(bsz):
                for t in range(seqlen):
                    for k in range(topk):
                        expert_id = int(top_experts[b, t, k])
                        expert_stats[expert_id].update_routing(num_tokens=1)

            # TODO: attach attention probability capture here once you identify
            # where per-expert attention weights live in the module.

        return hook

    for layer_idx, moe in enumerate(moe_layers):
        handles.append(moe.register_forward_hook(make_moe_forward_hook(layer_idx)))

    try:
        for prompt in calibration_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            if max_new_tokens > 0:
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens)
            else:
                _ = model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Finalize stats into avg entropy + routing counts per expert
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id, stats in expert_stats.items():
        result[expert_id] = stats.finalize()

    return result
