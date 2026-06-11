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

    def update_token_entropies(self, layer_idx: int, token_entropies: torch.Tensor) -> None:
        """Update running entropy statistics for specific tokens routed to this expert."""
        with torch.no_grad():
            self.entropy_sum[layer_idx] += token_entropies.sum().double()
            self.count[layer_idx] += token_entropies.numel()

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
    """Best-effort extraction of MoE layers from Mixtral / DeepSeek-style models."""
    moe_layers: List[nn.Module] = []
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "mixtureofexperts" in class_name or "moe" in class_name:
            moe_layers.append(module)
    return moe_layers


def _get_layer_pairs(model: nn.Module) -> List[Tuple[nn.Module, nn.Module]]:
    """Identify pairs of (attention_module, moe_module) for each layer."""
    pairs = []
    # Search for decoder blocks and extract their attention and MoE submodules
    for module in model.modules():
        class_name = module.__class__.__name__.lower()
        if "decoderlayer" in class_name or "block" in class_name:
            attn_module = None
            moe_module = None
            for name, child in module.named_children():
                child_class = child.__class__.__name__.lower()
                if "attention" in child_class or "attn" in child_class or "attention" in name or "attn" in name:
                    attn_module = child
                if "moe" in child_class or "mixtureofexperts" in child_class or "moe" in name:
                    moe_module = child
            if attn_module is not None and moe_module is not None:
                pairs.append((attn_module, moe_module))
                
    if not pairs:
        # Fallback to general listing if no decoder structure matches
        attns = []
        moes = []
        for module in model.modules():
            class_name = module.__class__.__name__.lower()
            if "attention" in class_name or "attn" in class_name:
                attns.append(module)
            if "moe" in class_name or "mixtureofexperts" in class_name:
                moes.append(module)
        if len(attns) == len(moes) and len(attns) > 0:
            pairs = list(zip(attns, moes))
            
    return pairs


def _entropy_from_logits(attn_logits: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Utility to convert attention logits to probabilities and compute entropy."""
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
        num_experts: number of experts per MoE layer.
        max_new_tokens: optional generation length; set >0 to capture decode-time stats.
        device: optional device override.

    Returns:
        Dict mapping expert_id -> {"avg_entropy": Tensor[L], "routing_count": Tensor[]}
        where L is the number of transformer layers.
    """
    model.eval()
    if device is not None:
        model.to(device)

    # Enable output_attentions in the model config during calibration
    original_output_attentions = getattr(model.config, "output_attentions", False)
    model.config.output_attentions = True

    # Locate layers with attention and MoE blocks
    layer_pairs = _get_layer_pairs(model)
    if not layer_pairs:
        # Fallback to just finding MoE layers
        moe_layers = _get_moe_layers(model)
        if not moe_layers:
            raise RuntimeError("No MoE layers found. Customize model parsing.")
        # Create dummy pairs where attention is not tracked
        layer_pairs = [(None, moe) for moe in moe_layers]

    num_layers = len(layer_pairs)

    # One ExpertStats per expert id
    expert_stats: Dict[int, ExpertStats] = {
        expert_id: ExpertStats(num_layers=num_layers) for expert_id in range(num_experts)
    }

    # Temporary storage to hold token attention entropy calculated in self_attn
    # before we route to experts in the same layer
    temp_entropy: Dict[int, torch.Tensor] = {}

    handles = []

    def make_attn_forward_hook(layer_idx: int):
        def hook(module, args, output):
            # Locate the attention weights tensor (typically 4D: batch, heads, query_len, key_len)
            attn_weights = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dim() == 4:
                        attn_weights = item
                        break
            elif isinstance(output, torch.Tensor) and output.dim() == 4:
                attn_weights = output

            if attn_weights is not None:
                with torch.no_grad():
                    eps = 1e-12
                    p = attn_weights.clamp_min(eps)
                    # Compute entropy over keys
                    entropy = -(p * p.log()).sum(dim=-1)  # (batch_size, num_heads, query_len)
                    # Average across attention heads
                    entropy_mean = entropy.mean(dim=1)  # (batch_size, query_len)
                    temp_entropy[layer_idx] = entropy_mean.cpu()
        return hook

    def make_moe_forward_hook(layer_idx: int):
        def hook(module, args, output):
            # Locate router_logits
            router_logits = None
            if isinstance(output, tuple):
                for item in output:
                    if isinstance(item, torch.Tensor) and item.size(-1) == num_experts:
                        router_logits = item
                        break
            elif isinstance(output, torch.Tensor) and output.size(-1) == num_experts:
                router_logits = output

            if router_logits is None:
                return

            with torch.no_grad():
                # Extract batch_size and seq_len from the input hidden states (args[0])
                hidden_states = args[0]
                batch_size = hidden_states.shape[0]
                seq_len = hidden_states.shape[1]

                # Reshape router_logits if batch/sequence dimensions are flattened (2D)
                if router_logits.dim() == 2:
                    router_logits = router_logits.view(batch_size, seq_len, -1)

                topk = min(2, num_experts)
                top_experts = router_logits.topk(k=topk, dim=-1).indices  # (batch_size, seq_len, topk)

                # Get token attention entropies for this layer
                layer_entropy = temp_entropy.get(layer_idx, None)

                # Assign token entropies to the selected experts
                for expert_id in range(num_experts):
                    # Mask indicating if expert_id is among topk experts for each token
                    selected_mask = (top_experts == expert_id).any(dim=-1)  # (batch_size, seq_len)
                    num_tokens = selected_mask.sum().item()
                    if num_tokens > 0:
                        expert_stats[expert_id].update_routing(num_tokens)
                        if layer_entropy is not None:
                            # Index layer_entropy to get entropy for only the selected tokens
                            token_entropies = layer_entropy[selected_mask]
                            expert_stats[expert_id].update_token_entropies(layer_idx, token_entropies)
        return hook

    # Register hooks on attention and MoE blocks
    for layer_idx, (attn, moe) in enumerate(layer_pairs):
        if attn is not None:
            handles.append(attn.register_forward_hook(make_attn_forward_hook(layer_idx)))
        if moe is not None:
            handles.append(moe.register_forward_hook(make_moe_forward_hook(layer_idx)))

    try:
        for prompt in calibration_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Clear previous sequence attention statistics
            temp_entropy.clear()
            if max_new_tokens > 0:
                _ = model.generate(**inputs, max_new_tokens=max_new_tokens, output_attentions=True)
            else:
                _ = model(**inputs, output_attentions=True)
    finally:
        # Clean up hooks
        for h in handles:
            h.remove()
        # Restore original configuration
        model.config.output_attentions = original_output_attentions

    # Finalize stats into avg entropy + routing counts per expert
    result: Dict[int, Dict[str, torch.Tensor]] = {}
    for expert_id, stats in expert_stats.items():
        result[expert_id] = stats.finalize()

    return result
