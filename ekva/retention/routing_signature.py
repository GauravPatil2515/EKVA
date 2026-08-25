"""EKVA v2: Routing Signature Data Structures and Extraction Hooks.

Extracts cross-layer routing signatures R_t = {E_t^{(1)}, ..., E_t^{(L)}} from
sparse Mixture-of-Experts (MoE) models (Mixtral, Qwen1.5-MoE, DeepSeek-MoE)
during forward passes without architectural modifications.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn


@dataclass
class RoutingSignature:
    """Stores the routing history of tokens across transformer layers."""
    # (batch_size, seq_len, num_layers, top_k)
    expert_indices: torch.Tensor
    # (batch_size, seq_len, num_layers, top_k) optional router probabilities
    routing_weights: Optional[torch.Tensor] = None

    @property
    def seq_len(self) -> int:
        return self.expert_indices.shape[1]

    @property
    def num_layers(self) -> int:
        return self.expert_indices.shape[2]

    @property
    def top_k(self) -> int:
        return self.expert_indices.shape[3]


class MoERoutingHook:
    """Forward hook context manager to capture routing decisions across layers."""

    def __init__(self, model: nn.Module, model_family: str = "auto"):
        self.model = model
        self.model_family = model_family.lower()
        self.hooks = []
        self.captured_routing: Dict[int, torch.Tensor] = {}
        self.captured_weights: Dict[int, torch.Tensor] = {}

    def _detect_family(self) -> str:
        name = self.model.__class__.__name__.lower()
        if "mixtral" in name:
            return "mixtral"
        elif "qwen" in name:
            return "qwen"
        elif "deepseek" in name:
            return "deepseek"
        return "generic_moe"

    def __enter__(self):
        family = self._detect_family() if self.model_family == "auto" else self.model_family
        self.captured_routing.clear()
        self.captured_weights.clear()

        layer_idx = 0
        for name, module in self.model.named_modules():
            # Match router/gate modules in popular MoE implementations
            is_router = any(k in name.lower() for k in ["gate", "router", "block_sparse_moe.gate"])
            if is_router and hasattr(module, "forward"):
                idx = layer_idx
                
                def make_hook(l_idx):
                    def hook_fn(mod, inp, out):
                        # out is typically (router_logits, selected_experts) or tuple
                        if isinstance(out, tuple):
                            if len(out) >= 2 and isinstance(out[1], torch.Tensor):
                                self.captured_routing[l_idx] = out[1].detach().cpu()
                            if len(out) >= 1 and isinstance(out[0], torch.Tensor):
                                self.captured_weights[l_idx] = torch.softmax(out[0], dim=-1).detach().cpu()
                        elif isinstance(out, torch.Tensor):
                            # Router logits: take topk
                            topk_w, topk_i = torch.topk(torch.softmax(out, dim=-1), k=min(4, out.shape[-1]), dim=-1)
                            self.captured_routing[l_idx] = topk_i.detach().cpu()
                            self.captured_weights[l_idx] = topk_w.detach().cpu()
                    return hook_fn

                h = module.register_forward_hook(make_hook(idx))
                self.hooks.append(h)
                layer_idx += 1

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_signature(self) -> RoutingSignature:
        """Assembles captured layer routing into a unified RoutingSignature tensor."""
        if not self.captured_routing:
            raise RuntimeError("No routing decisions captured. Ensure model was called during hook context.")

        sorted_layers = sorted(self.captured_routing.keys())
        stacked_indices = [self.captured_routing[l] for l in sorted_layers]
        
        # Format tensors into (batch, seq, num_layers, top_k)
        if stacked_indices[0].dim() == 2:  # (seq_len, top_k)
            stacked_indices = [t.unsqueeze(0) for t in stacked_indices]
        
        # (batch, seq_len, num_layers, top_k)
        expert_indices = torch.stack(stacked_indices, dim=2)
        
        expert_weights = None
        if self.captured_weights:
            stacked_w = [self.captured_weights[l] for l in sorted_layers]
            if stacked_w[0].dim() == 2:
                stacked_w = [t.unsqueeze(0) for t in stacked_w]
            expert_weights = torch.stack(stacked_w, dim=2)

        return RoutingSignature(expert_indices=expert_indices, routing_weights=expert_weights)
