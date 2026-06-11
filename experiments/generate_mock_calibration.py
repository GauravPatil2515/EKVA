import os
import sys
import torch
import torch.nn as nn
from typing import Tuple, Dict, List

# Ensure we can import from the root package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.calibration import calibrate_expert_entropy, ExpertStats
from ekva.budget import derive_kv_budget


class MockConfig:
    def __init__(self):
        self.output_attentions = False


class MockAttention(nn.Module):
    def __init__(self, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, None]:
        batch_size, seq_len, _ = hidden_states.shape
        # Mock attention weights: (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.softmax(torch.randn(batch_size, self.num_heads, seq_len, seq_len), dim=-1)
        # Mock attention output
        attn_output = torch.randn_like(hidden_states)
        return attn_output, attn_weights, None


class MockMoE(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.num_experts = num_experts

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        # Mock router logits: (batch_size * seq_len, num_experts)
        router_logits = torch.randn(batch_size * seq_len, self.num_experts)
        # Mock final outputs
        output = torch.randn_like(hidden_states)
        return output, router_logits


class MockDecoderLayer(nn.Module):
    def __init__(self, num_experts: int = 8):
        super().__init__()
        self.self_attn = MockAttention()
        self.block_sparse_moe = MockMoE(num_experts=num_experts)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        attn_output, _, _ = self.self_attn(hidden_states)
        moe_output, _ = self.block_sparse_moe(attn_output)
        return moe_output


class MockModel(nn.Module):
    def __init__(self, num_layers: int = 4, num_experts: int = 8):
        super().__init__()
        self.config = MockConfig()
        self.device = torch.device("cpu")
        self.layers = nn.ModuleList([MockDecoderLayer(num_experts=num_experts) for _ in range(num_layers)])

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        hidden_states = torch.randn(batch_size, seq_len, 128)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class MockBatchEncoding(dict):
    def to(self, device):
        return MockBatchEncoding({k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()})


class MockTokenizer:
    def __call__(self, prompt: str, return_tensors: str = "pt", **kwargs):
        seq_len = max(5, len(prompt.split()) * 2)
        input_ids = torch.randint(0, 1000, (1, seq_len))
        return MockBatchEncoding({"input_ids": input_ids})


def main():
    print("[EKVA Mock] Setting up mock model and tokenizer...")
    model = MockModel(num_layers=4, num_experts=8)
    tokenizer = MockTokenizer()
    
    calibration_prompts = [
        "Explain the concept of attention in transformers in simple terms.",
        "Summarize the following article about large language models.",
        "Write a Python function to compute the factorial of a number.",
        "Prove that the sum of two even integers is even.",
    ]
    
    print("[EKVA Mock] Running calibration pass...")
    entropy_map = calibrate_expert_entropy(
        model=model,
        tokenizer=tokenizer,
        calibration_prompts=calibration_prompts,
        num_experts=8,
        max_new_tokens=0,
    )
    
    print("[EKVA Mock] Deriving KV budgets...")
    budget_tensor = derive_kv_budget(
        entropy_map=entropy_map,
        total_budget=2048,
        min_per_expert=64,
        strategy="proportional",
    )
    
    payload = {
        "model": "mistralai/Mixtral-8x7B-v0.1",
        "num_experts": 8,
        "total_budget": 2048,
        "min_per_expert": 64,
        "max_new_tokens": 0,
        "entropy_map_keys": sorted(list(entropy_map.keys())),
        "budget": budget_tensor.tolist(),
    }
    
    os.makedirs("output", exist_ok=True)
    out_path = "output/mixtral_phase1.pt"
    
    print(f"[EKVA Mock] Saving results to {out_path}...")
    torch.save({
        "entropy_map": entropy_map,
        "budget_tensor": budget_tensor,
        "meta": payload
    }, out_path)
    
    print("[EKVA Mock] Finished generating mock calibration file!")


if __name__ == "__main__":
    main()
