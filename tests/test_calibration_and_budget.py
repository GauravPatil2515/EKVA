import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Optional
import sys
import os

# Add the repository root to sys.path so we can import ekva
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.calibration import calibrate_expert_entropy
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
        # Make it a valid probability distribution along the last dimension
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
        # Mock router logits: (batch_size, seq_len, num_experts) or (batch_size * seq_len, num_experts)
        # Let's test the 2D flattened version as it is common in Hugging Face (e.g. Mixtral)
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
    def __init__(self, num_layers: int = 2, num_experts: int = 8):
        super().__init__()
        self.config = MockConfig()
        self.device = torch.device("cpu")
        
        # Build layers structure to mimic Mixtral/Llama
        self.layers = nn.ModuleList([MockDecoderLayer(num_experts=num_experts) for _ in range(num_layers)])

    def forward(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # Simple embeddings mock
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
        # Generate a mock sequence of tokens (length depends on the prompt length)
        seq_len = max(5, len(prompt.split()) * 2)
        input_ids = torch.randint(0, 1000, (1, seq_len))
        return MockBatchEncoding({"input_ids": input_ids})


def test_calibration_and_budget_pipeline():
    print("Initializing Mock Model and Tokenizer...")
    model = MockModel(num_layers=4, num_experts=8)
    tokenizer = MockTokenizer()

    calibration_prompts = [
        "Explain transformers to a 10-year-old.",
        "Write a Python function to compute the FFT.",
        "Prove that every finite group of prime order is cyclic.",
    ]

    print("Running calibrate_expert_entropy...")
    entropy_map = calibrate_expert_entropy(
        model=model,
        tokenizer=tokenizer,
        calibration_prompts=calibration_prompts,
        num_experts=8,
        max_new_tokens=0,
    )

    print("\nCalibration Results:")
    for expert_id, stats in entropy_map.items():
        print(f"Expert {expert_id}:")
        print(f"  Average Entropy per layer: {stats['avg_entropy'].tolist()}")
        print(f"  Routing Count: {stats['routing_count'].item()}")

    print("\nRunning derive_kv_budget...")
    budget_tensor = derive_kv_budget(
        entropy_map=entropy_map,
        total_budget=2048,
        min_per_expert=64,
        strategy="proportional",
    )

    print("\nResulting KV Budget Tensor:")
    print(budget_tensor)
    print(f"Total Budget Sum: {budget_tensor.sum().item()}")

    assert budget_tensor.shape == (8,)
    assert budget_tensor.sum().item() == 2048
    assert (budget_tensor >= 64).all()
    print("\nAll pipeline assertions passed successfully!")


if __name__ == "__main__":
    test_calibration_and_budget_pipeline()
