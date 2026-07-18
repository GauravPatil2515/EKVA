"""Mock calibration generator (no real model / no transformers needed).

Produces output/{model}_phase1.pt using a tiny synthetic MoE so the rest of the
pipeline (plots, simulator, week scripts) can be developed on the 3050 without
downloading weights. Replace with experiments/week01_02_calibration.py on GPU.

Usage:
  python experiments/generate_mock_calibration.py --model mixtral-8x7b
"""
import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.calibration.entropy import calibrate_expert_entropy
from ekva.budget.derive import derive_kv_budget
from ekva.models import get_model_spec


class _MockCfg:
    output_attentions = False


class _MockAttn(nn.Module):
    def __init__(self, heads=4):
        super().__init__()
        self.heads = heads

    def forward(self, h, *a, **k):
        b, s, _ = h.shape
        w = torch.softmax(torch.randn(b, self.heads, s, s), dim=-1)
        return torch.randn_like(h), w, None


class _MockMoE(nn.Module):
    def __init__(self, n_experts=8):
        super().__init__()
        self.n_experts = n_experts

    def forward(self, h, *a, **k):
        b, s, _ = h.shape
        logits = torch.randn(b * s, self.n_experts)
        return torch.randn_like(h), logits


class _MockLayer(nn.Module):
    def __init__(self, n_experts=8):
        super().__init__()
        self.self_attn = _MockAttn()
        self.block_sparse_moe = _MockMoE(n_experts)

    def forward(self, h, *a, **k):
        o, _, _ = self.self_attn(h)
        m, _ = self.block_sparse_moe(o)
        return m


class _MockModel(nn.Module):
    def __init__(self, layers=4, n_experts=8):
        super().__init__()
        self.config = _MockCfg()
        self.device = torch.device("cpu")
        self.layers = nn.ModuleList([_MockLayer(n_experts) for _ in range(layers)])

    def forward(self, input_ids, **k):
        b, s = input_ids.shape
        h = torch.randn(b, s, 128)
        for layer in self.layers:
            h = layer(h)
        return h


class MockBatchEncoding(dict):
    """Minimal stand-in for transformers.BatchEncoding with a .to()."""
    def to(self, device):
        return MockBatchEncoding(
            {k: v.to(device) if hasattr(v, "to") else v for k, v in self.items()}
        )


class _Tok:
    def __call__(self, prompt, return_tensors="pt", **k):
        s = max(5, len(prompt.split()) * 2)
        return MockBatchEncoding({"input_ids": torch.randint(0, 1000, (1, s))})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mixtral-8x7b")
    ap.add_argument("--out-dir", default="output")
    args = ap.parse_args()
    spec = get_model_spec(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    model = _MockModel(layers=spec.num_layers, n_experts=spec.num_experts)
    prompts = ["Explain attention in transformers.", "Summarize LLMs.",
               "Write a factorial function.", "Prove even+even is even."]
    emap = calibrate_expert_entropy(model=model, tokenizer=_Tok(), calibration_prompts=prompts,
                                    num_experts=spec.num_experts)
    budget = derive_kv_budget(emap, total_budget=2048, strategy="proportional")
    out = Path(args.out_dir) / f"{args.model}_phase1.pt"
    torch.save({"entropy_map": emap, "budget_tensor": budget,
                "meta": {"model": args.model, "mock": True}}, out)
    print(f"[mock] Saved {out}; budget_sum={int(budget.sum())}")


if __name__ == "__main__":
    main()
