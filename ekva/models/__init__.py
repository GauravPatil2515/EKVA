"""Model registry — single source of truth for the 3 candidate models.

Used by every experiment script so model ids, expert counts, and VRAM guidance
live in one place. VRAM figures are *approximate fp16 weights* and assume the
plan's hardware (RTX 3050 6GB laptop, Colab T4 15GB, Colab A100 40GB).
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelSpec:
    key: str
    hf_id: str
    num_experts: int
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    approx_vram_gb_fp16: float
    recommended_hw: str
    top_k: int = 2
    notes: str = ""

    @property
    def download_url(self) -> str:
        # Hugging Face model hub page (weights pulled via `transformers` or git).
        return f"https://huggingface.co/{self.hf_id}"


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "qwen1.5-moe-a2.7b": ModelSpec(
        key="qwen1.5-moe-a2.7b",
        hf_id="Qwen/Qwen1.5-MoE-A2.7B",
        num_experts=60,  # 60 experts, 4 active per token
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        head_dim=128,
        approx_vram_gb_fp16=5.5,
        recommended_hw="RTX 3050 6GB (cpu offload) or Colab T4",
        top_k=4,
        notes="Smallest candidate; fits the 3050. Do this first (Week 1).",
    ),
    "mixtral-8x7b": ModelSpec(
        key="mixtral-8x7b",
        hf_id="mistralai/Mixtral-8x7B-v0.1",
        num_experts=8,
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        head_dim=128,
        approx_vram_gb_fp16=15.0,
        recommended_hw="Colab A100 40GB",
        top_k=2,
        notes="Standard 8-expert sparse MoE; primary results model.",
    ),
    "deepseek-moe-16b": ModelSpec(
        key="deepseek-moe-16b",
        hf_id="deepseek-ai/deepseek-moe-16b-base",
        num_experts=64,  # 2 shared + 62 routed, 6 active
        num_layers=28,
        hidden_size=2048,
        num_attention_heads=16,
        head_dim=128,
        approx_vram_gb_fp16=32.0,
        recommended_hw="Colab A100 40GB",
        top_k=6,
        notes="Falls back to this if DeepSeek-V2 (236B) is too heavy.",
    ),
}


def get_model_spec(key: str) -> ModelSpec:
    if key not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{key}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[key]


def list_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())
