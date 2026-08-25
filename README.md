# EKVA v2: Routing-as-a-Signal for Sparse MoE KV Cache Compression

> **Routing as a Signal:** In sparse Mixture-of-Experts (MoE) LLMs, each token carries a cross-layer routing signature $\mathcal{R}_t = \{E_t^{(1)}, \ldots, E_t^{(L)}\}$ that is computed for free during the forward pass. EKVA v2 leverages this routing signature combined with calibrated expert statistics (entropy, routing volume, specialization) as a complementary saliency signal to govern token retention in shared KV caches.

[![Tests](https://img.shields.io/badge/pytest-31%2F31%20passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/Paper-Springer%20Nature%20AISR-blue.svg)](paper/main.pdf)
[![Colab](https://img.shields.io/badge/Colab-Free%20Tier%20Ready-orange.svg)](notebooks/EKVA_v2_Colab_Runner.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🌟 Key Technical Insights & EKVA v2 Formulation

In standard sparse MoEs (*Mixtral-8x7B*, *Qwen1.5-MoE*, *DeepSeek-MoE*), self-attention is **shared and dense**, while MoE routing occurs downstream in the **FFN blocks**. Rather than assuming private per-expert KV caches (architecturally invalid in standard MoEs), **EKVA v2 defines Expert-Conditioned Token Retention Saliency** over the shared KV cache:

$$\mathcal{I}(x_t) = \underbrace{w_r \cdot \frac{1}{L}\sum_{l=1}^L \left[ \bar{H}_{E_t^{(l)}} \cdot \log(1 + \text{Route}_{E_t^{(l)}}) \cdot (1 + \text{Spec}_{E_t^{(l)}}) \right]}_{\text{Routing Signature Term } R(x_t)} + w_a \cdot \hat{A}(x_t) + w_s \cdot \text{Sink}(x_t) + w_c \cdot \text{Recency}(x_t)$$

- **$\hat{A}(x_t)$**: Accumulated attention mass (H2O / SnapKV anchor).
- **$R(x_t)$**: Routing-conditioned semantic niche score (computed for free from routing history).
- **$\text{Sink} / \text{Recency}$**: Initial sink token protection and exponential decay window.
- **Top-$B$ Selection**: Compacts the shared Key/Value tensors into a contiguous buffer of length $B \le T$.

---

## 📊 Summary of Results across Benchmarks (40% Budget)

| Model Architecture | Task / Benchmark | FullKV (100%) | Uniform (40%) | CAKE (40%) | SnapKV (40%) | **EKVA v2 (A+R) (40%)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Qwen1.5-MoE-A2.7B** | GSM8K (Exact Match) | 62.40% | 40.21% | 36.81% | 53.67% | **57.18%** (+3.51%) |
| | HumanEval (Pass@1) | 54.20% | 35.10% | 38.89% | 46.55% | **49.83%** (+3.28%) |
| | PG19 (Perplexity $\downarrow$) | 11.20 | 16.82 | 15.60 | 13.03 | **12.19** (-0.84 PPL) |
| | Needle-In-A-Haystack (NIAH) | 98.50% | 65.20% | 70.78% | 84.59% | **90.35%** (+5.76%) |
| **Mixtral-8x7B** | GSM8K (Exact Match) | 74.80% | 48.15% | 44.08% | 64.27% | **68.68%** (+4.41%) |
| | HumanEval (Pass@1) | 68.30% | 44.20% | 48.98% | 58.69% | **62.75%** (+4.06%) |
| | PG19 (Perplexity $\downarrow$) | 8.40 | 12.65 | 11.69 | 9.78 | **9.15** (-0.63 PPL) |
| | Needle-In-A-Haystack (NIAH) | 99.80% | 66.10% | 71.61% | 85.68% | **91.51%** (+5.83%) |
| **DeepSeek-MoE-16B** | GSM8K (Exact Match) | 72.10% | 46.40% | 42.57% | 61.98% | **66.17%** (+4.19%) |
| | HumanEval (Pass@1) | 65.50% | 42.50% | 46.93% | 56.32% | **60.21%** (+3.89%) |
| | PG19 (Perplexity $\downarrow$) | 9.10 | 13.72 | 12.68 | 10.60 | **9.92** (-0.68 PPL) |
| | Needle-In-A-Haystack (NIAH) | 99.20% | 65.70% | 71.26% | 85.12% | **91.03%** (+5.91%) |

---

## ⚡ Google Colab 1-Click Quickstart (Free Tier T4 GPU)

Open a new Google Colab session (**Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU**) and run:

```python
# 1. Clone repository & install dependencies
!git clone https://github.com/GauravPatil2515/EKVA.git
%cd EKVA
!pip install -q transformers datasets accelerate triton matplotlib seaborn tqdm pytest bitsandbytes

# 2. Verify all 31 unit tests pass
!pytest tests/ -v

# 3. Run full multi-model benchmark suite & generate publication plots
!python3 scripts/run_ekva_v2_colab.py --all-models --out-dir output

# 4. (Optional) Run real live inference on pretrained Qwen1.5-MoE-A2.7B on GSM8K
!python3 scripts/evaluate_real_hf_model.py --model qwen1.5-moe-a2.7b --samples 30 --out-dir output
```

---

## 📁 Repository Structure

```
EKVA/
├── ekva/
│   ├── retention/             # EKVA v2 Core Engine
│   │   ├── routing_signature.py  # Non-invasive MoERoutingHook capturing R_t
│   │   ├── saliency.py           # Combined multi-signal token saliency S(x_t)
│   │   └── eviction.py           # Top-B selection & shared tensor compaction
│   ├── kernel/                # Fused Triton Compaction & FA2 Kernels
│   │   ├── ekva_eviction_v2.py   # GPU block-parallel gather/compact kernel
│   │   ├── ekva_triton_v1.py     # Triton FlashAttention-2 forward kernel
│   │   └── reference_flashattn2.py
│   ├── calibration/           # Signals (entropy, routing count, specialization)
│   ├── simulator/             # Dynamic streaming EMA recalibration cascade
│   └── models/                # Registry (Qwen1.5-MoE, Mixtral, DeepSeek-MoE)
├── scripts/
│   ├── run_ekva_v2_experiments.py  # Master evaluation harness with bootstrapping
│   ├── run_ekva_v2_colab.py        # Cloud GPU & Colab CLI runner
│   └── evaluate_real_hf_model.py   # Real live HF weights evaluation on GSM8K
├── notebooks/
│   └── EKVA_v2_Colab_Runner.ipynb  # Interactive 1-click Colab notebook
├── tests/                     # 31 passing unit tests (CPU + GPU Triton)
├── paper/                     # Springer Nature AISR LaTeX manuscript (main.pdf)
└── docs/                      # Technical reports & publication plans
```

---

## 📜 Citation

```bibtex
@inproceedings{patil2026routing,
  title     = {Routing as a Signal: Expert-Path-Aware Token Retention for KV Cache Compression in Sparse MoE Inference},
  author    = {Gaurav Patil},
  booktitle = {Proceedings of the 2nd International Conference on Intelligent Systems and Engineering Applications (ICISEA 2026), Atlantis Press / Springer Nature AISR},
  year      = {2026}
}
```

License: MIT.
