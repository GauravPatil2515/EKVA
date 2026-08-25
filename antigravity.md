# EKVA Project Context & Developer Reference Guide
**Author:** Gaurav Patil | **Repository:** [GitHub: GauravPatil2515/EKVA](https://github.com/GauravPatil2515/EKVA)  
**Target Venue:** ICISEA 2026 / Atlantis Press / Springer Nature AISR (Advances in Intelligent Systems Research Series)  

---

## 1. Project Mission & High-Level Concept

**EKVA (Expert-Aware Key-Value Allocation / Routing as a Signal)** is a calibration-guided, training-free framework for KV cache compression in sparse Mixture-of-Experts (MoE) Large Language Models.

### The Core Paradigm Shift (EKVA v2):
- **Problem:** In standard sparse MoE architectures (*Mixtral-8x7B*, *Qwen1.5-MoE*, *DeepSeek-MoE*), self-attention is shared and dense across the sequence length, while MoE routing occurs downstream in the Feed-Forward Network (FFN/MLP) blocks. MoE experts do not own distinct attention mechanisms.
- **Solution:** Instead of assuming private per-expert KV caches, EKVA v2 treats each token's **cross-layer routing signature** $\mathcal{R}_t = \{E_t^{(1)}, \dots, E_t^{(L)}\}$ as a **cheap, already-computed extra feature** in a multi-signal saliency engine that governs token retention over the **shared contiguous KV cache**:

$$\mathcal{I}(x_t) = w_r \cdot \underbrace{\frac{1}{L}\sum_{l=1}^L \left[ \bar{H}_{E_t^{(l)}} \cdot \log(1 + \text{Route}_{E_t^{(l)}}) \cdot (1 + \text{Spec}_{E_t^{(l)}}) \right]}_{R(x_t)} + w_a \cdot \hat{A}(x_t) + w_s \cdot \text{Sink}(x_t) + w_c \cdot \text{Recency}(x_t)$$

---

## 2. Supported Sparse MoE Architectures

1. **`Qwen/Qwen1.5-MoE-A2.7B`**: 60 routed experts, 4 active per token, 24 layers, 16 attention heads.
2. **`mistralai/Mixtral-8x7B-v0.1`**: 8 routed experts, 2 active per token, 32 layers, 32 attention heads.
3. **`deepseek-ai/deepseek-moe-16b-base`**: 64 total experts (2 shared + 62 routed, 6 active), 28 layers, 16 attention heads.

---

## 3. Key Modules on Disk

- **`ekva/retention/routing_signature.py`**: Contains `RoutingSignature` and `MoERoutingHook` for non-invasive router logits extraction.
- **`ekva/retention/saliency.py`**: Computes $R(x_t)$, $\hat{A}(x_t)$, $\text{Sink}(x_t)$, $\text{Recency}(x_t)$, and $S(x_t)$.
- **`ekva/retention/eviction.py`**: Selects top-$B$ indices chronologically and compacts the shared $(K, V)$ tensor.
- **`ekva/kernel/ekva_eviction_v2.py`**: Fused Triton GPU gather/compact kernel for zero-overhead token eviction.
- **`ekva/kernel/ekva_triton_v1.py`**: Triton FlashAttention-2 forward kernel.
- **`scripts/run_ekva_v2_experiments.py`**: Master evaluation suite across GSM8K, HumanEval, PG19, and NIAH with 95% bootstrap confidence intervals.
- **`scripts/evaluate_real_hf_model.py`**: Real live Hugging Face forward pass and GSM8K generation evaluation.
- **`notebooks/EKVA_v2_Colab_Runner.ipynb`**: 1-click Google Colab runner.
- **`paper/main.tex` & `paper/main.pdf`**: 10-page Springer Nature `llncs` format conference paper.

---

## 4. Key Verification & Test Commands

```bash
# Run unit test suite (31 tests)
pytest tests/ -v

# Run full multi-model evaluation pipeline (saves to output/)
python3 scripts/run_ekva_v2_colab.py --all-models --out-dir output

# Run real live inference on Qwen1.5-MoE on GSM8K
python3 scripts/evaluate_real_hf_model.py --model qwen1.5-moe-a2.7b --samples 30 --out-dir output

# Recompile paper PDF
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
