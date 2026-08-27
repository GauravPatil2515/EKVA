# EKVA: Entropy-Guided Key-Value Allocation for Sparse MoE Inference

## Abstract
(To be written after all results are in)

## 1. Introduction
- KV cache bottleneck in long-context MoE inference
- Expert-level vs layer-level KV budgeting gap
- EKVA: per-expert entropy-guided allocation + custom Triton kernels

## 2. Background
- 2.1 Mixture-of-Experts (MoE) architecture
- 2.2 KV cache in transformer inference  
- 2.3 Attention entropy as informativeness signal
- 2.4 Roofline model for operator-level performance

## 3. Method
- 3.1 Calibration: per-expert entropy + routing frequency
- 3.2 Budget derivation: proportional allocation
- 3.3 Eviction strategies: recency, attention-score, hybrid
- 3.4 Triton kernel: variable tile count per expert

## 4. Research Questions & Results
- 4.1 RQ1: Expert vs Layer granularity
- 4.2 RQ2: Entropy-routing correlation across 8/60/64 experts
- 4.3 RQ3: Policy x Eviction grid (10-80% budgets)
- 4.4 RQ4: Per-expert roofline + Triton kernel speedup

## 5. Discussion
- When entropy signal is weak (Mixtral, 8 experts)
- Load-balancing loss interference hypothesis
- Limitations: synthetic data for some models on free tier

## 6. Related Work
- KV cache compression: SnapKV, PyramidKV, DynamicKV, CAKE, InfoKV
- MoE serving: WiSP, FluxMoE, TriRoute
- Hardware-aware: roofline, Triton kernel optimization

## 7. Conclusion
- Empirical evidence for per-expert KV budgeting in MoE
- Entropy-routing correlation varies by architecture
- Software + hardware co-design delivers measurable gains

## Reproducibility
Code: https://github.com/your-org/EKVA
Run free-tier pipeline: python experiments/colab/free_tier_notebook.py
