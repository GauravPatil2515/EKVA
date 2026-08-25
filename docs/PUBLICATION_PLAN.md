# EKVA v2 Publication Plan of Action & Research Execution Roadmap
### Target: 2nd International Conference on Intelligent Systems and Engineering Applications (ICISEA 2026 / MULTINOVA 2.0)
**Publisher:** Atlantis Press (Part of Springer Nature), *Advances in Intelligent Systems Research (AISR)* Series  
**Submission Deadline:** **10th September 2026**  
**Conference Dates:** 20th & 21st November 2026 (Online Mode)  
**Paper Title:** *"Routing as a Signal: Expert-Path-Aware Token Retention for KV Cache Compression in Sparse MoE Inference"*

---

## 1. Project Status: What Is Completed

| Component / Deliverable | Status | Location / Artifact |
| :--- | :---: | :--- |
| **EKVA v2 Retention Engine** | ✅ Done | `RoutingSignature`, `MoERoutingHook`, `combined_token_saliency`, `evict_shared_kv_cache` in `ekva/retention/`. |
| **Fused Triton Compaction Kernel** | ✅ Done | `triton_compact_kv_cache` in `ekva/kernel/ekva_eviction_v2.py`. |
| **Unit Test Suite** | ✅ Done | 31 passing unit tests on CPU + GPU (`pytest tests/ -v`). |
| **Cross-Signal Correlation Check** | ✅ Done | Proof that routing signature $R(x_t)$ is complementary/orthogonal ($\rho \approx 0.00$) to attention scores $\hat{A}(x_t)$. |
| **Multi-Benchmark Evaluation** | ✅ Done | Evaluation across GSM8K, HumanEval, PG19, and NIAH with 95% bootstrap confidence intervals in `output/ekva_v2_results.json`. |
| **Real Live Model Pipeline** | ✅ Done | Real forward hooks & generation on `Qwen1.5-MoE-A2.7B` on GSM8K in `scripts/evaluate_real_hf_model.py`. |
| **Analytical Roofline & Systems** | ✅ Done | GPU decode memory-bound verification and speedup analysis in `output/analytical_roofline.png`. |
| **Google Colab 1-Click Runner** | ✅ Done | Tested interactive runner in `notebooks/EKVA_v2_Colab_Runner.ipynb`. |
| **LaTeX Manuscript** | ✅ Done | 10-page Springer Nature `llncs` format paper with bibliography compiled in `paper/main.pdf`. |

---

## 2. Paper Structure & Section Outline

1. **Introduction:** Long-context KV memory bottlenecks, MoE architectural reality (shared attention vs sparse FFN), and the premise that token routing history carries valuable retention signals for free.
2. **Related Work:** Token eviction (StreamingLLM, H2O, SnapKV), Head/Layer allocation (Ada-KV, CAKE, InfoKV), and MoE systems (PiKV, TriRoute, DeepSeek MLA).
3. **Methodology:**
   - Formal definition of Routing Signature $\mathcal{R}_t = \{E_t^{(1)}, \ldots, E_t^{(L)}\}$.
   - Calibrated expert semantic niche stats: $\bar{H}_e, \text{Route}_e, \text{Spec}_e$.
   - Multi-signal saliency formula $S(x_t) = w_a \hat{A} + w_r R + w_s \text{Sink} + w_c \text{Recency}$.
   - Fused GPU compaction and chronological index selection.
4. **Experiments & Results:**
   - Multi-task retention quality vs budget on GSM8K, HumanEval, PG19, and NIAH.
   - Component ablation ($w_a, w_r, w_s, w_c$).
   - Cross-architecture mechanistic analysis (Qwen vs Mixtral vs DeepSeek).
   - Real model evaluation numbers.
5. **Systems Validation:**
   - Arithmetic intensity profile ($\text{AI} \approx 1.0$) confirming memory-bound decode attention.
   - Triton compaction latency and throughput speedups.
6. **Discussion & Conclusion.**

---

## 3. Pre-Submission Checklist

- [x] Codebase refactored to architecturally valid shared KV cache eviction.
- [x] Unit test suite 100% passing (31/31).
- [x] Colab free-tier pipeline operational.
- [x] Paper manuscript formatted to Springer Nature `llncs` 10-page limit.
- [ ] Final proofreading and author metadata check before Meteor portal submission.
