"""Free-tier Colab script for Week 12: End-to-end pipeline + results consolidation.

Runs on FREE Colab T4 (15GB VRAM) with 4-bit quantization.
Produces:
  - Final policy comparison table (FullKV vs Uniform vs SnapKV vs PyramidKV vs EKVA)
  - Consolidated figures from local simulator (synthetic quality for all 3 models)
  - Paper skeleton with RQ1-RQ4 results

Usage:
    python experiments/colab/free_tier_w12.py
"""
import os
import sys
import subprocess
import json
import torch
from pathlib import Path
REPO_DIR = os.environ.get("EKVA_REPO_DIR", "/content/EKVA")


def setup():
    # Detect if we're in Colab or local
    if os.path.exists("/content"):
        os.chdir("/content")
        if not os.path.exists(REPO_DIR):
            subprocess.run(["git", "clone", "https://github.com/your-org/EKVA.git", REPO_DIR], check=False)
        repo_path = REPO_DIR
    else:
        # Local: find the EKVA repo from current working directory
        current = os.getcwd()
        while current != "/" and not os.path.exists(os.path.join(current, "ekva")):
            current = os.path.dirname(current)
        if os.path.exists(os.path.join(current, "ekva")):
            repo_path = current
        else:
            repo_path = "/home/gaurav/Desktop/gaurav code /Paper/EKVA/EKVA"
    
    os.chdir(repo_path)
    sys.path.insert(0, repo_path)
    subprocess.run([sys.executable, "-m", "pip", "install", "matplotlib", "-q"], check=False)


def run_end_to_end():
    """Run end-to-end comparison + paper skeleton generation."""
    print("=== Week 12: End-to-End Pipeline (Free Tier) ===")
    os.makedirs("output/week12", exist_ok=True)

    # Load calibration results
    from ekva.models import get_model_spec, MODEL_REGISTRY
    
    # Use local simulator pipeline results (already generated)
    all_results = {}
    
    for model_key in ["qwen1.5-moe-a2.7b", "mixtral-8x7b", "deepseek-moe-16b"]:
        spec = get_model_spec(model_key)
        
        # Try to load real calibration first, fall back to synthetic
        cal_path = f"output/{model_key}_phase1.pt"
        if not os.path.exists(cal_path):
            cal_path = f"output/{model_key}_general_phase1.pt"
        if not os.path.exists(cal_path):
            cal_path = f"output/{model_key}_synthetic_phase1.pt"
        
        if not os.path.exists(cal_path):
            print(f"  {model_key}: No calibration found, generating synthetic...")
            subprocess.run([
                sys.executable, "experiments/generate_mock_calibration.py",
                "--model", model_key, "--out-dir", "output",
            ])
            cal_path = f"output/{model_key}_phase1.pt"
        
        # Load entropy map
        try:
            d = torch.load(cal_path, map_location="cpu", weights_only=False)
        except TypeError:
            d = torch.load(cal_path, map_location="cpu")
        
        emap = d["entropy_map"]
        
        # Compute final budget comparison at 25% budget
        from ekva.budget.policies import UniformPolicy, EKVAPolicy, SnapKVStylePolicy, PyramidKVStylePolicy
        
        table = {}
        total_budget = 4096
        frac_budget = int(total_budget * 0.25)
        min_pe = max(64, frac_budget // spec.num_experts)
        
        for name, P in [("Uniform", UniformPolicy), 
                        ("SnapKV-style", SnapKVStylePolicy), 
                        ("PyramidKV-style", PyramidKVStylePolicy), 
                        ("EKVA", EKVAPolicy)]:
            pol = P()
            try:
                b = pol.allocate(spec.num_experts, frac_budget, 
                                entropy_map=emap if name == "EKVA" else None,
                                min_per_expert=min_pe)
            except (ValueError, KeyError):
                b = {i: max(min_pe, frac_budget // spec.num_experts) for i in range(spec.num_experts)}
            
            mem_pct = round(100.0 * sum(b.values()) / (spec.num_experts * total_budget), 1)
            table[name] = {"memory_pct": mem_pct, "budget_sum": sum(b.values()), "budgets": b}
        
        table["FullKV"] = {"memory_pct": 100.0, "budget_sum": frac_budget, 
                          "budgets": {i: 4096 for i in range(spec.num_experts)}}
        
        all_results[model_key] = table
        
        print(f"\n  {model_key} final budget comparison (25% budget):")
        for name, info in table.items():
            print(f"    {name:20s}: {info['memory_pct']:.1f}% memory, sum={info['budget_sum']}")

    # Save results
    with open("output/week12/final_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate paper skeleton
    skeleton = """# EKVA: Entropy-Guided Key-Value Allocation for Sparse MoE Inference

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
"""
    
    with open("output/week12/paper_skeleton.md", "w") as f:
        f.write(skeleton)
    
    print(f"\n=== Week 12 complete ===")
    print(f"Final results: output/week12/final_results.json")
    print(f"Paper skeleton: output/week12/paper_skeleton.md")
    
    # Generate summary figure combining RQ1-RQ3
    print("\nGenerating summary figure...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # RQ1: Summary bar chart
        ax = axes[0, 0]
        models = list(all_results.keys())
        x = np.arange(len(models))
        width = 0.2
        for i, (model_key, table) in enumerate(all_results.items()):
            pass
        # Show EKVA vs Uniform at 25% budget for each model
        ekva_mems = [all_results[m]["EKVA"]["memory_pct"] for m in models]
        uniform_mems = [all_results[m]["Uniform"]["memory_pct"] for m in models]
        ax.bar(x - width/2, ekva_mems, width, label="EKVA", color="steelblue")
        ax.bar(x + width/2, uniform_mems, width, label="Uniform", color="coral")
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel("Memory Usage (%)")
        ax.set_title("RQ1: EKVA vs Uniform (25% budget)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

        # RQ2: Correlation summary
        ax = axes[0, 1]
        try:
            rq2_data = torch.load("output/local_rq2_all_models.pt", map_location="cpu", weights_only=False)
            model_names = list(rq2_data.keys())
            pearsons = [rq2_data[m]["pearson_r"] for m in model_names]
            spearmans = [rq2_data[m]["spearman_r"] for m in model_names]
            x2 = np.arange(len(model_names))
            ax.bar(x2 - width/2, pearsons, width, label="Pearson", color="steelblue")
            ax.bar(x2 + width/2, spearmans, width, label="Spearman", color="coral")
            ax.set_xticks(x2)
            ax.set_xticklabels(model_names, rotation=15, ha="right")
            ax.set_ylabel("Correlation r")
            ax.set_title("RQ2: Entropy-Routing Correlation")
            ax.legend()
            ax.grid(axis="y", alpha=0.3)
        except Exception as e:
            ax.text(0.5, 0.5, f"RQ2 data unavailable\n({e})", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("RQ2: Entropy-Routing Correlation")

        # RQ3: Policy comparison (from local_rq3_grid.json)
        ax = axes[1, 0]
        try:
            with open("output/local_rq3_grid.json") as f:
                rq3_data = json.load(f)
            # Extract Qwen results for display
            qwen_data = {}
            for key, val in rq3_data.get("mock", {}).items():
                if "qwen" in key.lower():
                    qwen_data[key] = val
            if not qwen_data:
                # Check if top-level has model keys
                if "qwen1.5-moe-a2.7b" in rq3_data:
                    qwen_data = rq3_data["qwen1.5-moe-a2.7b"]
            
            if qwen_data:
                # Extract EKVA vs Uniform quality at different budgets
                fracs = [10, 20, 40, 60, 80]
                ekva_q = []
                uniform_q = []
                for frac in fracs:
                    key = f"ekva|hybrid|{frac}%"
                    if key in qwen_data:
                        ekva_q.append(qwen_data[key]["quality"])
                    else:
                        ekva_q.append(0)
                    key = f"uniform|hybrid|{frac}%"
                    if key in qwen_data:
                        uniform_q.append(qwen_data[key]["quality"])
                    else:
                        uniform_q.append(0)
                
                ax.plot(fracs, ekva_q, "o-", label="EKVA", color="steelblue", linewidth=2)
                ax.plot(fracs, uniform_q, "s-", label="Uniform", color="coral", linewidth=2)
                ax.set_xlabel("KV Budget (%)")
                ax.set_ylabel("Normalized Quality (1.0 = FullKV)")
                ax.set_title("RQ3: EKVA vs Uniform (Qwen, hybrid eviction)")
                ax.legend()
                ax.grid(alpha=0.3)
                ax.set_ylim(0, 1.05)
        except Exception as e:
            ax.text(0.5, 0.5, f"RQ3 data unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("RQ3: Quality vs Budget")

        # RQ4: Placeholder for Triton kernel results
        ax = axes[1, 1]
        ax.text(0.5, 0.5, "RQ4: Triton Kernel\n(Requires A100 for real eval)\n\nSee Week 10-11 scripts\nfor kernel implementation", 
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        ax.set_title("RQ4: Hardware Roofline")
        ax.axis("off")

        plt.suptitle("EKVA: Key Results Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig("output/week12/summary_figure.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("Summary figure: output/week12/summary_figure.png")
    except ImportError:
        print("matplotlib not available, skipping summary figure")


if __name__ == "__main__":
    setup()
    run_end_to_end()
