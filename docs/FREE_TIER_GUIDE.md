# EKVA $0 Free-Tier Execution Guide
### Zero-Cost Experimental Execution Strategy for RTX 3050 & Free Colab T4

This guide outlines how to execute the entire EKVA experimental validation on **real pretrained weights (Qwen1.5-MoE-A2.7B & Mixtral-8x7B)** with **$0 total compute cost**.

---

## 1. Zero-Cost Hardware Mapping

| Model / Experiment | Quantization | Memory Required | Platform | Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen1.5-MoE-A2.7B**<br>*(60 experts, 24 layers)* | 4-bit (`bitsandbytes`) | **~2.2 GB VRAM** | **Local RTX 3050 (6GB)** or **Free Colab T4 (15GB)** | **$0.00** |
| **Mixtral-8x7B**<br>*(8 experts, 32 layers)* | 4-bit NF4 / AWQ | **~12.5 GB VRAM** | **Google Colab Free (T4 15GB)** | **$0.00** |
| **DeepSeek-MoE-16B** | Simulation track / 4-bit offload | ~8 GB RAM / CPU | **Local CPU / Free Colab** | **$0.00** |
| **Analytical Roofline & Systems** | Theoretical FLOPs/Byte | Minimal | **Local RTX 3050 / CPU** | **$0.00** |

---

## 2. Quickstart Execution (100% Free)

### Step 1: Local RTX 3050 Run (Qwen1.5-MoE 4-bit)
```bash
# 1. Install 4-bit acceleration
pip install bitsandbytes accelerate

# 2. Run real calibration on Qwen with 4-bit quantization (takes ~2.2 GB VRAM)
python experiments/week01_02_calibration.py \
    --model qwen1.5-moe-a2.7b \
    --device cuda \
    --load-in-4bit \
    --prompt-sets general code math

# 3. Run RQ1 & RQ3 evaluations
python experiments/rq1_granularity_and_ablation.py
python experiments/rq3_transferability_reasoning.py
```

### Step 2: Google Colab Free Tier (T4 15GB GPU)
For Mixtral-8x7B 4-bit calibration on Colab Free T4:
```bash
# In Colab notebook with GPU (T4) runtime:
!git clone https://github.com/GauravPatil2515/EKVA.git
%cd EKVA
!pip install -e .
!pip install transformers accelerate bitsandbytes scipy

!python experiments/colab/free_tier_pipeline.py --model mixtral-8x7b --load-in-4bit
```

---

## 3. Scope Protection
- **RQ1 (Granularity) & RQ3 (Transferability/Reasoning Stability):** Primary empirical claims evaluated on real checkpoints.
- **RQ4 (Systems/Roofline):** Analytical FLOPs/Byte roofline + Triton v1 decode speedup.
- **Total Compute Budget:** **$0.00**.
