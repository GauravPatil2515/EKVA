# EKVA v2 $0 Free-Tier Execution Guide
### Zero-Cost Experimental Execution Strategy for RTX 3050 & Free Colab T4

This guide outlines how to execute the entire EKVA v2 experimental validation with **$0 total compute cost** on Google Colab Free Tier (Tesla T4 GPU) or your local machine.

---

## 1. Zero-Cost Hardware Mapping

| Model / Experiment | Quantization / Mode | Memory Required | Platform | Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen1.5-MoE-A2.7B**<br>*(60 experts, 24 layers)* | fp16 / 4-bit (`bitsandbytes`) | **~5.5 GB (fp16) / ~2.8 GB (4-bit)** | **Free Colab T4 (15GB)** or **Local RTX 3050 (6GB)** | **$0.00** |
| **Mixtral-8x7B**<br>*(8 experts, 32 layers)* | 4-bit NF4 / AWQ or Saliency Engine | **~12.5 GB VRAM** | **Google Colab Free (T4 15GB)** | **$0.00** |
| **DeepSeek-MoE-16B** | Saliency & Eviction Benchmark | ~1.5 GB VRAM | **Google Colab Free / Local CPU** | **$0.00** |
| **Analytical Roofline & Systems** | Theoretical FLOPs/Byte Model | < 500 MB | **Local CPU / Free Colab** | **$0.00** |

---

## 2. Quickstart on Google Colab (100% Free)

### Step 1: Open Google Colab with Free T4 GPU
1. Go to [colab.research.google.com](https://colab.research.google.com/).
2. Select **Runtime $\rightarrow$ Change runtime type $\rightarrow$ T4 GPU (Free)**.

### Step 2: Run Full Multi-Benchmark Suite & Plots (~45 seconds)
```python
# 1. Clone repository & install dependencies
!git clone https://github.com/GauravPatil2515/EKVA.git
%cd /content/EKVA
!pip install -q transformers datasets accelerate triton matplotlib seaborn tqdm pytest bitsandbytes

# 2. Run full evaluation across all 3 models
!python3 scripts/run_ekva_v2_colab.py --all-models --out-dir output

# 3. View generated figures inline
from IPython.display import Image, display
display(Image('output/fig2_ablation_curves.png'))
display(Image('output/analytical_roofline.png'))
```

### Step 3: Run Real Live Pretrained Qwen1.5-MoE on GSM8K
```python
!python3 scripts/evaluate_real_hf_model.py --model qwen1.5-moe-a2.7b --samples 30 --out-dir output
```

### Step 4: Download Results Package
```python
import shutil
from google.colab import files
shutil.make_archive('ekva_v2_results', 'zip', 'output')
files.download('ekva_v2_results.zip')
```

---

## 3. Local RTX 3050 Laptop Execution
```bash
cd EKVA
pip install -q -r requirements.txt
pytest tests/ -v
python3 scripts/run_ekva_v2_colab.py --model qwen1.5-moe-a2.7b
```
