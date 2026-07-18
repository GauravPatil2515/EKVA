# Model Downloads & Hardware Guidance

EKVA targets three sparse-MoE models. Weights are pulled from the Hugging Face
Hub (not bundled in the repo). Approximate VRAM is for **fp16 weights only**,
excluding activation/KV overhead.

## 1. Qwen1.5-MoE-A2.7B  (do this FIRST — fits the 3050 / Colab T4)
- Hub page: https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
- Experts: 60 total, 4 active per token. Layers: 24.
- Approx fp16 VRAM: **~5.5 GB** (fits RTX 3050 6GB with cpu_offload, or Colab T4 15GB).
- Download:
  ```bash
  # via transformers (auto-cached in ~/.cache/huggingface)
  python3 -c "from transformers import AutoModelForCausalLM; \
    AutoModelForCausalLM.from_pretrained('Qwen/Qwen1.5-MoE-A2.7B')"
  # or git (full repo, slower):
  git lfs install
  git clone https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B
  ```

## 2. Mixtral-8x7B  (primary results model — Colab A100 40GB)
- Hub page: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
- Experts: 8, 2 active per token. Layers: 32.
- Approx fp16 VRAM: **~15 GB** (fits A100 40GB comfortably; T4 15GB is tight).
- Download:
  ```bash
  git clone https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
  ```

## 3. DeepSeek-MoE-16B  (fallback if DeepSeek-V2 / 236B is too heavy — Colab A100)
- Hub page: https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
- Experts: 64 (2 shared + 62 routed), 6 active per token. Layers: 28.
- Approx fp16 VRAM: **~32 GB** (needs A100 40GB; use `--load-in-8bit` / cpu_offload if short).
- If you specifically need **DeepSeek-V2** (236B), rent an H100 (80GB) or use 4-bit AWQ/GPTQ.
- Download:
  ```bash
  git clone https://huggingface.co/deepseek-ai/deepseek-moe-16b-base
  ```

## Hardware summary
| Model | Best HW | Approx fp16 VRAM |
|---|---|---|
| Qwen1.5-MoE-A2.7B | RTX 3050 6GB (offload) / T4 | 5.5 GB |
| Mixtral-8x7B | Colab A100 40GB | 15 GB |
| DeepSeek-MoE-16B | Colab A100 40GB | 32 GB |

## Notes
- The model registry (`ekva/models/registry.py`) and `configs/models.yaml` are the
  single source of truth for ids / expert counts. Update both if you add a model.
- Weights are git-ignored (`/models/` at repo root in `.gitignore`) — never
  commit them. Note `ekva/models/` is the code registry and IS tracked.
- For the 3050 laptop, use `device=cpu` or `device=cuda` with `--load-in-8bit` and
  small calibration prompt sets; reserve real training/benchmark runs for Colab.
