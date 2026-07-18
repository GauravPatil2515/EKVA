# Kernel & Roofline (Weeks 8–11)

The Triton kernel and roofline profiling require **`triton` + a CUDA GPU**
(Colab A100 / H100). They will NOT run on the RTX 3050 laptop and are
intentionally excluded from the CPU test track.

## Files
- `ekva/kernel/reference_flashattn2.py` — standard FA2 tiling (baseline).
- `ekva/kernel/ekva_triton_v1.py` — variable KV tile count per expert.
- `ekva/kernel/ekva_triton_v2.py` — v1 + fused important-token index selection.
- `ekva/profiling/instrument.py` — PyTorch Profiler + roofline math + Nsight JSON parse.

## Week 10 — kernel v1 (variable tile)
```bash
# on Colab A100
pip install triton
python3 experiments/week10_kernel_v1.py --device cuda
```
Correctness gate: `ekva_attention_v1(q,k,v, kv_budget=N)` must match exact
attention within fp16 tolerance. Then benchmark kernel time vs baseline FA2.

## Week 11 — kernel v2 (fused eviction) + Nsight
```bash
python3 experiments/week11_kernel_v2.py --device cuda
# wrap with Nsight Compute for HBM/occupancy/timing:
ncu --set full --export output/week11/ekva_v2.ncu-rep \
    python3 experiments/week11_kernel_v2.py --device cuda
```
Compare baseline vs v2 per expert class; expect low-entropy experts to shift
closer to the memory bandwidth ceiling.

## Weeks 8–9 — roofline
```bash
python3 experiments/week08_09_roofline.py --model mixtral-8x7b \
    --calibration output/mixtral-8x7b_general_phase1.pt
```
Uses PyTorch Profiler (CPU activity shown; add `ProfilerActivity.CUDA` on GPU)
and `compute_roofline()` to derive Arithmetic Intensity = FLOPs / Bytes.
Replace the placeholder FLOPs/bytes with values parsed from the profiler or
Nsight (`parse_nsight_json`).

## GPU ceilings (example A100)
- Peak fp16: ~312 TFLOP/s
- Peak HBM bandwidth: ~1.55 TB/s
Set these in `week08_09_roofline.py` (`PEAK_GFLOPS`, `PEAK_BANDWIDTH_GBS`) to
match your actual rented GPU.

## If the kernel is too hard
Fallback (see `PLAN.md`): report software-simulator results only and describe
the kernel as a "proposed implementation." The algorithmic contribution stands
alone.
