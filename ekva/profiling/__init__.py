"""Roofline instrumentation for EKVA (Weeks 8-9, 11).

  instrument.py — wrap per-expert attention calls with PyTorch Profiler / Nsight
  roofline.py    — convert profiler traces into per-expert FLOPs / bytes / AI

Arithmetic Intensity (AI) = FLOPs / Bytes moved from HBM.
Roofline ceiling: peak_GFLOPs and peak_bandwidth_GBs for the target GPU.

PyTorch Profiler is built into PyTorch; Nsight (ncu/nsys) runs on the host and
dumps JSON that roofline.py can parse. See docs/KERNEL.md.
"""
from typing import Dict, List, Optional

import torch


def profile_model_attention(model, calibration_prompts, tokenizer, device, trace_path: str = "output/runs/attn_trace.json"):
    """Run PyTorch Profiler over a forward pass; save chrome-trace JSON.

    Returns the raw profiler object (caller may inspect or dump). Replace the
    activity list with ProfilerActivity.CUDA on a GPU box.
    """
    from torch.profiler import profile, ProfilerActivity, schedule

    with profile(
        activities=[ProfilerActivity.CPU],  # add ProfilerActivity.CUDA on GPU
        schedule=schedule(wait=1, warmup=1, active=1, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler("output/runs"),
        record_shapes=True,
        with_flops=True,
    ) as prof:
        for prompt in calibration_prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                model(**inputs)
            prof.step()
    return prof


def compute_roofline(
    flops_per_expert: Dict[int, float],
    bytes_per_expert: Dict[int, float],
    peak_gflops: float,
    peak_bandwidth_gbs: float,
) -> Dict[int, Dict[str, float]]:
    """Compute arithmetic intensity + attained GFLOP/s per expert.

    AI = flops / bytes ; attained = flops / time_s (time inferred from roofline
    bound if wall-clock not supplied). Returns per-expert dict.
    """
    out: Dict[int, Dict[str, float]] = {}
    for eid in flops_per_expert:
        fl = flops_per_expert[eid]
        by = max(bytes_per_expert[eid], 1.0)
        ai = fl / by
        # Roofline attainable performance (GFLOP/s)
        attainable = min(peak_gflops, peak_bandwidth_gbs * ai)
        out[eid] = {
            "flops": fl,
            "bytes": by,
            "arithmetic_intensity": ai,
            "attainable_gflops": attainable,
            "bound": "compute" if peak_bandwidth_gbs * ai >= peak_gflops else "memory",
        }
    return out


def parse_nsight_json(path: str) -> List[dict]:
    """Stub: parse an Nsight Compute JSON export into per-kernel rows.

    Implement per Nsight schema (ncu --export; parse 'gpu__time_duration' and
    'dram__bytes' etc.). Placeholder returns [] until wired in Week 11.
    """
    # import json; return json.load(open(path))["range"] ...
    return []
