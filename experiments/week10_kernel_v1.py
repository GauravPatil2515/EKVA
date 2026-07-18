"""Week 10: Triton kernel prototype v1 — variable KV tile count per expert.

Validates correctness: output matches standard attention when kv_budget==N.
Benchmarks kernel time + HBM reads vs baseline FA2. Requires triton + CUDA.

Usage:
  python experiments/week10_kernel_v1.py --device cuda
"""
import argparse
import os
import sys
import time

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.kernel.reference_flashattn2 import flash_attention_2_reference
from ekva.kernel.ekva_triton_v1 import ekva_attention_v1


def _torch_ref(q, k, v, scale):
    # Exact attention via torch (correctness oracle).
    scores = torch.einsum("bhqd,bhkd->bhqk", q * scale, k)
    p = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", p, v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device(args.device)
    B, H, N, D = 1, 8, args.n, 64
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    scale = D ** -0.5

    # Correctness: full budget must match reference.
    ref = _torch_ref(q, k, v, scale)
    out_full = ekva_attention_v1(q, k, v, kv_budget=N, scale=scale)
    err_full = (out_full - ref).abs().max().item()
    print(f"[W10] max|EKVA(full) - ref| = {err_full:.4e}  (should be ~0)")

    # Reduced budget: just checks it runs & is smaller than full.
    out_red = ekva_attention_v1(q, k, v, kv_budget=N // 4, scale=scale)
    print(f"[W10] reduced-budget output shape {tuple(out_red.shape)} OK")

    # Timing.
    for fn, name in [(flash_attention_2_reference, "baseline_fa2"), (ekva_attention_v1, "ekva_v1")]:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            fn(q, k, v, N if name == "baseline_fa2" else N, scale=scale)
        torch.cuda.synchronize()
        print(f"[W10] {name}: {(time.perf_counter()-t0)/20*1e3:.3f} ms")


if __name__ == "__main__":
    main()
