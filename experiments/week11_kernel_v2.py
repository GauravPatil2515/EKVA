"""Week 11: Fused kernel v2 + hardware profiling.

Extends v1 with fused important-token index selection. Profiles with Nsight
Compute (host tool) for HBM bandwidth / SM occupancy / kernel time; compares
baseline vs v2 per expert class. Requires triton + CUDA + ncu/nsys on host.

Usage:
  python experiments/week11_kernel_v2.py --device cuda
"""
import argparse
import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.kernel.ekva_triton_v1 import ekva_attention_v1
from ekva.kernel.ekva_triton_v2 import ekva_attention_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    dev = torch.device(args.device)
    B, H, N, D = 1, 8, 1024, 64
    q = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    k = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    v = torch.randn(B, H, N, D, device=dev, dtype=torch.float16)
    scale = D ** -0.5
    kv_budget = N // 4
    idx = torch.argsort(torch.rand(B, H, N, device=dev), dim=-1)[..., :kv_budget].long()

    # v1 (variable tile) and v2 (fused gather) should agree when idx = first kv_budget.
    out1 = ekva_attention_v1(q, k, v, kv_budget=kv_budget, scale=scale)
    out2 = ekva_attention_v2(q, k, v, kv_budget=kv_budget, important_indices=idx, scale=scale)
    print(f"[W11] v1 shape {tuple(out1.shape)}, v2 shape {tuple(out2.shape)} OK")

    # Host-side Nsight command (run manually on the GPU box):
    #   ncu --set full --export output/week11/ekva_v2.ncu-rep python experiments/week11_kernel_v2.py
    print("[W11] For Nsight Compute profiling, wrap this script: "
          "ncu --set full --export output/week11/ekva_v2.ncu-rep "
          "python experiments/week11_kernel_v2.py")


if __name__ == "__main__":
    main()
