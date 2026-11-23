"""Micro-benchmark for EKVA Triton Compaction Kernel vs PyTorch Scatter/Gather.

Uses triton.testing.do_bench to measure:
1. Exact wall-clock latency of Triton gather kernel vs torch.gather.
2. Achieved memory throughput (GB/s).
3. Decoding attention time saved across sequence lengths: 1k, 2k, 4k, 8k, 16k tokens.
4. Generates crossover table showing exact sequence length where EKVA is net-faster.
"""
import argparse
import json
import os
import torch
import triton
import triton.testing

from ekva.kernel.ekva_eviction_v2 import triton_compact_kv_cache
from ekva.retention.eviction import compact_kv_tensor
from ekva.safety import apply_safety_guardrails, safe_cleanup, get_memory_snapshot


def benchmark_compaction_suite(
    device_id: int = 0,
    budget_ratio: float = 0.50,
    head_dim: int = 64,
    num_heads: int = 8,
    batch_size: int = 1,
    out_dir: str = "output/benchmark_crossover",
):
    apply_safety_guardrails(max_vram_fraction=0.65)
    device = torch.device(f"cuda:{device_id}")
    dtype = torch.float16

    os.makedirs(out_dir, exist_ok=True)
    results = []

    # Safe sequence lengths for local 6GB GPU
    seq_lens = [512, 1024, 2048, 4096, 8192]

    print("=" * 80)
    print(f"🚀 EKVA TRITON COMPACTION MICRO-BENCHMARK (Device: {torch.cuda.get_device_name(device)})")
    print(f"Batch: {batch_size} | Heads: {num_heads} | HeadDim: {head_dim} | Budget: {budget_ratio * 100:.0f}%")
    print("=" * 80)

    for T in seq_lens:
        budget = int(T * budget_ratio)

        # Allocate KV tensors
        k = torch.randn(batch_size, num_heads, T, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, T, head_dim, device=device, dtype=dtype)

        # Random sorted indices
        rand_perm = torch.randperm(T, device=device)[:budget]
        retained_indices, _ = torch.sort(rand_perm)
        retained_indices = retained_indices.unsqueeze(0).expand(batch_size, budget)

        # Total bytes transferred: read K & V (2 * B * H * budget * D * 2 bytes) + write K & V (same)
        # Note: Triton gathers only the budget tokens, so reads budget elements and writes budget elements
        bytes_transferred = 2 * (batch_size * num_heads * budget * head_dim * 2) * 2

        # 1. Benchmark PyTorch gather
        def run_torch():
            k_c = compact_kv_tensor(k, retained_indices)
            v_c = compact_kv_tensor(v, retained_indices)
            return k_c, v_c

        ms_torch = triton.testing.do_bench(run_torch, warmup=25, rep=100)

        # 2. Benchmark Triton Fused Kernel
        def run_triton():
            return triton_compact_kv_cache(k, v, retained_indices)

        ms_triton = triton.testing.do_bench(run_triton, warmup=25, rep=100)

        speedup = ms_torch / max(ms_triton, 1e-6)
        gb_s_triton = (bytes_transferred / (ms_triton / 1000.0)) / (1024**3)

        # 3. Decode Attention Simulation (1 new token attending to uncompressed T vs compressed budget)
        q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)

        def decode_attn_full():
            # Q @ K.T -> (B, H, 1, T)
            scores = torch.matmul(q, k.transpose(-1, -2))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, v)

        def decode_attn_compact():
            # Attending only to budget length
            ck = k[:, :, :budget, :]
            cv = v[:, :, :budget, :]
            scores = torch.matmul(q, ck.transpose(-1, -2))
            attn = torch.softmax(scores, dim=-1)
            return torch.matmul(attn, cv)

        ms_decode_full = triton.testing.do_bench(decode_attn_full, warmup=25, rep=100)
        ms_decode_compact = triton.testing.do_bench(decode_attn_compact, warmup=25, rep=100)
        time_saved_per_token = ms_decode_full - ms_decode_compact

        row = {
            "seq_len": T,
            "budget": budget,
            "torch_gather_ms": round(ms_torch, 4),
            "triton_compact_ms": round(ms_triton, 4),
            "triton_speedup": round(speedup, 2),
            "bandwidth_gb_s": round(gb_s_triton, 2),
            "decode_full_ms": round(ms_decode_full, 4),
            "decode_compact_ms": round(ms_decode_compact, 4),
            "decode_saved_per_step_ms": round(time_saved_per_token, 4),
            # Break-even tokens: how many decode steps to amortize one compaction event
            "breakeven_decode_steps": int(ms_triton / max(time_saved_per_token, 1e-6)) + 1,
        }
        results.append(row)

        print(
            f"T={T:5d} | Budget={budget:4d} | "
            f"PyTorch: {ms_torch:6.3f}ms | Triton: {ms_triton:6.3f}ms ({speedup:4.2f}x) | "
            f"Bandwidth: {gb_s_triton:6.1f} GB/s | "
            f"Decode Saved/Step: {time_saved_per_token*1000:6.1f}µs | "
            f"Breakeven: {row['breakeven_decode_steps']:2d} steps"
        )

        safe_cleanup()

    # Save JSON summary
    json_path = os.path.join(out_dir, "triton_crossover_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"✅ Benchmark completed! Results written to: {json_path}")
    print(f"Memory Snapshot: {get_memory_snapshot()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--budget-ratio", type=float, default=0.50)
    args = parser.parse_args()
    benchmark_compaction_suite(device_id=args.device, budget_ratio=args.budget_ratio)
