"""Hardware safety guardrails for resource-constrained local development (RTX 3050 6GB).

Ensures:
1. GPU VRAM hard allocation ceiling (prevents GPU driver crash/freezes).
2. Host RAM monitoring & garbage collection (prevents OOM killer / system freeze).
3. Memory cleanup utilities for benchmarks and testing.
"""
import gc
import os
import psutil
import torch


def apply_safety_guardrails(
    max_vram_fraction: float = 0.70,  # Caps PyTorch to 70% of 5.67GB (~3.9GB)
    min_available_ram_gb: float = 1.0,  # Aborts if host RAM drops below 1GB
) -> None:
    """Applies strict memory caps to prevent system freeze or GPU kernel crashes."""
    # 1. Host RAM check
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024**3)
    if available_gb < min_available_ram_gb:
        raise MemoryError(
            f"SYSTEM MEMORY GUARD: Host available RAM is critically low ({available_gb:.2f} GB). "
            f"Aborting execution to prevent system freeze."
        )

    # 2. GPU VRAM Hard Cap
    if torch.cuda.is_available():
        # Clean cached memory first
        torch.cuda.empty_cache()
        gc.collect()

        # Set per-process hard memory fraction
        torch.cuda.set_per_process_memory_fraction(max_vram_fraction, 0)


def safe_cleanup() -> None:
    """Explicitly releases all cached PyTorch memory and invokes Python GC."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_memory_snapshot() -> dict:
    """Returns a dictionary snapshot of current host RAM and GPU VRAM usage."""
    mem = psutil.virtual_memory()
    snapshot = {
        "host_ram_used_gb": round((mem.total - mem.available) / (1024**3), 2),
        "host_ram_avail_gb": round(mem.available / (1024**3), 2),
        "host_ram_percent": mem.percent,
    }
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info(0)
        snapshot.update({
            "gpu_vram_alloc_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
            "gpu_vram_free_gb": round(free_b / (1024**3), 2),
            "gpu_vram_total_gb": round(total_b / (1024**3), 2),
        })
    return snapshot
