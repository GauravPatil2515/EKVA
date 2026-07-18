"""Pytest config: keep the suite green on the 3050 CPU box.

The Triton kernel tests (Weeks 10-11) need `triton` + CUDA, which are absent on
the laptop. We exclude that file from collection here instead of relying on an
in-module skip (pytest's default import mode raises Skipped as an
INTERNALERROR during import). The kernel tests still run on Colab A100.
"""
import sys

collect_ignore = []
try:
    import triton  # noqa: F401
    import torch
    if not torch.cuda.is_available():
        collect_ignore.append("tests/test_kernel.py")
except Exception:
    collect_ignore.append("tests/test_kernel.py")
