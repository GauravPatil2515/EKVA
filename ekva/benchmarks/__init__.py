"""Benchmark harnesses for the EKVA evaluation matrix (Weeks 5-6, 12).

Each module exposes a `run(...)` returning a dict of task -> metric so the
experiment scripts can aggregate into the master results CSV.

  longbench.py     — LongBench subset (4-6 tasks)
  ruler.py         — RULER synthetic long-context suite
  needle.py        — Needle-in-Haystack (pass-key retrieval)
  infinitebench.py — InfiniteBench retrieval/QA subset
  perplexity.py    — plain WikiText/C4 perplexity

These are thin, dependency-light wrappers: the actual datasets are pulled by the
experiment scripts (see docs/BENCHMARKS.md for exact repos).
"""
from ekva.benchmarks.perplexity import perplexity_on_dataset
from ekva.benchmarks.needle import run_needle_in_haystack
from ekva.benchmarks.longbench import run_longbench_subset
from ekva.benchmarks.ruler import run_ruler
from ekva.benchmarks.infinitebench import run_infinitebench_subset

__all__ = [
    "perplexity_on_dataset",
    "run_needle_in_haystack",
    "run_longbench_subset",
    "run_ruler",
    "run_infinitebench_subset",
]
