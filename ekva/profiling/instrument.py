"""Lazy import surface for profiling submodules."""
from ekva.profiling.instrument import profile_model_attention, compute_roofline, parse_nsight_json

__all__ = ["profile_model_attention", "compute_roofline", "parse_nsight_json"]
