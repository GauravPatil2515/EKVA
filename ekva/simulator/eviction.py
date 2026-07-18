"""Eviction strategy registry for the simulator.

Strategies are also expressed as pure functions so the kernel (Weeks 10-11) and
the hook (Week 4) can share the same vocab: "recency", "attention", "random",
"hybrid".
"""
from typing import Dict

# Lazy import to avoid a circular dependency at module import time.
from ekva.simulator.kv_buffer import SUPPORTED_EVICTION  # noqa: F401

EVICTION_REGISTRY: Dict[str, str] = {name: name for name in SUPPORTED_EVICTION}


def get_eviction(name: str) -> str:
    if name not in EVICTION_REGISTRY:
        raise KeyError(f"Unknown eviction '{name}'. Available: {list(EVICTION_REGISTRY)}")
    return name


def validate_eviction(name: str) -> bool:
    return name in EVICTION_REGISTRY
