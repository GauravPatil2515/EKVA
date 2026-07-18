"""RULER synthetic long-context suite runner (NVIDIA/RULER)."""
from typing import Callable, Dict, List


def run_ruler(
    model, tokenizer, device, task_names: List[str],
    loader: Callable[[str], List[dict]],
) -> Dict[str, float]:
    """Run RULER tasks (e.g. retrieval, multiquery, aggregation, copy)."""
    results: Dict[str, float] = {}
    for task in task_names:
        examples = loader(task)
        scores = []
        for ex in examples:
            # TODO(Week 5): generate + score per RULER protocol
            scores.append(0.0)
        results[task] = sum(scores) / max(len(scores), 1)
    return results
