"""InfiniteBench subset runner (retrieval / QA long-context)."""
from typing import Callable, Dict, List


def run_infinitebench_subset(
    model, tokenizer, device, task_names: List[str],
    loader: Callable[[str], List[dict]],
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for task in task_names:
        examples = loader(task)
        scores = [0.0 for _ in examples]  # TODO(Week 5): real scoring
        results[task] = sum(scores) / max(len(scores), 1)
    return results
