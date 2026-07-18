"""LongBench subset runner.

LongBench is pulled separately (see docs/BENENCHMARKS.md). This harness takes a
list of task names and a local/path loader and returns {task: metric}.
"""
from typing import Callable, Dict, List


def run_longbench_subset(
    model, tokenizer, device, task_names: List[str],
    loader: Callable[[str], List[dict]],
    metric: str = "f1",
) -> Dict[str, float]:
    """Run a subset of LongBench tasks.

    Args:
        loader: callable(task_name) -> list of examples, each with "input" and
                "output" (reference answer) keys. Provided by the experiment so
                this module stays dataset-source agnostic.
    """
    results: Dict[str, float] = {}
    for task in task_names:
        examples = loader(task)
        # Placeholder metric aggregation; replace with task-specific scoring.
        scores = []
        for ex in examples:
            # TODO(Week 5): generate answer, compute EM/F1 vs ex["output"]
            scores.append(0.0)
        results[task] = sum(scores) / max(len(scores), 1)
    return results
