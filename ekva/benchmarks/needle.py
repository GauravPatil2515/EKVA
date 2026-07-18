"""Needle-in-Haystack pass-key retrieval benchmark.

Builds a long context with a random "secret" inserted at a depth fraction, then
asks the model to retrieve it. Returns retrieval accuracy over a sweep of
(context_length, depth) cells.
"""
from typing import Dict, List

import torch


def run_needle_in_haystack(
    model, tokenizer, device, ctx_lengths: List[int], depths: List[float],
    prompt_template: str = "The secret code is {code}. ", filler: str = "The weather is sunny. ",
    code: str = "CAT-7291", n_repeats_filler: int = 50,
) -> Dict[str, float]:
    """Return {"accuracy": frac_correct, "n": n_cells}."""
    model.eval()
    correct, total = 0, 0
    quest = f"What is the secret code? Answer with the code only."
    with torch.no_grad():
        for L in ctx_lengths:
            for d in depths:
                insert_at = int(L * d)
                hay = (filler * n_repeats_filler)[:insert_at] + prompt_template.format(code=code) + (filler * n_repeats_filler)
                hay = hay[:L]
                inp = tokenizer(hay + "\n" + quest, return_tensors="pt", truncation=True, max_length=L + 64).to(device)
                out = model.generate(**inp, max_new_tokens=16)
                decoded = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
                total += 1
                if code.lower() in decoded.lower():
                    correct += 1
    return {"accuracy": correct / max(total, 1), "n": total}
