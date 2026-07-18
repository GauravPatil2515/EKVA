"""Plain perplexity on a dataset (WikiText / C4)."""
from typing import List

import torch


def perplexity_on_dataset(model, tokenizer, dataset, device, max_samples: int = 32, ctx_len: int = 2048) -> float:
    """Average PPL over the first `max_samples` documents of `dataset`.

    `dataset` is any iterable yielding raw text strings (e.g. load_dataset(...)[split]["text"]).
    """
    model.eval()
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, text in enumerate(dataset):
            if i >= max_samples:
                break
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=ctx_len).to(device)
            if inputs["input_ids"].shape[1] < 2:
                continue
            out = model(**inputs, labels=inputs["input_ids"])
            n = inputs["input_ids"].shape[1]
            total_nll += out.loss.item() * n
            total_tokens += n
    return float(torch.exp(torch.tensor(total_nll / max(total_tokens, 1))))


def _demo_usage():
    # Example wiring (not executed by import):
    # from datasets import load_dataset
    # ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]
    # ppl = perplexity_on_dataset(model, tok, ds, "cuda")
    pass
