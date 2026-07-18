# Benchmarks

EKVA is evaluated on five families (see `configs/benchmarks.yaml`). The harness
wrappers live in `ekva/benchmarks/`; the datasets themselves are pulled
separately and passed in via a `loader` callable so the harness stays
source-agnostic.

## 1. Perplexity (WikiText / C4)
- `ekva.benchmarks.perplexity.perplexity_on_dataset(model, tok, dataset, device, ...)`
- Dataset: `datasets.load_dataset("wikitext", "wikitext-103-raw-v1", split="test")["text"]`
- Install: `pip install datasets`

## 2. Needle-in-Haystack
- `ekva.benchmarks.needle.run_needle_in_haystack(...)`: sweeps
  `(ctx_length, depth)` cells, returns retrieval accuracy.
- No external dataset; prompt template is built in.

## 3. LongBench (subset of 4–6 tasks)
- Repo: https://github.com/THUDM/LongBench
- `ekva.benchmarks.longbench.run_longbench_subset(model, tok, device, task_names, loader)`
- The `loader(task_name)` must return examples with `"input"` and `"output"` keys.
- Tasks subset: single_doc_qa, multi_doc_qa, summarization, few_shot, synthetic, code.

## 4. RULER
- Repo: https://github.com/NVIDIA/RULER
- `ekva.benchmarks.ruler.run_ruler(...)` with tasks
  niah_single_1, niah_multi_2/3, retrieval, aggregation, copy_first.
- Pull the RULER data per the upstream README; pass via `loader`.

## 5. InfiniteBench
- Repo: https://github.com/OpenBMB/InfiniteBench
- `ekva.benchmarks.infinitebench.run_infinitebench_subset(...)`; subset
  retrieve_passkey, retrieve_number, qa_known.

## Wiring into the sweep (Weeks 5–6)
`experiments/week05_06_benchmark_sweep.py` currently leaves `score_fn` as a
placeholder. To make it real:
1. Build a `loader` for each benchmark (per repo above).
2. Inside `score_fn`, construct `EKVACacheHook` with the policy budgets and call
   `hook.truncate(...)` per expert inside the MoE forward (Week 4 adapter).
3. Return the benchmark metric (PPL / EM / F1 / accuracy).

## Budget fractions
10%, 20%, 30%, 40%, 60%, 80% of FullKV (`configs/benchmarks.yaml`).
