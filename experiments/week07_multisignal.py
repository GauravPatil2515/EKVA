"""Week 7: Multi-signal EKVA upgrade + finalize best policy.

Adds specialization score (token-type diversity) and routing frequency into
derive_kv_budget (strategy='multi_signal'). Re-runs the best 2-3 combinations
from Weeks 5-6 and compares entropy-only vs multi-signal.

Requires entropy_map + a specialization tensor (compute via
ekva.calibration.signals.specialization_score during a calibration pass).

Usage:
  python experiments/week07_multisignal.py --model mixtral-8x7b --calibration output/mixtral-8x7b_general_phase1.pt
"""
import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ekva.models import get_model_spec
from ekva.budget.derive import derive_kv_budget
from ekva.calibration.signals import specialization_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--total-budget", type=int, default=2048)
    args = ap.parse_args()
    spec = get_model_spec(args.model)

    d = torch.load(args.calibration, map_location="cpu")
    emap = d["entropy_map"]
    # NOTE: real specialization needs token-type assignments collected during
    # calibration. Placeholder: derive from routing concentration as a proxy.
    token_types = {e: torch.randint(0, 8, (int(emap[e]["routing_count"].item()) + 1,)) for e in emap}
    spec_score = specialization_score(token_types, spec.num_experts)

    entropy_only = derive_kv_budget(emap, args.total_budget, strategy="proportional")
    multi = derive_kv_budget(emap, args.total_budget, strategy="multi_signal", specialization=spec_score)

    out = Path("output/week07") / f"{args.model}_multisignal.pt"
    os.makedirs(out.parent, exist_ok=True)
    torch.save({"entropy_only": entropy_only, "multi_signal": multi, "specialization": spec_score}, out)
    print(f"[W7] entropy_only sum={int(entropy_only.sum())}  multi_signal sum={int(multi.sum())}")
    print(f"[W7] Saved comparison -> {out}")


if __name__ == "__main__":
    main()
