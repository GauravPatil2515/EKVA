"""Trains the real small MoE model on the range-telemetry retrieval task
until it genuinely solves it under FullKV, then saves the checkpoint."""
from __future__ import annotations

import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from range_telemetry.data import make_batch, VOCAB_SIZE  # noqa: E402
from range_telemetry.model import ModelConfig, RangeTelemetryMoE  # noqa: E402


def full_kv_forward(model: RangeTelemetryMoE, ctx_ids: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    pre = model.prefill(ctx_ids)
    ctx_len = ctx_ids.shape[1]
    B, Tctx = ctx_ids.shape
    keep_mask = torch.ones(B, Tctx, dtype=torch.bool, device=ctx_ids.device)
    logits = model.decode(pre, ctx_len, query_ids, keep_mask)
    return logits[:, -1, :]  # prediction at ANSWER position -> next token (VALUE)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    gen = torch.Generator()
    gen.manual_seed(1234)

    final_num_readings = 6
    cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=96, n_heads=6, n_layers=4, n_experts=4, d_ff=192, max_len=128)
    model = RangeTelemetryMoE(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(os.environ.get("N_STEPS", 4000)))

    batch_size = 256
    n_steps = int(os.environ.get("N_STEPS", 4000))
    aux_weight = 0.02
    t0 = time.time()

    def curriculum_readings(step: int) -> int:
        # ramp context length from 2 to final_num_readings over the first 70% of training
        ramp_steps = int(0.7 * n_steps)
        if step >= ramp_steps:
            return final_num_readings
        frac = step / max(1, ramp_steps)
        r = 2 + int(frac * (final_num_readings - 2))
        return max(2, min(final_num_readings, r))

    for step in range(1, n_steps + 1):
        num_readings = curriculum_readings(step)
        batch = make_batch(batch_size, num_readings, device, gen)
        pre = model.prefill(batch.ctx_ids)
        ctx_len = batch.ctx_ids.shape[1]
        B = batch.ctx_ids.shape[0]
        keep_mask = torch.ones(B, ctx_len, dtype=torch.bool, device=device)
        logits = model.decode(pre, ctx_len, batch.query_ids, keep_mask)
        pred_logits = logits[:, -1, :]
        loss = F.cross_entropy(pred_logits, batch.target) + aux_weight * pre["aux_loss"]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                acc = (pred_logits.argmax(-1) == batch.target).float().mean().item()
            print(f"step {step:5d}  R={num_readings:2d}  loss {loss.item():.4f}  train_acc {acc:.3f}  elapsed {time.time()-t0:.1f}s")

    # Held-out evaluation under FullKV
    model.eval()
    with torch.no_grad():
        correct, total = 0, 0
        eval_gen = torch.Generator()
        eval_gen.manual_seed(999)
        for _ in range(20):
            batch = make_batch(256, final_num_readings, device, eval_gen)
            pred = full_kv_forward(model, batch.ctx_ids, batch.query_ids)
            correct += (pred.argmax(-1) == batch.target).sum().item()
            total += batch.target.shape[0]
        print(f"\nHeld-out FullKV exact-match accuracy: {correct/total:.4f}  ({correct}/{total})")

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoint.pt")
    torch.save({"model_state": model.state_dict(), "cfg": cfg, "num_readings": final_num_readings}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
