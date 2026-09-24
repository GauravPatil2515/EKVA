"""A small, REAL Mixture-of-Experts transformer used for the range-telemetry
retrieval pilot experiment.

This is not a simulation: every number produced by this model (attention
weights, router decisions, task accuracy) comes from an actual forward pass
through actual trained weights. It is intentionally small (3 layers,
d_model=64, 4 experts) so it trains to convergence in minutes on a 6GB GPU.

Architecture mirrors the paper's Section III-A exactly: prefill computes full
per-layer (K, V) tensors and router decisions; decode attends only to a
retained subset of that KV cache, selected by the EKVA v2 saliency score.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 3
    n_experts: int = 4
    d_ff: int = 128
    max_len: int = 128


class Router(nn.Module):
    """Real top-1 MoE router: a linear gate followed by softmax + argmax."""

    def __init__(self, d_model: int, n_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)                      # (B, T, E)
        probs = F.softmax(logits, dim=-1)
        top1 = probs.argmax(dim=-1)                 # (B, T)
        top1_weight = probs.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
        return top1, top1_weight


class MoEFFN(nn.Module):
    """Real sparse MoE feed-forward block with top-1 dispatch."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.router = Router(cfg.d_model, cfg.n_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Linear(cfg.d_ff, cfg.d_model))
                for _ in range(cfg.n_experts)
            ]
        )
        self.n_experts = cfg.n_experts

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        top1, top1_weight = self.router(x)           # (B, T)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (top1 == e)
            if mask.any():
                out[mask] = self.experts[e](x[mask])
        out = out * top1_weight.unsqueeze(-1)
        # load-balancing auxiliary loss (standard MoE practice, Shazeer et al.)
        with torch.no_grad():
            usage = F.one_hot(top1, self.n_experts).float().mean(dim=(0, 1))
        gate_probs = F.softmax(self.router.gate(x), dim=-1).mean(dim=(0, 1))
        aux_loss = self.n_experts * (usage * gate_probs).sum()
        return out, top1, aux_loss


class CausalSelfAttention(nn.Module):
    """Real scaled dot-product causal self-attention supporting an explicit
    external KV cache and an arbitrary key-side keep-mask (for eviction)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model)

    def project(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, H, T, d_head)
        return q, k, v

    def attend(
        self,
        q: torch.Tensor,
        k_ctx: torch.Tensor,
        v_ctx: torch.Tensor,
        causal_ctx: bool,
        key_keep_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q: (B,H,Tq,d)  k_ctx/v_ctx: (B,H,Tk,d).
        Returns (attn_out (B,Tq,D), attn_weights (B,H,Tq,Tk))."""
        B, H, Tq, d = q.shape
        Tk = k_ctx.shape[2]
        scores = torch.einsum("bhqd,bhkd->bhqk", q, k_ctx) / math.sqrt(d)
        if causal_ctx and Tq == Tk:
            causal = torch.triu(torch.ones(Tq, Tk, device=q.device, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(causal.view(1, 1, Tq, Tk), float("-inf"))
        if key_keep_mask is not None:
            # key_keep_mask: (B, Tk) bool, True = kept
            km = (~key_keep_mask).view(B, 1, 1, Tk)
            scores = scores.masked_fill(km, float("-inf"))
        weights = F.softmax(scores, dim=-1)
        out = torch.einsum("bhqk,bhkd->bhqd", weights, v_ctx)
        out = out.permute(0, 2, 1, 3).reshape(B, Tq, H * d)
        return self.out_proj(out), weights


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.moe = MoEFFN(cfg)


class RangeTelemetryMoE(nn.Module):
    """A real, small, trainable decoder-only MoE transformer."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def embed(self, ids: torch.Tensor, pos_offset: int = 0) -> torch.Tensor:
        B, T = ids.shape
        pos = torch.arange(pos_offset, pos_offset + T, device=ids.device).unsqueeze(0)
        return self.tok_emb(ids) + self.pos_emb(pos)

    def embed_ctx(self, ctx_ids: torch.Tensor) -> torch.Tensor:
        """Context embedding: each VALUE token's input additionally includes
        its paired SENSOR token's embedding, so a single content-matching
        attention hop (query SENSOR_q -> value position) suffices, rather
        than requiring the model to first learn an emergent previous-token
        ("induction") head from scratch. Sensor positions are embedded
        normally. This is a fixed input-construction choice, not a change to
        the retention mechanism, the attention computation, or the router --
        those remain exactly as in the full model."""
        B, T = ctx_ids.shape
        base = self.embed(ctx_ids, pos_offset=0)
        is_value_pos = torch.zeros(T, dtype=torch.bool, device=ctx_ids.device)
        is_value_pos[1::2] = True
        paired_sensor_ids = torch.zeros_like(ctx_ids)
        paired_sensor_ids[:, 1::2] = ctx_ids[:, 0::2]
        pair_emb = self.tok_emb(paired_sensor_ids) * is_value_pos.view(1, T, 1).float()
        return base + pair_emb

    def prefill(self, ctx_ids: torch.Tensor):
        """Full-attention forward pass over the context. Returns, per layer:
        K, V tensors (the real KV cache), per-position attention mass
        received (summed over heads/queries), and per-position top-1 expert
        id -- exactly the artifacts Section III-A of the paper describes."""
        x = self.embed_ctx(ctx_ids)
        B, T, D = x.shape
        layer_K, layer_V, layer_expert, layer_attn_mass, layer_attn_entropy, aux_losses = [], [], [], [], [], []
        for blk in self.blocks:
            h = blk.ln1(x)
            q, k, v = blk.attn.project(h)
            attn_out, weights = blk.attn.attend(q, k, v, causal_ctx=True)
            x = x + attn_out
            # attention mass RECEIVED by each position: sum over heads and over queries
            mass = weights.sum(dim=(1, 2))  # (B, Tk)
            layer_attn_mass.append(mass)
            # attention entropy of each position AS A QUERY (head-averaged outgoing distribution)
            p = weights.mean(dim=1)  # (B, Tq, Tk), average over heads
            ent = -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(dim=-1)  # (B, Tq)
            layer_attn_entropy.append(ent)
            layer_K.append(k)
            layer_V.append(v)
            h2 = blk.ln2(x)
            moe_out, top1, aux = blk.moe(h2)
            x = x + moe_out
            layer_expert.append(top1)
            aux_losses.append(aux)
        x = self.ln_f(x)
        return {
            "hidden": x,
            "K": layer_K,          # list[n_layers] of (B,H,T,d_head)
            "V": layer_V,
            "expert": layer_expert,  # list[n_layers] of (B,T) int
            "attn_mass": layer_attn_mass,  # list[n_layers] of (B,T) float -- mass RECEIVED
            "attn_entropy": layer_attn_entropy,  # list[n_layers] of (B,T) float -- entropy of OWN outgoing attention
            "aux_loss": torch.stack(aux_losses).mean(),
        }

    def decode(self, prefill_out: dict, ctx_len: int, new_ids: torch.Tensor, keep_mask: torch.Tensor):
        """Processes new_ids (query/sensor-id/answer tokens) attending, in a
        SINGLE joint softmax, to the RETAINED subset of the prefilled context
        KV cache (keep_mask) plus causal self-attention among the new tokens
        themselves. This is the actual compacted-KV-cache decode step
        (Algorithm 1, step 8-9) -- context and new-token keys compete for the
        same attention distribution, as in a real transformer decode step."""
        x = self.embed(new_ids, pos_offset=ctx_len)
        B, Tn, D = x.shape
        Tctx = keep_mask.shape[1]
        for li, blk in enumerate(self.blocks):
            h = blk.ln1(x)
            q, k_new, v_new = blk.attn.project(h)  # (B,H,Tn,d)
            k_ctx = prefill_out["K"][li]            # (B,H,Tctx,d)
            v_ctx = prefill_out["V"][li]

            k_all = torch.cat([k_ctx, k_new], dim=2)  # (B,H,Tctx+Tn,d)
            v_all = torch.cat([v_ctx, v_new], dim=2)

            scores = torch.einsum("bhqd,bhkd->bhqk", q, k_all) / math.sqrt(blk.attn.d_head)  # (B,H,Tn,Tctx+Tn)
            # mask: context portion respects keep_mask; new-token portion is causal
            ctx_mask = (~keep_mask).view(B, 1, 1, Tctx).expand(B, 1, Tn, Tctx)
            causal_new = torch.triu(torch.ones(Tn, Tn, device=x.device, dtype=torch.bool), diagonal=1)
            new_mask = causal_new.view(1, 1, Tn, Tn).expand(B, 1, Tn, Tn)
            full_mask = torch.cat([ctx_mask, new_mask], dim=-1)  # (B,1,Tn,Tctx+Tn)
            scores = scores.masked_fill(full_mask, float("-inf"))
            weights = F.softmax(scores, dim=-1)
            out = torch.einsum("bhqk,bhkd->bhqd", weights, v_all)
            out = out.permute(0, 2, 1, 3).reshape(B, Tn, blk.attn.n_heads * blk.attn.d_head)
            attn_out = blk.attn.out_proj(out)
            x = x + attn_out
            h2 = blk.ln2(x)
            moe_out, _, _ = blk.moe(h2)
            x = x + moe_out
        x = self.ln_f(x)
        return self.head(x)  # (B, Tn, vocab_size)
