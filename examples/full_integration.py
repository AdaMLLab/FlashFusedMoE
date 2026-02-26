"""Full integration example: custom transformer + MoELayer + EP + OverlappedGradSync.

Demonstrates every FlashFuseMoE feature working together:
- MoELayer with shared expert, aux loss, capacity factor
- Expert Parallelism with shard_experts()
- OverlappedGradSync for dense gradient overlap
- torch.compile
- Gradient clipping + AdamW

Usage:
    # Single GPU
    python examples/full_integration.py

    # Multi GPU EP
    torchrun --nproc_per_node=2 examples/full_integration.py --ep_size 2
"""

import argparse
import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from flashfusemoe import MoELayer, OverlappedGradSync, get_dense_params, set_ep_group


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


class CausalAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, num_experts, top_k,
                 num_shared_experts=0, aux_loss_coeff=0.0):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = CausalAttention(d_model, n_heads)
        self.ffn_norm = RMSNorm(d_model)
        self.moe = MoELayer(
            d_model=d_model,
            ffn_dim=ffn_dim,
            num_experts=num_experts,
            top_k=top_k,
            activation="swiglu",
            num_shared_experts=num_shared_experts,
            aux_loss_coeff=aux_loss_coeff,
        )

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        moe_out, losses = self.moe(self.ffn_norm(x))
        return x + moe_out, losses


class SmallMoETransformer(nn.Module):
    def __init__(self, vocab_size=1024, d_model=256, n_heads=4, n_layers=4,
                 ffn_dim=1024, num_experts=8, top_k=2, max_seq_len=128,
                 num_shared_experts=1, aux_loss_coeff=0.01):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, num_experts, top_k,
                             num_shared_experts, aux_loss_coeff)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        x = self.tok_emb(input_ids) + self.pos_emb(
            torch.arange(T, device=input_ids.device).unsqueeze(0)
        )
        all_losses = {}
        for layer in self.layers:
            x, losses = layer(x)
            for k, v in losses.items():
                if k in all_losses:
                    all_losses[k] = all_losses[k] + v
                else:
                    all_losses[k] = v
        logits = self.lm_head(self.norm(x))
        return logits, all_losses


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    use_ep = args.ep_size > 1
    is_distributed = use_ep

    if is_distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        world_size = 1
        device = torch.device("cuda:0")
        torch.cuda.set_device(0)

    is_main = rank == 0

    # Build model
    model = SmallMoETransformer(
        vocab_size=1024, d_model=256, n_heads=4, n_layers=4,
        ffn_dim=1024, num_experts=8, top_k=2, max_seq_len=128,
        num_shared_experts=1, aux_loss_coeff=0.01,
    ).to(device)

    # Shard experts if EP
    grad_syncer = None
    if use_ep:
        ep_group = dist.new_group(ranks=list(range(args.ep_size)))
        for module in model.modules():
            if isinstance(module, MoELayer):
                module.shard_experts(ep_group)

        dense_params = get_dense_params(model, backward_order=True)
        grad_syncer = OverlappedGradSync(dense_params, ep_group, num_buckets=4)

    # Compile
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    if is_main:
        print(f"Parameters: {n_params / 1e6:.1f}M | EP: {args.ep_size} | compile: {args.compile}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # Training loop
    seq_len = 128
    batch_size = 8
    step_times = []

    for step in range(args.steps):
        # Random data
        input_ids = torch.randint(0, 1024, (batch_size, seq_len), device=device)
        targets = torch.randint(0, 1024, (batch_size, seq_len), device=device)

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits, losses = model(input_ids)

        ce_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(),
            targets.reshape(-1),
        )
        total_loss = ce_loss + losses.get("aux_loss", 0)
        total_loss.backward()

        if grad_syncer is not None:
            grad_syncer.finish()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000
        step_times.append(step_ms)

        if is_main and (step % 5 == 0 or step == args.steps - 1):
            aux_str = f" | aux {losses['aux_loss'].item():.4f}" if "aux_loss" in losses else ""
            print(f"step {step:>3d} | loss {ce_loss.item():.4f}{aux_str} | {step_ms:.1f} ms")

    # Summary
    if is_main and len(step_times) > 5:
        timed = step_times[5:]
        p50 = sorted(timed)[len(timed) // 2]
        tokens_per_step = batch_size * seq_len
        tok_s = tokens_per_step / (p50 / 1000)
        print(f"\np50 step time: {p50:.1f} ms | {tok_s:,.0f} tok/s")
        print(f"Peak memory: {torch.cuda.max_memory_allocated() / 1e6:.0f} MB")

    if grad_syncer is not None:
        grad_syncer.remove_hooks()

    if use_ep:
        set_ep_group(None)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()
    train(args)
