"""Multi-GPU Expert Parallelism benchmark.

Benchmark matrix (4 modes × 4 GPU counts):
  Modes:     unfused_bf16, fused_bf16, unfused_fp8, fused_fp8
  GPU counts: 1, 2, 4, 8

Metrics per config:
  - Tokens/sec (per GPU and aggregate)
  - Step time (avg, p50, p95)
  - Scaling efficiency
  - Communication time
  - Peak memory per GPU

Usage:
    torchrun --nproc_per_node=8 benchmarks/train_multigpu.py
    torchrun --nproc_per_node=2 benchmarks/train_multigpu.py --ep_sizes 1,2
    torchrun --nproc_per_node=8 benchmarks/train_multigpu.py --modes fused_bf16,fused_fp8
"""

import argparse
import json
import math
import os
import random
import time

import tiktoken
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import load_dataset

from flashfusemoe import OverlappedGradSync, get_dense_params

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from deepseek.model import (
    DeepSeekTransformer, DeepSeekConfig, MOE_UNFUSED, MOE_FUSED,
    count_params, shard_experts, DeepSeekMoEFFN,
)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

DEFAULT_TRAIN_STEPS = 20
WARMUP_SKIP = 15  # skip first N steps for timing (compile warmup can take 10+ steps)
SEQ_LEN = 1024
MICRO_BATCH = 8
GRAD_ACCUM = 1  # simplify for benchmark — 1 micro-batch per step
TOKENS_PER_STEP = MICRO_BATCH * SEQ_LEN

LR = 3e-4
MUON_LR = 0.02
WARMUP_STEPS = 5
MUON_MOMENTUM_WARMUP = 300
MUON_MOMENTUM_COOLDOWN = 50
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0

VAL_SEQUENCES = 32


# ---------------------------------------------------------------------------
# Muon optimizer (based on github.com/KellerJordan/modded-nanogpt)
# ---------------------------------------------------------------------------

def _zeropower_via_newtonschulz5(G, steps=5):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization.

    Uses a quintic iteration with coefficients maximizing slope at zero.
    Produces something like US'V^T where S' ~ Uniform(0.5, 1.5), which
    empirically doesn't hurt model performance vs exact UV^T.

    Reference: KellerJordan/Muon (github.com/KellerJordan/Muon)
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """Muon — MomentUm Orthogonalized by Newton-schulz.

    Runs standard SGD-momentum, then replaces each 2D parameter's update with
    the nearest orthogonal matrix via Newton-Schulz iteration. Runs stably in
    bfloat16 on GPU.

    Only use for hidden 2D weight matrices. Embeddings, output heads, biases,
    norms, and 3D expert weights should use AdamW.

    In distributed mode, NS computation is sharded across ranks: each rank
    orthogonalizes 1/world_size of the params, then all_gathers results.

    Reference: github.com/KellerJordan/Muon
    """

    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95,
                 nesterov=True, ns_steps=5, distributed=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        nesterov=nesterov, ns_steps=ns_steps)
        # Accept list of params (not param groups)
        if isinstance(params, list) and len(params) > 0 and isinstance(params[0], torch.nn.Parameter):
            params = sorted(params, key=lambda x: x.numel(), reverse=True)
        super().__init__([{"params": list(params)}], defaults)
        self.distributed = distributed

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            if not params:
                continue

            lr = group["lr"]
            wd = group["weight_decay"]
            beta = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]

            if self.distributed and dist.is_initialized():
                ws = dist.get_world_size()
                rank = dist.get_rank()
                # Pad to multiple of world_size
                pad = (ws - len(params) % ws) % ws
                padded = params + [torch.zeros(1, device=params[0].device)] * pad
                # Side stream for broadcasts — overlap with next batch's NS compute
                comm_stream = torch.cuda.Stream(device=params[0].device)
                pending = []
                for base_i in range(0, len(padded), ws):
                    idx = base_i + rank
                    if idx < len(params):
                        self._update_param(params[idx], lr, wd, beta, nesterov, ns_steps)
                    # Launch broadcasts on side stream (overlap with next NS batch)
                    ready = torch.cuda.Event()
                    ready.record()
                    with torch.cuda.stream(comm_stream):
                        comm_stream.wait_event(ready)
                        for j in range(ws):
                            src_idx = base_i + j
                            if src_idx < len(params):
                                dist.broadcast(params[src_idx].data, src=j)
                        ev = torch.cuda.Event()
                        ev.record()
                        pending.append(ev)
                # Sync all broadcasts before returning
                for ev in pending:
                    torch.cuda.current_stream().wait_event(ev)
            else:
                for p in params:
                    self._update_param(p, lr, wd, beta, nesterov, ns_steps)

        return loss

    def _update_param(self, p, lr, wd, beta, nesterov, ns_steps):
        grad = p.grad
        state = self.state[p]
        if len(state) == 0:
            state["momentum_buffer"] = torch.zeros_like(grad)

        buf = state["momentum_buffer"]
        buf.lerp_(grad, 1 - beta)
        update = grad.lerp_(buf, beta) if nesterov else buf

        # Newton-Schulz orthogonalization
        if update.ndim == 4:  # conv filters
            update = update.view(len(update), -1)
        update = _zeropower_via_newtonschulz5(update, steps=ns_steps)

        # Shape-based LR scaling (wider matrices get larger updates)
        update *= max(1, update.size(-2) / update.size(-1)) ** 0.5

        # Decoupled weight decay + update
        p.mul_(1 - lr * wd)
        p.add_(update.reshape(p.shape).to(p.dtype), alpha=-lr)


def _get_muon_momentum(step, total_steps, warmup=MUON_MOMENTUM_WARMUP,
                       cooldown=MUON_MOMENTUM_COOLDOWN,
                       lo=0.85, hi=0.95):
    """Momentum warmup 0.85→0.95, then cooldown 0.95→0.85 at end."""
    cd_start = total_steps - cooldown
    if step < warmup:
        return lo + (hi - lo) * step / warmup
    elif step > cd_start:
        return hi - (hi - lo) * (step - cd_start) / cooldown
    return hi


def create_optimizer(model, optimizer_type="adamw", lr=3e-4, muon_lr=0.02,
                     weight_decay=0.1, betas=(0.9, 0.95), distributed=False):
    """Create optimizer. Muon for 2D hidden weights, AdamW for the rest."""
    if optimizer_type == "muon":
        muon_params = []
        adamw_params = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Muon: only strictly 2D hidden weight matrices
            # Exclude: embeddings, lm_head, 1D norms/biases, 3D expert weights
            if p.dim() == 2 and "tok_emb" not in name and "lm_head" not in name:
                muon_params.append(p)
            else:
                adamw_params.append(p)
        opt_muon = Muon(muon_params, lr=muon_lr, weight_decay=weight_decay,
                        momentum=0.95, nesterov=True, distributed=distributed)
        opt_adamw = torch.optim.AdamW(
            adamw_params, lr=lr, betas=betas,
            weight_decay=weight_decay, fused=True,
        )
        return _CombinedOptimizer(opt_muon, opt_adamw)
    return torch.optim.AdamW(
        model.parameters(), lr=lr, betas=betas,
        weight_decay=weight_decay, fused=True,
    )


class _CombinedOptimizer:
    """Wraps Muon + AdamW so they behave as one optimizer."""
    def __init__(self, *optimizers):
        self.optimizers = list(optimizers)
        self.param_groups = []
        for opt in self.optimizers:
            self.param_groups.extend(opt.param_groups)

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none)

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]

    def load_state_dict(self, state_dicts):
        for opt, sd in zip(self.optimizers, state_dicts):
            opt.load_state_dict(sd)

    @property
    def muon(self):
        """Access the Muon optimizer (first in list) for momentum scheduling."""
        return self.optimizers[0] if isinstance(self.optimizers[0], Muon) else None


# ---------------------------------------------------------------------------
# FP8 support (copied from train_fineweb.py)
# ---------------------------------------------------------------------------

def _check_fp8():
    """Check if FP8 training is available."""
    try:
        a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        a_fp8 = a.to(torch.float8_e4m3fn)
        b_fp8 = b.to(torch.float8_e4m3fn)
        scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
        torch._scaled_mm(a_fp8, b_fp8.t(), scale_a=scale, scale_b=scale,
                         out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


class FP8Matmul(torch.autograd.Function):
    """FP8 forward matmul, BF16 backward."""

    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()

        amax_x = x_2d.abs().amax().clamp(min=1e-12)
        amax_w = weight.abs().amax().clamp(min=1e-12)
        scale_x = (amax_x / 448.0).float()
        scale_w = (amax_w / 448.0).float()

        x_fp8 = (x_2d.float() / scale_x).clamp(-448, 448).to(torch.float8_e4m3fn)
        w_fp8 = (weight.float() / scale_w).clamp(-448, 448).to(torch.float8_e4m3fn)

        out = torch._scaled_mm(x_fp8, w_fp8.t(), scale_a=scale_x, scale_b=scale_w,
                                out_dtype=torch.bfloat16)
        return out.reshape(*orig_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        go_2d = grad_output.reshape(-1, grad_output.shape[-1])
        x_2d = x.reshape(-1, x.shape[-1])
        w = weight.to(go_2d.dtype)
        grad_x = (go_2d @ w).reshape(x.shape)
        grad_weight = go_2d.t().float() @ x_2d.float()
        return grad_x, grad_weight


class FP8Linear(torch.nn.Linear):
    """Drop-in nn.Linear replacement that uses FP8 matmul in forward."""

    def forward(self, x):
        out = FP8Matmul.apply(x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


def swap_linears_to_fp8(model):
    """Replace all nn.Linear layers with FP8Linear (in-place)."""
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, torch.nn.Linear) and not isinstance(child, FP8Linear):
                fp8 = FP8Linear(child.in_features, child.out_features,
                                bias=child.bias is not None)
                fp8.weight = child.weight
                if child.bias is not None:
                    fp8.bias = child.bias
                setattr(module, child_name, fp8)
    return model


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def make_data_iterator(tokenizer, split_offset=0):
    """Stream FineWeb-Edu, tokenize, yield [seq_len+1] chunks."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    buffer = []
    for i, example in enumerate(ds):
        if i < split_offset:
            continue
        tokens = tokenizer.encode(example["text"])
        buffer.extend(tokens)
        while len(buffer) >= SEQ_LEN + 1:
            yield buffer[:SEQ_LEN + 1]
            buffer = buffer[SEQ_LEN + 1:]


def collect_val_data(tokenizer, n_sequences):
    """Collect fixed validation set."""
    sequences = []
    it = make_data_iterator(tokenizer, split_offset=0)
    for seq in it:
        sequences.append(torch.tensor(seq, dtype=torch.long))
        if len(sequences) >= n_sequences:
            break
    return torch.stack(sequences)


def make_train_batches(tokenizer, val_offset, rank, world_size, micro_batch=MICRO_BATCH):
    """Yield training micro-batches, offset by rank for data parallelism."""
    it = make_data_iterator(tokenizer, split_offset=val_offset)
    batch = []
    seq_idx = 0
    for seq in it:
        if seq_idx % world_size == rank:
            batch.append(torch.tensor(seq, dtype=torch.long))
            if len(batch) == micro_batch:
                yield torch.stack(batch).cuda()
                batch = []
        seq_idx += 1


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step, train_steps):
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, train_steps - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_data):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(val_data), MICRO_BATCH):
        batch = val_data[i:i + MICRO_BATCH].cuda()
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               targets.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# All-reduce for dense (non-expert) params
# ---------------------------------------------------------------------------

def sync_all_grads(model, group):
    """All-reduce ALL gradients (DDP mode — full model replicated)."""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG, group=group)


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------

def train_run(mode_name, base_config, use_fp8, use_compile, ep_size,
              rank, world_size, tokenizer, val_data, val_offset,
              calibrated_max_tokens, train_steps, use_ddp=False,
              micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM, ep_backend="nccl",
              optimizer_type="adamw", mtp_layers=0):
    """Train one configuration for train_steps steps.

    ALL ranks must call this (collective ops inside).
    Non-participating ranks (rank >= ep_size) idle-wait.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    is_main = rank == 0
    participating = rank < ep_size

    # new_group is collective — ALL ranks must call it
    ep_ranks = list(range(ep_size))
    ep_group = dist.new_group(ranks=ep_ranks)

    result = None
    if participating:
        result = _do_train(
            mode_name, base_config, use_fp8, use_compile, ep_size,
            ep_group, rank, world_size, tokenizer, val_data, val_offset,
            calibrated_max_tokens, train_steps, device, is_main,
            use_ddp=use_ddp, micro_batch=micro_batch, grad_accum=grad_accum,
            ep_backend=ep_backend, optimizer_type=optimizer_type,
            mtp_layers=mtp_layers,
        )

    # ALL ranks synchronize here
    dist.barrier()
    return result


def _do_train(mode_name, base_config, use_fp8, use_compile, ep_size,
              ep_group, rank, world_size, tokenizer, val_data, val_offset,
              calibrated_max_tokens, train_steps, device, is_main, use_ddp=False,
              micro_batch=MICRO_BATCH, grad_accum=GRAD_ACCUM, ep_backend="nccl",
              optimizer_type="adamw", mtp_layers=0):
    """Actual training logic — only called by participating ranks.

    use_ddp=True: DDP mode — full model replicated, all grads synced (unfused multi-GPU)
    use_ddp=False: EP mode — experts sharded, only dense grads synced (fused multi-GPU)
    """
    mode_label = "DDP" if use_ddp else "EP"
    tokens_per_step = micro_batch * SEQ_LEN * grad_accum

    if is_main:
        print(f"\n{'='*70}")
        print(f"  {mode_name} | {mode_label} Size: {ep_size} GPUs | "
              f"micro_batch={micro_batch} grad_accum={grad_accum}")
        print(f"{'='*70}")

    # Deterministic seeding
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)

    # Build config — full-size ~500M active / ~1.1B total MoE model
    config = DeepSeekConfig(
        use_moe=True, first_k_dense=1, n_routed_experts=8, n_shared_experts=1,
        moe_ffn_dim=3072, top_k=2, scoring_func="softmax",
        moe_backend=base_config.moe_backend,
        allow_dropped_tokens=True,
        max_tokens_per_expert=calibrated_max_tokens if base_config.moe_backend == "fused" else None,
        mtp_num_layers=mtp_layers,
    )

    model = DeepSeekTransformer(config).to(device)

    if use_fp8:
        swap_linears_to_fp8(model)

    if ep_size > 1 and not use_ddp:
        shard_experts(model, ep_group, ep_backend=ep_backend)

    if use_compile:
        torch._dynamo.config.cache_size_limit = 64
        model = torch.compile(model)

    n_params = count_params(model)
    if is_main:
        print(f"  Parameters per GPU: {n_params/1e6:.1f}M | FP8: {use_fp8} | compile: {use_compile}")

    use_distributed_muon = optimizer_type == "muon" and ep_size > 1
    optimizer = create_optimizer(
        model, optimizer_type=optimizer_type, lr=LR, muon_lr=MUON_LR,
        weight_decay=WEIGHT_DECAY, betas=(BETA1, BETA2),
        distributed=use_distributed_muon,
    )

    # Set up overlapped dense grad sync for EP mode (hooks must be on uncompiled model params)
    grad_syncer = None
    if ep_size > 1 and not use_ddp:
        dense_params = get_dense_params(model, backward_order=True)
        grad_syncer = OverlappedGradSync(dense_params, ep_group, num_buckets=4)
        if is_main:
            print(f"  Overlapped dense sync: {grad_syncer.num_buckets} buckets")

    # Training
    train_iter = make_train_batches(tokenizer, val_offset, rank, ep_size, micro_batch=micro_batch)
    step_times = []
    comm_times = []

    torch.cuda.reset_peak_memory_stats(device)

    for step in range(train_steps):
        lr_scale = get_lr(step, train_steps) / LR  # normalized 0→1→0 schedule
        for pg in optimizer.param_groups:
            base_lr = pg.get("_base_lr", pg["lr"])
            if "_base_lr" not in pg:
                pg["_base_lr"] = pg["lr"]
            pg["lr"] = base_lr * lr_scale

        # Muon momentum warmup/cooldown schedule
        if optimizer_type == "muon" and hasattr(optimizer, "muon") and optimizer.muon:
            mom = _get_muon_momentum(step, train_steps)
            for pg in optimizer.muon.param_groups:
                pg["momentum"] = mom

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()

        # Gradient accumulation loop
        mtp_loss_val = 0.0
        for accum_step in range(grad_accum):
            batch = next(train_iter)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                if mtp_layers > 0:
                    result = model(input_ids, targets=targets)
                    if isinstance(result, tuple):
                        logits, mtp_losses = result
                    else:
                        logits, mtp_losses = result, {}
                else:
                    logits = model(input_ids)
                    mtp_losses = {}
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1))
            if "mtp_loss" in mtp_losses:
                loss = loss + mtp_losses["mtp_loss"]
                mtp_loss_val = mtp_losses["mtp_loss"].item()
            if grad_accum > 1:
                loss = loss / grad_accum
            loss.backward()

        # Sync gradients
        torch.cuda.synchronize(device)
        t_comm_start = time.perf_counter()
        if ep_size > 1:
            if use_ddp:
                sync_all_grads(model, ep_group)
            elif grad_syncer is not None:
                # Overlapped: allreduces already launched by hooks during backward,
                # just wait + unpack
                grad_syncer.finish()
            else:
                # Fallback: should not normally reach here since grad_syncer is always set
                pass
        torch.cuda.synchronize(device)
        t_comm_end = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize(device)
        t1 = time.perf_counter()

        step_ms = (t1 - t0) * 1000
        comm_ms = (t_comm_end - t_comm_start) * 1000
        step_times.append(step_ms)
        comm_times.append(comm_ms)

        if is_main and (step % 5 == 0 or step == train_steps - 1):
            tok_per_sec = tokens_per_step / (step_ms / 1000)
            mtp_str = f" | mtp {mtp_loss_val:.4f}" if mtp_layers > 0 else ""
            print(f"  step {step:>3d} | loss {loss.item():.4f}{mtp_str} | "
                  f"{step_ms:>7.1f} ms | {tok_per_sec:>8.0f} tok/s | "
                  f"comm {comm_ms:.1f} ms")

    # Memory stats
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6

    # Compute metrics (skip warmup)
    timed = step_times[WARMUP_SKIP:]
    comm_timed = comm_times[WARMUP_SKIP:]

    avg_step_ms = sum(timed) / len(timed)
    sorted_times = sorted(timed)
    p50 = sorted_times[len(sorted_times) // 2]
    p95_idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
    p95 = sorted_times[p95_idx]

    avg_comm_ms = sum(comm_timed) / len(comm_timed)
    comm_fraction = avg_comm_ms / avg_step_ms if avg_step_ms > 0 else 0

    per_gpu_tok_s = tokens_per_step / (avg_step_ms / 1000)
    agg_tok_s = per_gpu_tok_s * ep_size

    result = {
        "mode": mode_name,
        "parallel": mode_label,
        "ep_size": ep_size,
        "micro_batch": micro_batch,
        "grad_accum": grad_accum,
        "tokens_per_step": tokens_per_step,
        "n_params_M": n_params / 1e6,
        "avg_step_ms": avg_step_ms,
        "p50_step_ms": p50,
        "p95_step_ms": p95,
        "per_gpu_tok_s": per_gpu_tok_s,
        "agg_tok_s": agg_tok_s,
        "avg_comm_ms": avg_comm_ms,
        "comm_fraction": comm_fraction,
        "peak_mem_mb": peak_mem_mb,
    }

    if is_main:
        print(f"\n  --- {mode_name} {mode_label}={ep_size} Summary ---")
        print(f"  Avg step time:     {avg_step_ms:.1f} ms (p50={p50:.1f}, p95={p95:.1f})")
        print(f"  Per-GPU tok/s:     {per_gpu_tok_s:,.0f}")
        print(f"  Aggregate tok/s:   {agg_tok_s:,.0f}")
        print(f"  Avg comm time:     {avg_comm_ms:.1f} ms ({comm_fraction:.1%} of step)")
        print(f"  Peak memory:       {peak_mem_mb:.0f} MB")

    if grad_syncer is not None:
        grad_syncer.remove_hooks()
    del model, optimizer, grad_syncer
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    # Clean up EP global state for next run
    from flashfusemoe.fused_moe import set_ep_group
    set_ep_group(None)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    parser = argparse.ArgumentParser()
    parser.add_argument("--ep_sizes", default=None,
                        help="Comma-separated EP sizes (default: all powers of 2 up to world_size)")
    parser.add_argument("--modes", default=None,
                        help="Comma-separated modes: unfused_bf16,fused_bf16,unfused_fp8,fused_fp8 (default: all)")
    parser.add_argument("--steps", type=int, default=DEFAULT_TRAIN_STEPS)
    parser.add_argument("--micro_batch", type=int, default=MICRO_BATCH,
                        help="Micro-batch size per GPU (default: 8)")
    parser.add_argument("--large_batch", action="store_true",
                        help="Run large-batch comparison: fused uses 2x micro_batch, "
                             "unfused uses 2x grad_accum to match global batch")
    parser.add_argument("--ep_backend", default="nccl", choices=["nccl", "deep_ep", "hybrid"],
                        help="EP comm backend (default: nccl)")
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "muon"],
                        help="Optimizer type (default: adamw)")
    parser.add_argument("--mtp_layers", type=int, default=0,
                        help="Number of MTP prediction heads (default: 0 = disabled)")
    args = parser.parse_args()

    train_steps = args.steps

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"World size: {world_size}")
        print(f"Tokens per step per GPU: {TOKENS_PER_STEP:,}")
        print(f"Training steps: {train_steps}")

    has_fp8 = _check_fp8()
    if rank == 0:
        print(f"FP8 available: {has_fp8}")

    # EP sizes
    if args.ep_sizes:
        ep_sizes = [int(x) for x in args.ep_sizes.split(",")]
    else:
        ep_sizes = []
        s = 1
        while s <= world_size:
            ep_sizes.append(s)
            s *= 2

    # Modes: (name, base_config, use_fp8, use_compile)
    all_modes = [
        ("Unfused BF16", MOE_UNFUSED, False, False),
        ("Fused BF16", MOE_FUSED, False, True),
        ("Unfused FP8", MOE_UNFUSED, True, False),
        ("Fused FP8", MOE_FUSED, True, True),
    ]

    if args.modes:
        requested = set(args.modes.split(","))
        mode_map = {
            "unfused_bf16": "Unfused BF16",
            "fused_bf16": "Fused BF16",
            "unfused_fp8": "Unfused FP8",
            "fused_fp8": "Fused FP8",
        }
        selected_names = {mode_map[m] for m in requested if m in mode_map}
        modes = [m for m in all_modes if m[0] in selected_names]
    else:
        modes = all_modes

    # Filter out FP8 modes if not available
    if not has_fp8:
        modes = [m for m in modes if not m[2]]

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Validation data
    if rank == 0:
        print(f"Collecting validation data...")
        val_data = collect_val_data(tokenizer, VAL_SEQUENCES)
        print(f"  Val data: {val_data.shape}")
    else:
        val_data = torch.zeros(VAL_SEQUENCES, SEQ_LEN + 1, dtype=torch.long)

    val_data = val_data.cuda()
    dist.broadcast(val_data, src=0)
    val_data = val_data.cpu()

    val_offset = VAL_SEQUENCES * 5

    # Calibrate max_tokens_per_expert for fused modes (rank 0 only, broadcast result)
    calibrated_max_tokens = None
    has_fused = any(m[1].moe_backend == "fused" for m in modes)
    if has_fused:
        if rank == 0:
            print("\n  Calibrating max_tokens_per_expert...")
            torch.manual_seed(42)
            cal_model = DeepSeekTransformer(MOE_FUSED).to(device)
            E = MOE_FUSED.n_routed_experts
            observed_max = 0
            # Use largest micro_batch that will be used
            cal_mb = args.micro_batch * 2 if args.large_batch else args.micro_batch
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                for _ in range(3):
                    x = torch.randn(cal_mb * SEQ_LEN, MOE_FUSED.d_model,
                                    device=device, dtype=torch.bfloat16)
                    for layer in cal_model.layers:
                        if hasattr(layer.ffn, 'gate'):
                            logits = x @ layer.ffn.gate.weight.T
                            _, topk_idx = torch.topk(logits, MOE_FUSED.top_k, dim=-1)
                            tpe = torch.bincount(topk_idx.view(-1), minlength=E)
                            observed_max = max(observed_max, tpe.max().item())
            calibrated_max_tokens = int(observed_max * 1.1)
            del cal_model
            torch.cuda.empty_cache()
            print(f"  Observed max tokens/expert: {observed_max}")
            print(f"  Calibrated buffer size:     {calibrated_max_tokens}")

        # Broadcast calibrated value to all ranks
        cal_tensor = torch.tensor([calibrated_max_tokens or 0], dtype=torch.long, device=device)
        dist.broadcast(cal_tensor, src=0)
        calibrated_max_tokens = cal_tensor.item() if cal_tensor.item() > 0 else None

    # Run all benchmarks
    results = []
    for mode_name, base_config, use_fp8, use_compile in modes:
        for ep_size in ep_sizes:
            if ep_size > world_size:
                if rank == 0:
                    print(f"\nSkipping {mode_name} EP={ep_size} (need {ep_size} GPUs)")
                continue

            # Determine micro_batch and grad_accum for this run
            mb = args.micro_batch
            ga = GRAD_ACCUM
            if args.large_batch:
                if base_config.moe_backend == "fused":
                    # Fused has ~20GB headroom — double the micro-batch
                    mb = args.micro_batch * 2
                    ga = 1
                else:
                    # Unfused can't fit 2x batch — use grad accum instead
                    mb = args.micro_batch
                    ga = 2

            result = train_run(
                mode_name, base_config, use_fp8, use_compile, ep_size,
                rank, world_size, tokenizer, val_data, val_offset,
                calibrated_max_tokens, train_steps, use_ddp=False,
                micro_batch=mb, grad_accum=ga, ep_backend=args.ep_backend,
                optimizer_type=args.optimizer, mtp_layers=args.mtp_layers,
            )
            if result is not None:
                results.append(result)

    # Print comparison table
    if rank == 0 and results:
        print(f"\n{'='*110}")
        table_title = "MULTI-GPU MoE BENCHMARK RESULTS"
        if args.large_batch:
            table_title += " (Large Batch: fused 2x micro_batch, unfused 2x grad_accum)"
        print(f"  {table_title}")
        print(f"{'='*110}")
        print(f"  {'Mode':<16s} | {'EP':>3s} | {'MB×GA':>6s} | {'Tok/step':>9s} | "
              f"{'Agg Tok/s':>10s} | {'Step ms':>8s} | {'p50 ms':>7s} | "
              f"{'Comm %':>7s} | {'Peak MB':>8s}")
        print(f"  {'-'*16}-+-{'-'*3}-+-{'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}")

        for r in results:
            mb_ga = f"{r['micro_batch']}x{r['grad_accum']}"
            print(f"  {r['mode']:<16s} | {r['ep_size']:>3d} | {mb_ga:>6s} | {r['tokens_per_step']:>9,d} | "
                  f"{r['agg_tok_s']:>10,.0f} | {r['avg_step_ms']:>7.1f} | "
                  f"{r['p50_step_ms']:>6.1f} | {r['comm_fraction']:>6.1%} | "
                  f"{r['peak_mem_mb']:>7.0f}")

        # Scaling efficiency for fused modes
        for mode_name, _, _, _ in modes:
            mode_results = [r for r in results if r["mode"] == mode_name]
            if len(mode_results) > 1:
                baseline = mode_results[0]["per_gpu_tok_s"]
                print(f"\n  Scaling efficiency ({mode_name}):")
                for r in mode_results:
                    eff = r["agg_tok_s"] / (r["ep_size"] * baseline) if baseline > 0 else 0
                    print(f"    EP={r['ep_size']:>2d}: {eff:.2%} "
                          f"({r['agg_tok_s']:,.0f} agg tok/s)")

        # Save results
        out_path = os.path.join(os.path.dirname(__file__), "..", "multigpu_results.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
