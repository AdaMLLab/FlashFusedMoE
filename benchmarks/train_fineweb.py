"""Train MoE Fused vs Unfused on FineWeb-Edu (streamed).

Comparison matrix:
  1. MoE Unfused BF16    2. MoE Fused BF16
  3. MoE Unfused FP8     4. MoE Fused FP8

Reports: val_loss, throughput (tokens/sec), step times.

Usage:
    python benchmarks/train_fineweb.py
    python benchmarks/train_fineweb.py --mode unfused_bf16
    python benchmarks/train_fineweb.py --mode fused_bf16
    python benchmarks/train_fineweb.py --mode all
"""

import argparse
import json
import math
import os
import random
import time

import tiktoken
import torch
import torch.nn.functional as F
from datasets import load_dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "examples"))
from deepseek.model import (
    DeepSeekTransformer, DeepSeekConfig, MOE_UNFUSED, MOE_FUSED, count_params,
)


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

TRAIN_STEPS = 20
SEQ_LEN = 1024
TOKENS_PER_BATCH = 204_800          # ~200K tokens
MICRO_BATCH = 8                      # sequences per micro-batch
GRAD_ACCUM = TOKENS_PER_BATCH // (MICRO_BATCH * SEQ_LEN)  # = 25

LR = 3e-4
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP = 1.0

VAL_SEQUENCES = 64                   # ~64K tokens for validation
VAL_EVERY = 10                       # evaluate every N steps
DEVICE = "cuda"


# ---------------------------------------------------------------------------
# FP8 support via torch._scaled_mm
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
        # Cast to same dtype (weight is FP32 param, grad is BF16 under autocast)
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
# Data pipeline: stream FineWeb-Edu → tokenize → batch
# ---------------------------------------------------------------------------

def make_data_iterator(tokenizer, split_offset=0):
    """Stream FineWeb-Edu sample-10BT, tokenize, yield [seq_len+1] chunks."""
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
    """Collect fixed validation set from the first examples."""
    print(f"  Collecting {n_sequences} validation sequences...")
    sequences = []
    it = make_data_iterator(tokenizer, split_offset=0)
    for seq in it:
        sequences.append(torch.tensor(seq, dtype=torch.long))
        if len(sequences) >= n_sequences:
            break
    return torch.stack(sequences)  # [n_sequences, seq_len+1]


def make_train_batches(tokenizer, val_offset):
    """Yield training micro-batches (skip val examples)."""
    it = make_data_iterator(tokenizer, split_offset=val_offset)
    batch = []
    for seq in it:
        batch.append(torch.tensor(seq, dtype=torch.long))
        if len(batch) == MICRO_BATCH:
            yield torch.stack(batch).to(DEVICE)  # [micro_batch, seq_len+1]
            batch = []


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step):
    """Cosine decay with linear warmup."""
    if step < WARMUP_STEPS:
        return LR * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(1, TRAIN_STEPS - WARMUP_STEPS)
    return LR * 0.5 * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, val_data, compute_dtype):
    """Compute validation loss on fixed held-out set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for i in range(0, len(val_data), MICRO_BATCH):
        batch = val_data[i:i + MICRO_BATCH].to(DEVICE)
        input_ids = batch[:, :-1]
        targets = batch[:, 1:]
        with torch.amp.autocast("cuda", dtype=compute_dtype):
            logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               targets.reshape(-1))
        total_loss += loss.item()
        n_batches += 1
    model.train()
    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_run(model_config, mode_name, compute_dtype, use_fp8, use_compile,
              tokenizer, val_data, train_iter):
    """Train one configuration for TRAIN_STEPS steps."""
    print(f"\n{'='*70}")
    print(f"  {mode_name}")
    print(f"{'='*70}")

    # Full deterministic seeding for reproducibility across runs
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    model = DeepSeekTransformer(model_config).to(DEVICE)
    n_params = count_params(model)
    print(f"  Parameters: {n_params/1e6:.1f}M")
    print(f"  Compute dtype: {compute_dtype}")
    print(f"  MoE backend: {model_config.moe_backend}")
    print(f"  FP8 linears: {use_fp8}")

    if use_fp8:
        swap_linears_to_fp8(model)

    if use_compile:
        model = torch.compile(model)
        print("  torch.compile: enabled")
    else:
        print("  torch.compile: disabled")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, betas=(BETA1, BETA2),
        weight_decay=WEIGHT_DECAY, fused=True,
    )

    # Initial val loss
    val_loss = evaluate(model, val_data, compute_dtype)
    print(f"  Initial val_loss: {val_loss:.4f}")

    # Training
    step_times = []
    val_losses = [val_loss]
    train_losses = []

    for step in range(TRAIN_STEPS):
        lr = get_lr(step)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Gradient accumulation
        total_loss = 0.0
        for micro_step in range(GRAD_ACCUM):
            batch = next(train_iter)
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            with torch.amp.autocast("cuda", dtype=compute_dtype):
                logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   targets.reshape(-1))
            loss = loss / GRAD_ACCUM
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_ms = (t1 - t0) * 1000
        step_times.append(step_ms)
        train_losses.append(total_loss)

        # Evaluation
        if (step + 1) % VAL_EVERY == 0 or step == TRAIN_STEPS - 1:
            val_loss = evaluate(model, val_data, compute_dtype)
            val_losses.append(val_loss)

        if step % 5 == 0 or step == TRAIN_STEPS - 1:
            tok_per_sec = TOKENS_PER_BATCH / (step_ms / 1000)
            recent_val = val_losses[-1]
            print(f"  step {step:>4d} | loss {total_loss:.4f} | val {recent_val:.4f} | "
                  f"{step_ms:>7.1f} ms | {tok_per_sec:>8.0f} tok/s | lr {lr:.2e}")

    # Summary
    skip = min(10, len(step_times) - 1)  # skip warmup steps but keep at least 1
    avg_step_ms = sum(step_times[skip:]) / max(len(step_times) - skip, 1)
    avg_tok_per_sec = TOKENS_PER_BATCH / (avg_step_ms / 1000)

    result = {
        "mode": mode_name,
        "n_params_M": n_params / 1e6,
        "compute_dtype": str(compute_dtype),
        "moe_backend": model_config.moe_backend,
        "fp8_linears": use_fp8,
        "initial_val_loss": val_losses[0],
        "final_val_loss": val_losses[-1],
        "final_train_loss": train_losses[-1],
        "avg_step_ms": avg_step_ms,
        "avg_tok_per_sec": avg_tok_per_sec,
        "val_losses": val_losses,
    }

    print(f"\n  --- {mode_name} Summary ---")
    print(f"  Final val_loss:  {val_losses[-1]:.4f}")
    print(f"  Avg step time:   {avg_step_ms:.1f} ms")
    print(f"  Avg throughput:  {avg_tok_per_sec:,.0f} tok/s")
    print(f"  Tokens trained:  {TRAIN_STEPS * TOKENS_PER_BATCH:,}")

    del model, optimizer
    torch.cuda.empty_cache()

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global TRAIN_STEPS

    # Must be set before first CUDA call for deterministic cuBLAS
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="all",
                        choices=["unfused_bf16", "fused_bf16",
                                 "unfused_fp8", "fused_fp8",
                                 "bf16", "fp8", "all"])
    parser.add_argument("--steps", type=int, default=TRAIN_STEPS)
    args = parser.parse_args()

    TRAIN_STEPS = args.steps

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Tokens per batch: {TOKENS_PER_BATCH:,} ({MICRO_BATCH}×{GRAD_ACCUM}×{SEQ_LEN})")
    print(f"Training steps: {TRAIN_STEPS}")
    print(f"Total tokens: {TRAIN_STEPS * TOKENS_PER_BATCH:,}")

    has_fp8 = _check_fp8()
    print(f"FP8 _scaled_mm available: {has_fp8}")

    # Tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    print(f"Tokenizer: GPT-2 ({tokenizer.n_vocab} tokens)")

    # Validation data
    val_data = collect_val_data(tokenizer, VAL_SEQUENCES)
    val_offset = VAL_SEQUENCES * 5  # skip more than we used
    print(f"  Val data: {val_data.shape}")

    # --- Calibrate max_tokens_per_expert for fused runs ---
    # Run a few forward passes with a throwaway model to measure actual routing
    # distribution, then set max_tokens_per_expert = observed_max * 1.1 (10% headroom).
    # This avoids the heuristic overallocation and ensures no graph breaks.
    calibrated_max_tokens = None
    if args.mode in ("fused_bf16", "fused_fp8", "bf16", "fp8", "all"):
        print("\n  Calibrating max_tokens_per_expert...")
        import torch.nn.functional as _F
        cal_model = DeepSeekTransformer(MOE_FUSED).to(DEVICE)
        cal_iter = make_train_batches(tokenizer, val_offset)
        observed_max = 0
        E = MOE_FUSED.n_routed_experts
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for cal_step in range(3):
                batch = next(cal_iter)
                input_ids = batch[:, :-1]
                # Manually measure routing distribution
                for layer in cal_model.layers:
                    if hasattr(layer.ffn, 'gate'):
                        x = torch.randn(input_ids.shape[0] * input_ids.shape[1],
                                        MOE_FUSED.d_model, device=DEVICE, dtype=torch.bfloat16)
                        logits = x @ layer.ffn.gate.weight.T
                        _, topk_idx = torch.topk(logits, MOE_FUSED.top_k, dim=-1)
                        tpe = torch.bincount(topk_idx.view(-1), minlength=E)
                        observed_max = max(observed_max, tpe.max().item())
        calibrated_max_tokens = int(observed_max * 1.1)  # 10% headroom
        del cal_model, cal_iter
        torch.cuda.empty_cache()
        print(f"  Observed max tokens/expert: {observed_max}")
        print(f"  Calibrated buffer size:     {calibrated_max_tokens}")

    MOE_FUSED_CAL = DeepSeekConfig(
        use_moe=True, first_k_dense=1, n_routed_experts=8, n_shared_experts=1,
        moe_ffn_dim=3072, top_k=2, scoring_func="softmax", moe_backend="fused",
        max_tokens_per_expert=calibrated_max_tokens,
    )

    # Define runs: (name, config, dtype, use_fp8, use_compile)
    # Fused MoE uses @torch.compiler.allow_in_graph so torch.compile works
    runs = []
    if args.mode in ("unfused_bf16", "bf16", "all"):
        runs.append(("MoE Unfused BF16", MOE_UNFUSED, torch.bfloat16, False, False))
    if args.mode in ("fused_bf16", "bf16", "all"):
        runs.append(("MoE Fused BF16", MOE_FUSED_CAL, torch.bfloat16, False, True))
    if args.mode in ("unfused_fp8", "fp8", "all") and has_fp8:
        runs.append(("MoE Unfused FP8", MOE_UNFUSED, torch.bfloat16, True, False))
    if args.mode in ("fused_fp8", "fp8", "all") and has_fp8:
        runs.append(("MoE Fused FP8", MOE_FUSED_CAL, torch.bfloat16, True, True))

    results = []
    for mode_name, config, dtype, use_fp8, use_compile in runs:
        # Fresh training data iterator for each run
        train_iter = make_train_batches(tokenizer, val_offset)
        result = train_run(config, mode_name, dtype, use_fp8, use_compile,
                           tokenizer, val_data, train_iter)
        results.append(result)

    # Final comparison table
    print(f"\n{'='*70}")
    print("  COMPARISON: Fused vs Unfused MoE Kernels")
    print(f"{'='*70}")
    print(f"  {'Mode':<20s} | {'Params':>7s} | {'Val Loss':>9s} | {'Tok/s':>10s} | {'Step ms':>8s}")
    print(f"  {'-'*20}-+-{'-'*7}-+-{'-'*9}-+-{'-'*10}-+-{'-'*8}")
    for r in results:
        print(f"  {r['mode']:<20s} | {r['n_params_M']:>6.1f}M | {r['final_val_loss']:>9.4f} | "
              f"{r['avg_tok_per_sec']:>10,.0f} | {r['avg_step_ms']:>7.1f}")

    # Compute speedups
    bf16_results = [r for r in results if "BF16" in r["mode"]]
    fp8_results = [r for r in results if "FP8" in r["mode"]]

    for group_name, group in [("BF16", bf16_results), ("FP8", fp8_results)]:
        if len(group) == 2:
            unfused = [r for r in group if "Unfused" in r["mode"]][0]
            fused = [r for r in group if "Fused" in r["mode"]][0]
            speedup = unfused["avg_step_ms"] / fused["avg_step_ms"]
            print(f"\n  {group_name} Fused vs Unfused:")
            print(f"    Throughput speedup: {speedup:.2f}x")
            print(f"    Val loss (unfused): {unfused['final_val_loss']:.4f}")
            print(f"    Val loss (fused):   {fused['final_val_loss']:.4f}")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "..", "fineweb_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
