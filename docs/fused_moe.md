# The Fused MoE Kernel

FlashFuseMoE replaces the standard MoE forward-backward with fused Triton kernels and cuBLAS batched GEMM.

## What gets fused

A standard MoE forward pass looks like this:
1. Gate matmul: `logits = x @ gate.T`
2. Top-k routing: `topk_weights, topk_indices = topk(softmax(logits))`
3. Scatter tokens to experts
4. Per-expert up-projection
5. GLU activation (silu(gate) * up)
6. Per-expert down-projection
7. Gather and weight-combine results

Unfused PyTorch does this with ~23 kernel launches per layer (expert loops, index_add_, scatter, cat, etc). FlashFuseMoE does it in 5 kernel launches:

1. `fused_topk_routing` -- scoring + top-k + weight normalization in 1 Triton kernel
2. `fused_routing_scatter` -- sort tokens into `[E, max_tokens, D]` batched layout (1 Triton kernel)
3. `torch.bmm` x2 -- up-proj and down-proj via cuBLAS batched GEMM
4. `activation_forward` -- SwiGLU/GeGLU/ReLU^2 in 1 Triton kernel
5. `fused_gather_reduce` -- unsort + weighted combine in 1 Triton kernel

Across 23 MoE layers, that's 115 kernel launches vs 529 unfused. 414 fewer per step.

## Batched layout

Tokens are sorted by expert assignment into a padded `[E, max_tokens, D]` tensor. This enables `torch.bmm` for all expert matmuls with no Python loops over experts. The padding overhead is small: `max_tokens` is set by a heuristic (`ceil(N*top_k/E) + 100% margin`) or calibrated from a few forward passes.

cuBLAS bmm was chosen over Triton grouped GEMM because it's 3-6x faster at the matrix sizes typical for MoE models (D=1024-4096, ffn=3072-14336, E_local >= 2).

## torch.compile

The non-EP path wraps `_FusedMoEAutograd` with `@torch.compiler.allow_in_graph`. Dynamo treats the entire MoE layer as an opaque node and compiles everything around it (attention, RMSNorm, embeddings, LM head).

## Backward

The backward pass uses 4 `torch.bmm` calls (2 for activation grads, 2 for weight grads) plus fused Triton kernels for scatter/gather. Routing gradients use a hand-written ~10-op backward instead of tracing through PyTorch autograd (which generates hundreds of small kernels).

Activation checkpointing (`recompute_activations=True`) saves ~35% activation memory by recomputing 3 large intermediate tensors in backward.

## Benchmarks

H100 80GB, ~500M active / ~1.1B total MoE model, BF16, torch.compile, 1 GPU:

| Metric | Unfused | Fused | Speedup |
|--------|---------|-------|---------|
| Step time (p50) | 236 ms | 137 ms | 1.72x |
| Tokens/sec | 33k | 57k | 1.73x |
| Peak memory | 41 GB | 35 GB | 15% less |

The memory savings come from the custom autograd function saving only essential tensors (158 MB/layer vs 740 MB/layer for unfused PyTorch autograd).
