# Triton Kernel Details

FlashFuseMoE includes 12 Triton kernels that fuse multiple PyTorch operations into single GPU kernel launches.

## Activation kernels (`kernels/activations.py`)

Two kernels: forward and backward for GLU activations.

**`activation_forward`**: Fuses `silu(gate) * up` (SwiGLU), `gelu(gate) * up` (GeGLU), or `relu(gate)^2 * up` (ReLU^2) into a single kernel. Input `[*, ffn_dim]` splits at midpoint into gate and up halves. Replaces 3 separate PyTorch ops.

**`activation_backward`**: Computes both gate and up gradients in one pass. Writes directly to `[:, :half]` and `[:, half:]` of the output tensor, eliminating a `torch.cat` allocation (114 MB per call at typical sizes).

## Routing kernels (`kernels/moe_ops.py`)

Four kernels that replace ~21 PyTorch operations with 4 Triton kernel launches.

**`fused_routing_scatter`**: Takes `[N, D]` hidden states + routing metadata (expert assignments, sort order, per-expert counts) and scatters tokens into `[E, max_tokens, D]` batched layout. Replaces ~9 PyTorch ops (argsort, expand, index_select, cumsum, etc).

**`fused_gather_reduce`**: Gathers expert outputs from `[E*max_tokens, D]` back to `[N, D]` using the expand_to_batch mapping. Replaces 5 scatter-back ops.

**`fused_weighted_scatter`**: Same as fused_routing_scatter but for the backward pass where gradient values need to be weighted. Replaces 5 ops.

**`fused_routing_weight_grad`**: Computes routing weight gradients. Replaces 3 ops.

## Fused top-k routing (`kernels/fused_topk.py`)

**`fused_topk_routing`**: Fuses scoring + top-k selection + weight normalization. Supports both softmax and sigmoid gating.

- Softmax path: 2 PyTorch ops to 1 kernel
- Sigmoid path: 4 PyTorch ops to 1 kernel
- Performance: 13.8us vs 30.7us PyTorch (2.2x faster)
- Saves ~0.8ms across 23 MoE layers per step

## Fused dispatch pack (`kernels/fused_dispatch.py`)

**`fused_dispatch_pack`**: Packs tokens + expert IDs into a single `[M, D+1]` buffer for EP all-to-all dispatch. Replaces 3 ops (expand, index, concatenate) with 1 kernel.

## Grouped GEMM (`kernels/moe_grouped_gemm.py`)

Persistent Triton grouped GEMM kernel. Not used in the default path (cuBLAS bmm is faster at typical sizes) but available for the `E_local=1` case where bmm has 50% padding waste.

## Forward kernel (`kernels/fused_moe_forward.py`)

Fused forward kernel that combines routing + up-proj + activation + down-proj. Used in some benchmark configurations.

## Weight gradient kernel (`kernels/moe_weight_grad.py`)

**`moe_weight_grad`**: Computes per-expert weight gradients `grad_W[e] = grad_out[e].T @ input[e]` using expert boundaries for efficient per-expert accumulation. Used in the grouped GEMM backward path.

## Token alignment (`ops/moe_align.py`)

**`moe_align_block_size_vectorized`**: Sorts tokens by expert assignment and pads to block size boundaries for grouped GEMM. Vectorized version avoids per-expert Python loops.
