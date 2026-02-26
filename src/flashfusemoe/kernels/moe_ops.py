"""Fused MoE scatter/gather Triton kernels.

Replaces ~21 small PyTorch kernel launches with 4 fused Triton kernels:
1. fused_routing_scatter: Sort tokens into batched [E, max_tokens, D] layout
2. fused_gather_reduce: Gather from batched layout, optionally weight, reduce over top-k
3. fused_weighted_scatter: Scatter weighted grad_output into batched layout
4. fused_routing_weight_grad: Compute routing weight gradient via dot products
"""

import threading

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Workspace buffer cache (thread-local, reuses CUDA allocations)
# ---------------------------------------------------------------------------

_workspace = threading.local()


def _get_zero_buffer(key: str, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return a zeroed buffer, reusing allocation when shapes match."""
    cache = getattr(_workspace, 'buffers', None)
    if cache is None:
        cache = {}
        _workspace.buffers = cache
    buf = cache.get(key)
    if buf is not None and buf.shape == shape and buf.dtype == dtype and buf.device == device:
        buf.zero_()
        return buf
    buf = torch.zeros(shape, dtype=dtype, device=device)
    cache[key] = buf
    return buf


# ---------------------------------------------------------------------------
# Kernel 1: Fused Routing Scatter
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _fused_routing_scatter_kernel(
    sorted_order_ptr,      # [N*top_k] sorted position -> original expanded idx
    flat_expert_ids_ptr,   # [N*top_k] original expanded idx -> expert id
    expert_starts_ptr,     # [E+1] cumulative token counts per expert
    hidden_states_ptr,     # [N, D] input hidden states
    hidden_batched_ptr,    # [E*max_tokens, D] output batched hidden states (pre-zeroed)
    expand_to_batch_ptr,   # [N*top_k] output: original expanded idx -> batch position
    N_topk,                # N * top_k
    D: tl.constexpr,       # hidden dimension
    max_tokens,            # max tokens per expert in batched layout
    top_k,                 # number of experts per token
    E_max_tokens,          # E * max_tokens (total batched size)
    BLOCK_D: tl.constexpr,
):
    """Map sorted positions to batched layout and scatter hidden states.

    Each program handles one sorted position (pid_j) and one D-tile (pid_d).
    """
    pid_j = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_j >= N_topk:
        return

    # Look up which original expanded token this sorted position corresponds to
    orig_idx = tl.load(sorted_order_ptr + pid_j)

    # Which expert does this token go to?
    expert = tl.load(flat_expert_ids_ptr + orig_idx)

    # What rank within this expert?
    expert_start = tl.load(expert_starts_ptr + expert)
    rank = pid_j - expert_start

    # Safety: skip overflow tokens if any expert exceeds max_tokens.
    # Their expand_to_batch stays at default (0), contributing near-zero
    # to the output. Use max_tokens_per_expert for exact correctness.
    if rank >= max_tokens:
        return

    # Compute batch position
    batch_pos = expert * max_tokens + rank

    # Store the expand_to_batch mapping ONLY from first D-tile (avoid redundant writes)
    if pid_d == 0:
        tl.store(expand_to_batch_ptr + orig_idx, batch_pos)

    # Scatter hidden state: hidden_batched[batch_pos, :] = hidden_states[orig_idx // top_k, :]
    token_idx = orig_idx // top_k
    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    src = tl.load(hidden_states_ptr + token_idx * D + offs_d, mask=mask_d, other=0.0)
    tl.store(hidden_batched_ptr + batch_pos * D + offs_d, src, mask=mask_d)


def fused_routing_scatter(
    sorted_order: torch.Tensor,      # [num_sorted] or [N*top_k]
    flat_expert_ids: torch.Tensor,   # [N*top_k]
    expert_starts: torch.Tensor,     # [E+1]
    hidden_states: torch.Tensor,     # [N, D]
    E: int,
    max_tokens: int,
    top_k: int,
    num_sorted: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused routing scatter: sort tokens into batched layout.

    Args:
        num_sorted: Number of entries in sorted_order to process.
            If None, defaults to N * top_k. Use a smaller value when
            capacity dropping has filtered out some tokens.

    Returns:
        hidden_batched: [E*max_tokens, D] batched hidden states
        expand_to_batch: [N*top_k] mapping from original expanded idx to batch position
    """
    N, D = hidden_states.shape
    N_topk = N * top_k
    if num_sorted is None:
        num_sorted = N_topk
    device = hidden_states.device
    dtype = hidden_states.dtype

    hidden_batched = torch.zeros(E * max_tokens, D, dtype=dtype, device=device)
    # Initialize expand_to_batch to 0 so dropped tokens map to position 0
    # (their weights are 0 so they won't affect output)
    expand_to_batch = torch.zeros(N_topk, dtype=torch.long, device=device)

    grid = lambda META: (num_sorted, triton.cdiv(D, META['BLOCK_D']))
    _fused_routing_scatter_kernel[grid](
        sorted_order, flat_expert_ids, expert_starts,
        hidden_states, hidden_batched, expand_to_batch,
        num_sorted, D, max_tokens, top_k, E * max_tokens,
    )

    return hidden_batched, expand_to_batch


# ---------------------------------------------------------------------------
# Kernel 2: Fused Gather-Reduce (autotuned)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D', 'top_k'],
)
@triton.jit
def _fused_gather_reduce_kernel(
    batched_data_ptr,       # [E*max_tokens, D]
    expand_to_batch_ptr,    # [N*top_k]
    weights_ptr,            # [N*top_k] or None
    output_ptr,             # [N, D]
    N,
    D: tl.constexpr,
    top_k: tl.constexpr,
    USE_WEIGHTS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather from batched layout, optionally apply weights, sum over top-k.

    Each program handles one original token (pid_n) and one D-tile (pid_d).
    """
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_n >= N:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    for k in range(top_k):
        expanded_idx = pid_n * top_k + k
        batch_pos = tl.load(expand_to_batch_ptr + expanded_idx)
        val = tl.load(batched_data_ptr + batch_pos * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

        if USE_WEIGHTS:
            w = tl.load(weights_ptr + expanded_idx).to(tl.float32)
            val = val * w

        acc += val

    tl.store(output_ptr + pid_n * D + offs_d, acc.to(output_ptr.dtype.element_ty), mask=mask_d)


def fused_gather_reduce(
    batched_data: torch.Tensor,       # [E*max_tokens, D]
    expand_to_batch: torch.Tensor,    # [N*top_k]
    weights: torch.Tensor | None,     # [N*top_k] or None
    N: int,
    top_k: int,
) -> torch.Tensor:
    """Gather from batched layout, optionally weight, reduce over top-k.

    Args:
        batched_data: [E*max_tokens, D] batched expert outputs
        expand_to_batch: [N*top_k] mapping to batch positions
        weights: [N*top_k] routing weights (None for unweighted sum)
        N: number of original tokens
        top_k: experts per token

    Returns:
        output: [N, D] reduced output
    """
    D = batched_data.shape[-1]
    device = batched_data.device
    dtype = batched_data.dtype

    output = torch.empty(N, D, dtype=dtype, device=device)

    grid = lambda META: (N, triton.cdiv(D, META['BLOCK_D']))

    _fused_gather_reduce_kernel[grid](
        batched_data, expand_to_batch,
        weights if weights is not None else expand_to_batch,  # dummy ptr when not used
        output,
        N, D, top_k,
        USE_WEIGHTS=weights is not None,
    )

    return output


# ---------------------------------------------------------------------------
# Kernel 3: Fused Weighted Scatter (autotuned)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 128}, num_warps=8),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _fused_weighted_scatter_kernel(
    grad_output_ptr,        # [N, D]
    expand_to_batch_ptr,    # [N*top_k]
    flat_weights_ptr,       # [N*top_k]
    output_ptr,             # [E*max_tokens, D] (pre-zeroed)
    N_topk,
    D: tl.constexpr,
    top_k,
    BLOCK_D: tl.constexpr,
):
    """Scatter weighted grad_output into batched layout.

    Each program handles one expanded token j and one D-tile.
    """
    pid_j = tl.program_id(0)
    pid_d = tl.program_id(1)

    if pid_j >= N_topk:
        return

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < D

    batch_pos = tl.load(expand_to_batch_ptr + pid_j)
    token_idx = pid_j // top_k
    w = tl.load(flat_weights_ptr + pid_j).to(tl.float32)

    val = tl.load(grad_output_ptr + token_idx * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
    result = val * w

    tl.store(output_ptr + batch_pos * D + offs_d, result.to(output_ptr.dtype.element_ty), mask=mask_d)


def fused_weighted_scatter(
    grad_output: torch.Tensor,        # [N, D]
    expand_to_batch: torch.Tensor,    # [N*top_k]
    flat_weights: torch.Tensor,       # [N*top_k]
    E: int,
    max_tokens: int,
    top_k: int,
    output_rows: int | None = None,
) -> torch.Tensor:
    """Scatter weighted grad_output into batched layout.

    Args:
        grad_output: [N, D] gradient of output
        expand_to_batch: [N*top_k] mapping to batch positions
        flat_weights: [N*top_k] routing weights
        E: number of experts
        max_tokens: max tokens per expert
        top_k: experts per token
        output_rows: total rows in output (default E*max_tokens). Use E*max_tokens+1
            when capacity dropping redirects dropped tokens to a sentinel position.

    Returns:
        weighted_go_batched: [output_rows, D]
    """
    N, D = grad_output.shape
    N_topk = N * top_k
    device = grad_output.device
    dtype = grad_output.dtype
    if output_rows is None:
        output_rows = E * max_tokens

    # Reuse workspace buffer (backward-only, consumed immediately -- safe to reuse)
    output = _get_zero_buffer('weighted_scatter', (output_rows, D), dtype, device)

    grid = lambda META: (N_topk, triton.cdiv(D, META['BLOCK_D']))

    _fused_weighted_scatter_kernel[grid](
        grad_output, expand_to_batch, flat_weights, output,
        N_topk, D, top_k,
    )

    return output


# ---------------------------------------------------------------------------
# Kernel 4: Fused Routing Weight Gradient (autotuned)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=4),
        triton.Config({'BLOCK_D': 512}, num_warps=4),
        triton.Config({'BLOCK_D': 256}, num_warps=8),
    ],
    key=['D', 'top_k'],
    reset_to_zero=['grad_topk_weights_ptr'],
)
@triton.jit
def _fused_routing_weight_grad_kernel(
    down_batched_ptr,       # [E*max_tokens, D]
    expand_to_batch_ptr,    # [N*top_k]
    grad_output_ptr,        # [N, D]
    grad_topk_weights_ptr,  # [N, top_k] output (float32)
    N,
    D: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute routing weight gradient via dot products.

    For each token i and expert slot k:
        grad_topk_weights[i, k] = dot(down_batched[batch_pos, :], grad_output[i, :])

    Each program handles one original token with full D reduction (no atomics).
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    for k in range(top_k):
        expanded_idx = pid * top_k + k
        batch_pos = tl.load(expand_to_batch_ptr + expanded_idx)

        # Tiled dot product across full D dimension
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for d_off in tl.range(0, D, BLOCK_D):
            offs_d = d_off + tl.arange(0, BLOCK_D)
            mask_d = offs_d < D

            down_val = tl.load(down_batched_ptr + batch_pos * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)
            go_val = tl.load(grad_output_ptr + pid * D + offs_d, mask=mask_d, other=0.0).to(tl.float32)

            acc += down_val * go_val

        result = tl.sum(acc)
        tl.store(grad_topk_weights_ptr + pid * top_k + k, result)


def fused_routing_weight_grad(
    down_batched: torch.Tensor,       # [E*max_tokens, D]
    expand_to_batch: torch.Tensor,    # [N*top_k]
    grad_output: torch.Tensor,        # [N, D]
    N: int,
    top_k: int,
) -> torch.Tensor:
    """Compute routing weight gradient via fused dot products.

    Args:
        down_batched: [E*max_tokens, D] down-projection outputs in batched layout
        expand_to_batch: [N*top_k] mapping to batch positions
        grad_output: [N, D] gradient of final output
        N: number of original tokens
        top_k: experts per token

    Returns:
        grad_topk_weights: [N, top_k] in float32
    """
    D = down_batched.shape[-1]
    device = down_batched.device

    grad_topk_weights = torch.zeros(N, top_k, dtype=torch.float32, device=device)

    grid = (N,)
    _fused_routing_weight_grad_kernel[grid](
        down_batched, expand_to_batch, grad_output, grad_topk_weights,
        N, D, top_k,
    )

    return grad_topk_weights
