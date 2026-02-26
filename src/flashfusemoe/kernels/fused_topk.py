"""Fused top-k routing Triton kernel.

Fuses scoring function + top-k selection + weight normalization into a single
kernel launch, replacing 2-4 separate PyTorch ops in _apply_routing().

Supports:
- Softmax gating: topk(logits) → softmax(selected) — 2 ops → 1 kernel
- Sigmoid gating: sigmoid(logits) → topk(scores) → gather → normalize — 4 ops → 1 kernel
- Arbitrary E (any number of experts, not just power-of-2)
- Arbitrary top_k (1 to 8+)

Falls back to PyTorch for advanced features (group routing, expert bias, pre_softmax).

Reference: Megatron-LM's fused_topk_with_score_function() in moe_utils.py
uses a TE CUDA kernel for the same purpose. We use Triton for portability.
"""

import torch
import triton
import triton.language as tl


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


@triton.jit
def _fused_topk_softmax_kernel(
    logits_ptr,         # [N, E] input logits
    topk_weights_ptr,   # [N, top_k] output weights (float32)
    topk_indices_ptr,   # [N, top_k] output indices (int64)
    N,                  # number of tokens
    E: tl.constexpr,    # actual number of experts
    BLOCK_E: tl.constexpr,  # padded to power of 2
    TOP_K: tl.constexpr,    # actual top-k
    BLOCK_K: tl.constexpr,  # padded to power of 2
):
    """Fused softmax gating: topk(logits) → softmax(selected).

    Each program handles one token. BLOCK_E and BLOCK_K are power-of-2
    padded versions of E and TOP_K for Triton arange compatibility.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load E logits, pad invalid experts with -inf
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < E
    logits = tl.load(logits_ptr + pid * E + offs_e, mask=e_mask, other=-1e30).to(tl.float32)

    # --- Top-k via iterative argmax ---
    NEG_INF: tl.constexpr = -1e30
    exclude = tl.zeros([BLOCK_E], dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    selected_logits = tl.zeros([BLOCK_K], dtype=tl.float32)
    selected_indices = tl.zeros([BLOCK_K], dtype=tl.int64)

    for k in tl.static_range(TOP_K):
        masked_logits = logits + exclude
        idx = tl.argmax(masked_logits, axis=0)
        # Extract the value at idx using reduction (avoids scalar load)
        val = tl.sum(tl.where(offs_e == idx, logits, tl.zeros([BLOCK_E], dtype=tl.float32)), axis=0)
        selected_logits = tl.where(offs_k == k, val, selected_logits)
        selected_indices = tl.where(offs_k == k, idx.to(tl.int64), selected_indices)
        exclude = tl.where(offs_e == idx, NEG_INF, exclude)

    # --- Softmax over selected logits (only TOP_K valid entries) ---
    k_mask = offs_k < TOP_K
    valid_logits = tl.where(k_mask, selected_logits, NEG_INF)
    max_val = tl.max(valid_logits, axis=0)
    exp_vals = tl.where(k_mask, tl.exp(selected_logits - max_val), 0.0)
    sum_exp = tl.sum(exp_vals, axis=0)
    weights = exp_vals / sum_exp

    # Store results (only TOP_K entries)
    tl.store(topk_weights_ptr + pid * TOP_K + offs_k, weights, mask=k_mask)
    tl.store(topk_indices_ptr + pid * TOP_K + offs_k, selected_indices, mask=k_mask)


@triton.jit
def _fused_topk_sigmoid_kernel(
    logits_ptr,         # [N, E] input logits
    topk_weights_ptr,   # [N, top_k] output weights (float32)
    topk_indices_ptr,   # [N, top_k] output indices (int64)
    N,                  # number of tokens
    E: tl.constexpr,    # actual number of experts
    BLOCK_E: tl.constexpr,  # padded to power of 2
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
    EPS: tl.constexpr,
):
    """Fused sigmoid gating: sigmoid(logits) → topk → gather → normalize.

    Each program handles one token.
    """
    pid = tl.program_id(0)
    if pid >= N:
        return

    # Load E logits with masking, compute sigmoid
    offs_e = tl.arange(0, BLOCK_E)
    e_mask = offs_e < E
    logits = tl.load(logits_ptr + pid * E + offs_e, mask=e_mask, other=-1e30).to(tl.float32)
    # sigmoid(-1e30) ≈ 0, so padded entries won't be selected
    scores = tl.sigmoid(logits)

    # --- Top-k via iterative argmax ---
    NEG_INF: tl.constexpr = -1e30
    exclude = tl.where(e_mask, 0.0, NEG_INF)

    offs_k = tl.arange(0, BLOCK_K)
    selected_scores = tl.zeros([BLOCK_K], dtype=tl.float32)
    selected_indices = tl.zeros([BLOCK_K], dtype=tl.int64)

    for k in tl.static_range(TOP_K):
        masked_scores = scores + exclude
        idx = tl.argmax(masked_scores, axis=0)
        val = tl.sum(tl.where(offs_e == idx, scores, tl.zeros([BLOCK_E], dtype=tl.float32)), axis=0)
        selected_scores = tl.where(offs_k == k, val, selected_scores)
        selected_indices = tl.where(offs_k == k, idx.to(tl.int64), selected_indices)
        exclude = tl.where(offs_e == idx, NEG_INF, exclude)

    # --- Normalize: weights = selected / sum(selected) ---
    k_mask = offs_k < TOP_K
    valid_scores = tl.where(k_mask, selected_scores, 0.0)
    weight_sum = tl.sum(valid_scores, axis=0) + EPS
    weights = valid_scores / weight_sum

    # Store results
    tl.store(topk_weights_ptr + pid * TOP_K + offs_k, weights, mask=k_mask)
    tl.store(topk_indices_ptr + pid * TOP_K + offs_k, selected_indices, mask=k_mask)


def fused_topk_routing(logits, gating, top_k):
    """Fused top-k routing: scoring + selection + normalization in one kernel.

    Handles the common case (no groups, no bias, no pre_softmax).
    The caller should fall back to _apply_routing() for advanced features.

    Args:
        logits: [N, E] raw gate logits (any dtype, computed in float32 internally)
        gating: "softmax" or "sigmoid"
        top_k: number of experts to select (any value, typically 1-8)

    Returns:
        topk_weights: [N, top_k] float32 normalized weights
        topk_indices: [N, top_k] int64 expert indices
    """
    N, E = logits.shape
    device = logits.device

    # Triton requires power-of-2 arange sizes
    BLOCK_E = _next_power_of_2(E)
    BLOCK_K = _next_power_of_2(top_k)

    topk_weights = torch.empty(N, top_k, dtype=torch.float32, device=device)
    topk_indices = torch.empty(N, top_k, dtype=torch.int64, device=device)

    grid = (N,)

    if gating == "softmax":
        _fused_topk_softmax_kernel[grid](
            logits, topk_weights, topk_indices,
            N, E, BLOCK_E, top_k, BLOCK_K,
        )
    elif gating == "sigmoid":
        _fused_topk_sigmoid_kernel[grid](
            logits, topk_weights, topk_indices,
            N, E, BLOCK_E, top_k, BLOCK_K, 1e-6,
        )
    else:
        raise ValueError(f"Unknown gating: {gating}")

    return topk_weights, topk_indices
