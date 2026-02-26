"""Fused MoE: gate -> top-k route -> GLU experts -> weighted combine.

Single-function API for fused Mixture-of-Experts with:
- Fused Triton scatter/gather kernels
- cuBLAS batched GEMM (torch.bmm)
- Fused GLU activations (SwiGLU, GeGLU, ReLU²)
- Softmax or sigmoid gating
- Dynamic overflow detection with retry
- Expert capacity with token dropping
- Load-balancing auxiliary loss (Switch Transformer style)
- Activation checkpointing (recompute 3 large tensors in backward)

Usage:
    from flashfusemoe import fused_moe
    output, losses = fused_moe(hidden_states, gate_weight, w_up, w_down, top_k=2)
"""

import torch
import torch.nn.functional as F

from flashfusemoe.kernels.activations import activation_forward, activation_backward
from flashfusemoe.kernels.fused_topk import fused_topk_routing
from flashfusemoe.kernels.moe_ops import (
    fused_routing_scatter, fused_gather_reduce,
    fused_weighted_scatter,
)
from flashfusemoe.expert_parallel import ExpertParallelDispatcher, DispatchHandle, create_dispatcher

_WEIGHT_EPS = 1e-6  # Division-by-zero guard for weight normalization

# Module-level stream cache for shared expert overlap (Phase 3a)
_shared_streams: dict[int, torch.cuda.Stream] = {}
# Module-level stream cache for A2A overlap in backward
_a2a_streams: dict[int, torch.cuda.Stream] = {}
# Module-level dispatcher cache (avoids object creation per call)
_dispatcher_cache: dict[tuple, "ExpertParallelDispatcher"] = {}


def _get_shared_stream():
    """Get or create a high-priority CUDA stream for shared expert overlap."""
    idx = torch.cuda.current_device()
    if idx not in _shared_streams:
        _shared_streams[idx] = torch.cuda.Stream(device=idx, priority=-1)
    return _shared_streams[idx]


def _get_a2a_stream():
    """Get or create a CUDA stream for A2A overlap in backward."""
    idx = torch.cuda.current_device()
    if idx not in _a2a_streams:
        _a2a_streams[idx] = torch.cuda.Stream(device=idx)
    return _a2a_streams[idx]


def _get_dispatcher(E, ep_size, ep_group, backend="nccl"):
    """Get or create an EP dispatcher (cached by config)."""
    key = (E, ep_size, id(ep_group), backend)
    if key not in _dispatcher_cache:
        _dispatcher_cache[key] = create_dispatcher(E, ep_size, ep_group, backend=backend)
    return _dispatcher_cache[key]


# Module-level storage for EP process group.
# ProcessGroup can't be a dynamo graph node, so we store it here and
# the EP autograd function reads it directly. Set by set_ep_group()
# before calling fused_moe(). Safe because each rank is a separate process.
_EP_GROUP = None
_EP_ACTIVE = False  # True when set_ep_group() was called with a non-None group
_EP_BACKEND = "nccl"  # EP comm backend: "nccl", "deep_ep", "hybrid"
_fc = None  # Lazy import of torch.distributed._functional_collectives

# Module-level routing config (set by fused_moe() before calling autograd).
# These are routing enhancement parameters that shouldn't be graph nodes.
_ROUTING_CFG = {
    "n_group": None,
    "topk_group": None,
    "scoring_factor": None,
    "pre_softmax": False,
    "input_jitter": 0.0,
    "expert_bias": None,
}


def _get_fc():
    """Lazy import functional collectives to avoid side effects on single-GPU."""
    global _fc
    if _fc is None:
        import torch.distributed._functional_collectives as fc
        _fc = fc
    return _fc


def _get_compute_dtype(fallback_dtype):
    """Get the autocast compute dtype if active, otherwise use fallback."""
    if torch.is_autocast_enabled('cuda'):
        return torch.get_autocast_dtype('cuda')
    return fallback_dtype


def _estimate_max_tokens(N, top_k, E):
    """Static upper bound for max tokens per expert — avoids GPU->CPU sync.

    Uses ceil(N*top_k/E) + 100% margin for imbalance. No GPU sync needed.
    The kernel has a safety guard that silently skips tokens if the buffer
    is exceeded. For exact correctness under heavy routing skew, use the
    max_tokens_per_expert parameter (set via calibration) instead.
    """
    max_tokens = (N * top_k + E - 1) // E  # ceil division = perfect balance
    max_tokens = max_tokens + max(max_tokens, top_k)  # +100% margin
    # For very small N, use absolute upper bound
    if N * top_k <= 4 * E:
        max_tokens = N * top_k
    return max_tokens


def _apply_routing(logits, gating, top_k, n_group=None, topk_group=None,
                   scoring_factor=None, pre_softmax=False, expert_bias=None):
    """Compute routing weights and indices from logits.

    Supports DeepSeek-V2/V3 routing enhancements:
    - pre_softmax: Apply softmax before top-k selection
    - scoring_factor: Scale weights by factor/top_k after selection
    - expert_bias: Aux-loss-free routing bias (added before top-k, not to weights)
    - n_group + topk_group: Group-limited top-k routing

    Uses fused Triton kernel for the common path (no groups, no bias, no pre_softmax).
    Falls back to PyTorch ops for advanced features.

    Returns:
        topk_weights: [N, top_k] float32
        topk_indices: [N, top_k] int64
    """
    E = logits.shape[1]

    # Fast path: fused Triton kernel for common case
    # Skip when tracing with FakeTensors (torch.compile allow_in_graph path)
    has_advanced = (n_group is not None or expert_bias is not None or pre_softmax
                    or scoring_factor is not None)
    if (not has_advanced and logits.is_cuda
            and not isinstance(logits, torch._subclasses.FakeTensor)):
        topk_weights, topk_indices = fused_topk_routing(logits, gating, top_k)
        return topk_weights, topk_indices

    # Slow path: PyTorch ops for advanced routing features
    if gating == "softmax":
        if pre_softmax:
            # Softmax before top-k (DeepSeek-V3 style)
            all_probs = F.softmax(logits.float(), dim=-1)
            if expert_bias is not None:
                all_probs_biased = all_probs + expert_bias.float().unsqueeze(0)
            else:
                all_probs_biased = all_probs

            if n_group is not None and topk_group is not None:
                topk_weights, topk_indices = _group_limited_topk(
                    all_probs_biased, all_probs, top_k, n_group, topk_group)
            else:
                _, topk_indices = torch.topk(all_probs_biased, top_k, dim=-1)
                topk_weights = all_probs.gather(1, topk_indices)

            # Renormalize selected weights
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + _WEIGHT_EPS)
        else:
            # Standard: top-k then softmax on selected logits
            if expert_bias is not None:
                logits_biased = logits + expert_bias.unsqueeze(0)
            else:
                logits_biased = logits

            if n_group is not None and topk_group is not None:
                topk_weights, topk_indices = _group_limited_topk(
                    logits_biased, logits, top_k, n_group, topk_group)
                topk_weights = F.softmax(topk_weights.float(), dim=-1)
            else:
                topk_logits, topk_indices = torch.topk(logits_biased, top_k, dim=-1)
                # Use unbiased logits for weight computation
                if expert_bias is not None:
                    topk_logits = logits.gather(1, topk_indices)
                topk_weights = F.softmax(topk_logits.float(), dim=-1)

    elif gating == "sigmoid":
        scores = torch.sigmoid(logits.float())
        if expert_bias is not None:
            scores_biased = scores + expert_bias.float().unsqueeze(0)
        else:
            scores_biased = scores

        if n_group is not None and topk_group is not None:
            topk_weights, topk_indices = _group_limited_topk(
                scores_biased, scores, top_k, n_group, topk_group)
        else:
            _, topk_indices = torch.topk(scores_biased, top_k, dim=-1)
            topk_weights = scores.gather(1, topk_indices)

        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + _WEIGHT_EPS)

    if scoring_factor is not None:
        topk_weights = topk_weights * (scoring_factor / top_k)

    return topk_weights, topk_indices


def _group_limited_topk(scores_for_selection, scores_for_weights, top_k, n_group, topk_group):
    """Group-limited top-k routing (DeepSeek-V2/V3).

    Divides experts into n_group groups, selects the top topk_group groups per token,
    then selects top_k experts from those groups only.

    Args:
        scores_for_selection: [N, E] scores used to select experts (may include bias)
        scores_for_weights: [N, E] unbiased scores used for weight computation
        top_k: number of experts to select
        n_group: number of expert groups
        topk_group: number of groups to select per token

    Returns:
        topk_weights: [N, top_k] from unbiased scores
        topk_indices: [N, top_k] selected expert indices
    """
    N, E = scores_for_selection.shape
    assert E % n_group == 0, f"E ({E}) must be divisible by n_group ({n_group})"
    experts_per_group = E // n_group

    # Reshape to [N, n_group, experts_per_group]
    grouped = scores_for_selection.view(N, n_group, experts_per_group)

    # Select top groups by sum of top-(top_k // topk_group) scores per group
    # (Megatron-LM approach: more stable than max-based scoring)
    scores_per_group = top_k // topk_group
    if scores_per_group > 1 and scores_per_group <= experts_per_group:
        group_scores = grouped.topk(scores_per_group, dim=-1).values.sum(dim=-1)  # [N, n_group]
    else:
        group_scores = grouped.max(dim=-1).values  # [N, n_group]
    _, top_group_indices = torch.topk(group_scores, topk_group, dim=-1)  # [N, topk_group]

    # Create mask for selected groups
    group_mask = torch.zeros(N, n_group, dtype=torch.bool, device=scores_for_selection.device)
    group_mask.scatter_(1, top_group_indices, True)

    # Expand mask to expert level
    expert_mask = group_mask.unsqueeze(-1).expand(-1, -1, experts_per_group).reshape(N, E)

    # Mask out non-selected group experts with -inf
    masked_scores = scores_for_selection.clone()
    masked_scores[~expert_mask] = float('-inf')

    # Top-k from remaining
    _, topk_indices = torch.topk(masked_scores, top_k, dim=-1)
    topk_weights = scores_for_weights.gather(1, topk_indices)

    return topk_weights, topk_indices


def _validate_inputs(hidden_states, gate_weight, w_up, w_down, top_k, activation, gating, ep_group=None):
    assert hidden_states.dim() == 2, f"hidden_states must be 2D [N, D], got {hidden_states.shape}"
    N, D = hidden_states.shape
    E = gate_weight.shape[0]
    E_w = w_up.shape[0]  # E_local when EP is active, E otherwise
    ffn_dim = w_up.shape[1]
    assert gate_weight.shape == (E, D), f"gate_weight must be [{E}, {D}], got {gate_weight.shape}"
    assert w_up.shape == (E_w, ffn_dim, D), f"w_up must be [{E_w}, {ffn_dim}, {D}], got {w_up.shape}"
    assert w_down.shape == (E_w, D, ffn_dim // 2), \
        f"w_down must be [{E_w}, {D}, {ffn_dim//2}], got {w_down.shape}"
    if ep_group is not None:
        ep_size = torch.distributed.get_world_size(ep_group)
        assert E % ep_size == 0, f"n_experts ({E}) must be divisible by ep_size ({ep_size})"
        assert E_w == E // ep_size, \
            f"With EP, w_up must have E/ep_size={E//ep_size} experts, got {E_w}"
    else:
        assert E_w == E, f"w_up must have {E} experts, got {E_w}"
    assert ffn_dim % 2 == 0, f"ffn_dim must be even, got {ffn_dim}"
    assert 1 <= top_k <= E, f"top_k must be in [1, {E}], got {top_k}"
    dev = hidden_states.device
    assert all(t.device == dev for t in [gate_weight, w_up, w_down]), "All tensors must be on same device"
    assert activation in ("swiglu", "geglu", "relu_squared"), f"Unknown activation: {activation}"
    assert gating in ("softmax", "sigmoid"), f"Unknown gating: {gating}"


def _validate_ep_inputs(hidden_states, gate_weight, w_up, w_down, top_k, activation, gating):
    """Compile-friendly EP validation — no torch.distributed calls (would cause graph break)."""
    assert hidden_states.dim() == 2, f"hidden_states must be 2D [N, D], got {hidden_states.shape}"
    E = gate_weight.shape[0]
    E_w = w_up.shape[0]
    ffn_dim = w_up.shape[1]
    assert w_down.shape[2] == ffn_dim // 2
    assert ffn_dim % 2 == 0
    assert 1 <= top_k <= E
    assert E_w < E, f"EP mode: w_up should have E_local < E experts, got {E_w} >= {E}"
    assert activation in ("swiglu", "geglu", "relu_squared")
    assert gating in ("softmax", "sigmoid")


@torch.compiler.allow_in_graph
class _FusedMoEAutograd(torch.autograd.Function):
    """Fused MoE with batched cuBLAS GEMM (torch.bmm) forward + backward.

    Sorts tokens by expert into [E, max_tokens, D] batched layout, then uses
    torch.bmm for all matmuls (6 total: 2 forward, 4 backward). This replaces
    the Triton grouped GEMM + cuBLAS expert loop with pure cuBLAS batched ops.

    Supports activation checkpointing: when recompute_activations=True, saves
    9 tensors instead of 12 by recomputing up_batched, activated_batched,
    down_batched in backward.
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden_states, gate_weight, w_up, w_down, top_k,
                activation, gating, capacity_factor, recompute_activations,
                max_tokens_per_expert, allow_dropped_tokens, return_metrics):
        N, D = hidden_states.shape
        E = gate_weight.shape[0]
        ffn_dim = w_up.shape[1]
        half_ffn = ffn_dim // 2
        device = hidden_states.device

        # Determine compute dtype: use autocast target if active, else input dtype
        dtype = _get_compute_dtype(hidden_states.dtype)

        # Cast to compute dtype only if needed (avoid redundant copy_ kernels)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        w_up_c = w_up if w_up.dtype == dtype else w_up.to(dtype)
        w_down_c = w_down if w_down.dtype == dtype else w_down.to(dtype)

        # Routing (uses _apply_routing for enhancement support)
        gate_weight_c = gate_weight if gate_weight.dtype == dtype else gate_weight.to(dtype)
        rcfg = _ROUTING_CFG
        if rcfg["input_jitter"] > 0.0 and hidden_states.requires_grad:
            jitter = rcfg["input_jitter"]
            noise = torch.empty_like(hidden_states).uniform_(1.0 - jitter, 1.0 + jitter)
            logits = (hidden_states * noise) @ gate_weight_c.T
        else:
            logits = hidden_states @ gate_weight_c.T

        topk_weights, topk_indices = _apply_routing(
            logits, gating, top_k,
            n_group=rcfg["n_group"], topk_group=rcfg["topk_group"],
            scoring_factor=rcfg["scoring_factor"], pre_softmax=rcfg["pre_softmax"],
            expert_bias=rcfg["expert_bias"],
        )

        # Routing: sort expanded tokens by expert
        flat_expert_ids = topk_indices.view(-1)  # [N * top_k]
        sorted_order = torch.argsort(flat_expert_ids, stable=True)
        tokens_per_expert = torch.bincount(flat_expert_ids, minlength=E)
        expert_starts = torch.zeros(E + 1, dtype=torch.long, device=device)
        expert_starts[1:] = torch.cumsum(tokens_per_expert, dim=0)

        # --- Feature 2: Expert capacity with token dropping ---
        if capacity_factor is not None:
            expert_capacity = int((N / E) * capacity_factor)
            expert_capacity = max(expert_capacity, top_k)  # minimum viable

            # Compute rank within expert using sorted_order
            # sorted_ranks[j] = position j in sorted order minus expert_start
            sorted_expert_ids = flat_expert_ids[sorted_order]
            sorted_ranks = torch.arange(len(sorted_order), device=device)
            sorted_ranks = sorted_ranks - expert_starts[sorted_expert_ids]

            # keep_mask_sorted[j] = True if sorted position j is within capacity
            keep_mask_sorted = sorted_ranks < expert_capacity

            # Map back to original expanded token order
            rank_within_expert = torch.zeros_like(flat_expert_ids)
            rank_within_expert[sorted_order] = sorted_ranks
            keep_mask = rank_within_expert < expert_capacity

            # Zero out dropped token weights and re-normalize remaining weights.
            # Tokens with ALL experts dropped get zero weights (no NaN from 0/0).
            keep_mask_2d = keep_mask.view(N, top_k).float()
            topk_weights = topk_weights * keep_mask_2d
            weight_sum = topk_weights.sum(dim=-1, keepdim=True)
            has_any = (weight_sum > 0).float()  # 1.0 if any expert kept, 0.0 if all dropped
            topk_weights = topk_weights / (weight_sum + (1.0 - has_any))  # avoid 0/0: denom is 1 when all dropped

            # Filter sorted_order to only include kept tokens
            sorted_order = sorted_order[keep_mask_sorted]
            # Recompute tokens_per_expert and expert_starts for kept tokens only
            kept_expert_ids = flat_expert_ids[sorted_order]
            tokens_per_expert = torch.bincount(kept_expert_ids, minlength=E)
            expert_starts = torch.zeros(E + 1, dtype=torch.long, device=device)
            expert_starts[1:] = torch.cumsum(tokens_per_expert, dim=0)

            # With capacity, max_tokens is exact — no overflow possible
            max_tokens = expert_capacity

            # Save keep_mask as float for backward gather-reduce
            # (dropped tokens must not contribute to hidden gradient)
            keep_mask_float = keep_mask.float().to(dtype).view(-1)  # [N*top_k]
        else:
            keep_mask_float = None
            # Use user-provided max_tokens or static estimate (no GPU sync)
            if max_tokens_per_expert is not None:
                max_tokens = max_tokens_per_expert
            else:
                max_tokens = _estimate_max_tokens(N, top_k, E)

            # Overflow check: detect if any expert received more tokens than buffer
            if not allow_dropped_tokens:
                actual_max = tokens_per_expert.max().item()
                if actual_max > max_tokens:
                    raise RuntimeError(
                        f"MoE routing overflow: expert received {actual_max} tokens but buffer "
                        f"is {max_tokens}. {actual_max - max_tokens} tokens would be silently "
                        f"dropped. Fix options:\n"
                        f"  1. Set max_tokens_per_expert={actual_max} (or higher) via calibration\n"
                        f"  2. Set capacity_factor to enable intentional token dropping\n"
                        f"  3. Set allow_dropped_tokens=True to suppress this error"
                    )

        flat_weights = topk_weights.to(dtype).view(-1)  # [N * top_k]

        # Number of tokens in sorted_order (may be < N*top_k if capacity dropped some)
        num_sorted = sorted_order.shape[0]

        # *** Fused routing scatter (replaces ~9 PyTorch ops) ***
        hidden_batched, expand_to_batch = fused_routing_scatter(
            sorted_order, flat_expert_ids, expert_starts,
            hidden_states, E, max_tokens, top_k,
            num_sorted=num_sorted if num_sorted != N * top_k else None,
        )

        # Megatron-LM pattern: dropped tokens must map to a safe dummy position,
        # not position 0 (which holds real data). In backward, scatter/gather kernels
        # read/write expand_to_batch positions — dropped tokens at position 0 would
        # overwrite expert 0's first slot. Redirect them to a zeroed sentinel row.
        if capacity_factor is not None:
            dummy_pos = E * max_tokens  # sentinel position (one past real data)
            expand_to_batch[~keep_mask] = dummy_pos

        hidden_batched = hidden_batched.view(E, max_tokens, D)

        # Up-projection: [E, max_tokens, D] @ [E, D, ffn_dim] = [E, max_tokens, ffn_dim]
        up_batched = torch.bmm(hidden_batched, w_up_c.transpose(1, 2))

        # Fused GLU activation
        activated_batched = activation_forward(up_batched, activation)  # [E, max_tokens, half_ffn]

        # Down-projection: [E, max_tokens, half_ffn] @ [E, half_ffn, D] = [E, max_tokens, D]
        down_batched = torch.bmm(activated_batched, w_down_c.transpose(1, 2))

        # *** Fused gather-reduce (replaces 5 PyTorch ops) ***
        # When capacity dropping is active, expand_to_batch has sentinel values at
        # E*max_tokens for dropped tokens. Pad batched data with a zero row so the
        # sentinel reads zero instead of out-of-bounds.
        down_flat = down_batched.view(E * max_tokens, D)
        if capacity_factor is not None:
            _zero = torch.zeros(1, D, dtype=down_flat.dtype, device=device)
            down_flat = torch.cat([down_flat, _zero], dim=0)
        output = fused_gather_reduce(
            down_flat, expand_to_batch, flat_weights, N, top_k
        )

        # Save for backward
        ctx.top_k = top_k
        ctx.max_tokens = max_tokens
        ctx.activation = activation
        ctx.gating = gating
        ctx.recompute_activations = recompute_activations
        ctx.has_capacity = capacity_factor is not None

        if recompute_activations:
            ctx.save_for_backward(
                hidden_states, hidden_batched, gate_weight_c, w_up_c, w_down_c,
                topk_indices, topk_weights,
                expand_to_batch, flat_weights,
                *([keep_mask_float] if keep_mask_float is not None else [])
            )
        else:
            ctx.save_for_backward(
                hidden_states, hidden_batched, gate_weight_c, w_up_c, w_down_c,
                topk_indices, topk_weights,
                up_batched, activated_batched, down_batched,
                expand_to_batch, flat_weights,
                *([keep_mask_float] if keep_mask_float is not None else [])
            )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """Backward with torch.bmm -- all matmuls are cuBLAS batched GEMM."""
        top_k = ctx.top_k
        max_tokens = ctx.max_tokens
        activation = ctx.activation
        gating = ctx.gating
        recompute_activations = ctx.recompute_activations

        has_capacity = ctx.has_capacity

        if recompute_activations:
            saved = ctx.saved_tensors
            (hidden_states, hidden_batched, gate_weight_c, w_up_c, w_down_c,
             topk_indices, topk_weights,
             expand_to_batch, flat_weights) = saved[:9]
            keep_mask_float = saved[9] if has_capacity else None

            # Recompute the 3 large tensors
            up_batched = torch.bmm(hidden_batched, w_up_c.transpose(1, 2))
            activated_batched = activation_forward(up_batched, activation)
            down_batched = torch.bmm(activated_batched, w_down_c.transpose(1, 2))
        else:
            saved = ctx.saved_tensors
            (hidden_states, hidden_batched, gate_weight_c, w_up_c, w_down_c,
             topk_indices, topk_weights,
             up_batched, activated_batched, down_batched,
             expand_to_batch, flat_weights) = saved[:12]
            keep_mask_float = saved[12] if has_capacity else None

        N, D = grad_output.shape
        E = gate_weight_c.shape[0]
        half_ffn = w_up_c.shape[1] // 2
        device = grad_output.device

        # Determine compute dtype (same as forward)
        dtype = _get_compute_dtype(grad_output.dtype)
        if grad_output.dtype != dtype:
            grad_output = grad_output.to(dtype)

        # When capacity dropping is active, expand_to_batch has sentinel values at
        # E*max_tokens for dropped tokens. Pad batched tensors with a zero row so
        # sentinel reads zero instead of out-of-bounds. (Megatron-LM pattern)
        _pad = has_capacity  # shorthand

        # *** Routing weight gradient (replaces gather+broadcast multiply+reduce) ***
        down_flat = down_batched.view(E * max_tokens, D)
        if _pad:
            down_flat = torch.cat([down_flat, torch.zeros(1, D, dtype=down_flat.dtype, device=device)])
        # Routing weight gradient: dot(down_batched[batch_pos], grad_output[i])
        # Uses vectorized PyTorch indexing (fast, no Triton autotune issues).
        # gathered_down: [N*top_k, D] = down_flat[expand_to_batch]
        gathered_down = down_flat[expand_to_batch]  # [N*top_k, D]
        grad_output_expanded = grad_output.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, D)  # [N*top_k, D]
        grad_topk_weights = (gathered_down.float() * grad_output_expanded.float()).sum(dim=-1).view(N, top_k)

        # *** Fused weighted scatter (replaces 5 ops) ***
        # When capacity dropping is active, allocate +1 row for the sentinel position.
        weighted_go_batched = fused_weighted_scatter(
            grad_output, expand_to_batch, flat_weights, E, max_tokens, top_k,
            output_rows=(E * max_tokens + 1) if _pad else None,
        )
        # Slice off sentinel row (if present) before reshaping for bmm
        weighted_go_batched = weighted_go_batched[:E * max_tokens].view(E, max_tokens, D)

        # --- Phase 1: Down-proj backward via bmm ---
        grad_activated_batched = torch.bmm(weighted_go_batched, w_down_c)

        # --- Phase 2: Fused GLU activation backward ---
        grad_up_input_batched = activation_backward(grad_activated_batched, up_batched, activation)

        # --- Phase 3: Up-proj backward via bmm ---
        grad_hidden_batched = torch.bmm(grad_up_input_batched, w_up_c)

        # *** Fused gather-reduce (replaces 4 ops) ***
        grad_hid_flat = grad_hidden_batched.view(E * max_tokens, D)
        if _pad:
            grad_hid_flat = torch.cat([grad_hid_flat, torch.zeros(1, D, dtype=grad_hid_flat.dtype, device=device)])
        grad_hidden = fused_gather_reduce(
            grad_hid_flat, expand_to_batch, keep_mask_float, N, top_k
        )

        # --- Phase 4: Weight gradients via bmm (no expert loop!) ---
        grad_w_down = torch.bmm(weighted_go_batched.transpose(1, 2), activated_batched)
        grad_w_up = torch.bmm(grad_up_input_batched.transpose(1, 2), hidden_batched)

        # --- Phase 5: Routing gradient (gating-dependent) ---
        # When capacity dropping zeroes all weights for a token, mask grad to prevent NaN.
        if has_capacity:
            token_active = (topk_weights.sum(-1, keepdim=True) > 0).float()  # [N, 1]
            grad_topk_weights = grad_topk_weights * token_active

        if gating == "softmax":
            s = topk_weights
            sg = (grad_topk_weights * s).sum(-1, keepdim=True)
            grad_tl = s * (grad_topk_weights - sg)
        elif gating == "sigmoid":
            logits = hidden_states @ gate_weight_c.T
            scores = torch.sigmoid(logits.float())
            topk_scores = scores.gather(1, topk_indices)
            topk_scores_sum = topk_scores.sum(-1, keepdim=True)
            g_dot_w = (grad_topk_weights * topk_weights).sum(-1, keepdim=True)
            grad_topk_scores = (grad_topk_weights - g_dot_w) / (topk_scores_sum + _WEIGHT_EPS)
            grad_tl = grad_topk_scores * topk_scores * (1.0 - topk_scores)

        # Common scatter + gate gradient
        grad_logits = torch.zeros(N, E, dtype=torch.float32, device=device)
        grad_logits.scatter_add_(1, topk_indices, grad_tl)
        gl = grad_logits.to(dtype)
        grad_hidden = grad_hidden + gl @ gate_weight_c
        grad_gate_weight = gl.T @ hidden_states

        # Return None for: top_k, activation, gating, capacity_factor, recompute_activations,
        # max_tokens_per_expert, allow_dropped_tokens, return_metrics
        return grad_hidden, grad_gate_weight, grad_w_up, grad_w_down, None, None, None, None, None, None, None, None


class _FusedMoEEPAutograd(torch.autograd.Function):
    """Expert-parallel fused MoE: routing → dispatch → bmm expert compute → combine.

    Routing (gate matmul + fused topk) runs INSIDE this function. Backward
    hand-writes routing gradient in ~10 ops instead of hundreds of autograd-traced
    kernels. Called via @torch.compiler.disable wrapper (_fused_moe_ep).

    Note: splitting routing outside for torch.compile was investigated but adds
    ~10ms overhead from extra graph transitions (23 layers × 2 breaks). The
    @allow_in_graph approach crashes due to FakeTensorMode mismatch from
    distributed collectives. See dev diary Step 4c for full investigation.

    Key optimizations (Megatron-inspired):
    - Fused routing + dispatch + expert compute + combine in single autograd
    - Fused topk Triton kernel for routing (2.2x faster than PyTorch ops)
    - Fused combine + weight application via fused_gather_reduce
    - Fused dispatch backward via fused_weighted_scatter
    - Async A2A overlap with routing grad + weight gradients in backward
    - Shared expert overlap on side CUDA stream
    - Minimal dtype casts: FP32 only for softmax/sigmoid grad, BF16 everywhere else
    """

    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden_states, gate_weight, w_up, w_down,
                top_k, activation, gating, max_tokens_per_expert,
                shared_w_gate, shared_w_up, shared_w_down,
                latent_w_down, latent_w_up):
        ep_group = _EP_GROUP
        N, D = hidden_states.shape
        E_local = w_up.shape[0]
        ffn_dim = w_up.shape[1]
        device = hidden_states.device
        ep_size = torch.distributed.get_world_size(ep_group)
        E = ep_size * E_local  # total experts (no GPU sync needed)

        has_shared = shared_w_gate is not None
        has_latent = latent_w_down is not None

        # Cast to compute dtype only if needed (avoid redundant copy_ kernels)
        dtype = _get_compute_dtype(hidden_states.dtype)
        if hidden_states.dtype != dtype:
            hidden_states = hidden_states.to(dtype)
        w_up_c = w_up if w_up.dtype == dtype else w_up.to(dtype)
        w_down_c = w_down if w_down.dtype == dtype else w_down.to(dtype)
        gate_weight_c = gate_weight if gate_weight.dtype == dtype else gate_weight.to(dtype)

        # Routing: gate matmul + top-k + gating (inside autograd for fast backward)
        rcfg = _ROUTING_CFG
        if rcfg["input_jitter"] > 0.0 and hidden_states.requires_grad:
            jitter = rcfg["input_jitter"]
            noise = torch.empty_like(hidden_states).uniform_(1.0 - jitter, 1.0 + jitter)
            logits = (hidden_states * noise) @ gate_weight_c.T
        else:
            logits = hidden_states @ gate_weight_c.T

        topk_weights, topk_indices = _apply_routing(
            logits, gating, top_k,
            n_group=rcfg["n_group"], topk_group=rcfg["topk_group"],
            scoring_factor=rcfg["scoring_factor"], pre_softmax=rcfg["pre_softmax"],
            expert_bias=rcfg["expert_bias"],
        )

        # Optional latent projection before dispatch (reduces A2A volume)
        if has_latent:
            lw_down_c = latent_w_down if latent_w_down.dtype == dtype else latent_w_down.to(dtype)
            lw_up_c = latent_w_up if latent_w_up.dtype == dtype else latent_w_up.to(dtype)
            # Project: [N, D] → [N, latent_dim]
            dispatch_tokens = hidden_states @ lw_down_c.T
        else:
            dispatch_tokens = hidden_states

        # Non-differentiable dispatch (fused: 1 all-to-all instead of 3)
        # Use cached dispatcher to avoid object creation overhead per call
        dispatcher = _get_dispatcher(E, ep_size, ep_group, backend=_EP_BACKEND)
        received_tokens, handle = dispatcher.dispatch_fused(
            dispatch_tokens, topk_indices,
        )

        # Launch shared expert on side stream (overlaps with dispatch A2A + expert compute)
        if has_shared:
            shared_stream = _get_shared_stream()
            ready_event = torch.cuda.Event()
            ready_event.record()
            with torch.cuda.stream(shared_stream):
                shared_stream.wait_event(ready_event)
                sw_gate = shared_w_gate if shared_w_gate.dtype == dtype else shared_w_gate.to(dtype)
                sw_up = shared_w_up if shared_w_up.dtype == dtype else shared_w_up.to(dtype)
                sw_down = shared_w_down if shared_w_down.dtype == dtype else shared_w_down.to(dtype)
                gate_out = hidden_states @ sw_gate.T
                up_out = hidden_states @ sw_up.T
                shared_out = (F.silu(gate_out) * up_out) @ sw_down.T
            done_event = torch.cuda.Event()
            done_event.record(shared_stream)

        # === Local expert compute via bmm ===
        local_expert_ids = handle.local_expert_ids.long()
        M = received_tokens.shape[0]

        # Un-project received tokens if latent projection active
        if has_latent:
            # received_tokens: [M, latent_dim] → [M, D]
            received_latent = received_tokens  # save for backward weight grad
            received_tokens = received_latent @ lw_up_c.T

        # Sort received tokens by local expert and build batched layout
        sorted_order_local = torch.argsort(local_expert_ids, stable=True)
        tokens_per_expert_local = torch.bincount(local_expert_ids, minlength=E_local)
        expert_starts_local = torch.zeros(E_local + 1, dtype=torch.long, device=device)
        expert_starts_local[1:] = torch.cumsum(tokens_per_expert_local, dim=0)

        # Max tokens per local expert (static estimate or from routing)
        if max_tokens_per_expert is not None:
            max_tokens_local = max_tokens_per_expert
        else:
            max_tokens_local = _estimate_max_tokens(M, 1, E_local)

        # Fused routing scatter: [M, D] → [E_local, max_tokens_local, D]
        hidden_batched, expand_to_batch = fused_routing_scatter(
            sorted_order_local, local_expert_ids, expert_starts_local,
            received_tokens, E_local, max_tokens_local, 1,
        )
        hidden_batched = hidden_batched.view(E_local, max_tokens_local, D)

        # Up-projection: [E_local, max_tokens_local, D] @ [E_local, D, ffn_dim]
        up_batched = torch.bmm(hidden_batched, w_up_c.transpose(1, 2))

        # Fused GLU activation
        activated_batched = activation_forward(up_batched, activation)

        # Down-projection: [E_local, max_tokens_local, half_ffn] @ [E_local, half_ffn, D]
        down_batched = torch.bmm(activated_batched, w_down_c.transpose(1, 2))

        # Fused gather-reduce: [E_local * max_tokens_local, D] → [M, D] (unweighted)
        down_flat = down_batched.view(E_local * max_tokens_local, D)
        local_output = fused_gather_reduce(
            down_flat, expand_to_batch, None, M, 1,
        )

        # Re-project to latent before combine (reduces return A2A volume)
        if has_latent:
            # local_output: [M, D] → [M, latent_dim]
            local_output = local_output @ lw_down_c.T

        # Fused combine: reverse A2A + unsort → [N*top_k, combine_dim]
        unsorted = dispatcher.fused_combine(local_output, handle)

        # Un-project after combine if latent projection active
        if has_latent:
            # unsorted: [N*top_k, latent_dim] → [N*top_k, D]
            unsorted_latent = unsorted  # save for backward weight grad
            unsorted = unsorted_latent @ lw_up_c.T

        # Fused weight application + reduce: apply topk_weights during gather-reduce
        # (Megatron pattern: fuse probs into unpermute)
        identity_map = torch.arange(N * top_k, dtype=torch.long, device=device)
        topk_weights_flat = topk_weights.to(dtype).view(-1)
        output = fused_gather_reduce(
            unsorted, identity_map, topk_weights_flat, N, top_k,
        )

        # Sync shared expert and add output
        if has_shared:
            torch.cuda.current_stream().wait_event(done_event)
            output = output + shared_out

        # Save for backward
        ctx.top_k = top_k
        ctx.max_tokens_local = max_tokens_local
        ctx.activation = activation
        ctx.gating = gating
        ctx.ep_size = ep_size
        ctx.has_shared = has_shared
        ctx.has_latent = has_latent
        ctx.M = M
        ctx.send_splits_dispatch = handle.send_splits
        ctx.recv_splits_dispatch = handle.recv_splits
        ctx.ep_group = ep_group
        shared_tensors = [sw_gate, sw_up, sw_down] if has_shared else []
        latent_tensors = [lw_down_c, lw_up_c, received_latent, unsorted_latent] if has_latent else []
        ctx.save_for_backward(
            hidden_states, gate_weight_c, w_up_c, w_down_c,
            topk_indices, topk_weights,
            hidden_batched, up_batched, activated_batched,
            expand_to_batch,
            handle.sort_order, unsorted,
            *shared_tensors, *latent_tensors,
        )
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        """Backward with fused routing gradient + 1 A2A for weight grads.

        Two logically separable phases (for future Pipeline Parallelism):

        **Phase dX** (Steps 1-3, 6): Activation gradient
          - Routing weight gradient from expert outputs
          - Launch A2A on side stream (non-blocking)
          - Hand-written routing backward (gate matmul grad)
          - Shared expert hidden grad (on side stream, overlapped)
          → Returns: grad_hidden, grad_gate_weight

        **Phase dW** (Steps 4-5): Weight gradient
          - Sync A2A (blocks until combine-reverse completes)
          - Expert weight grads via bmm (w_up, w_down)
          - Shared expert weight grads (sync side stream)
          - Latent projection weight grads
          → Returns: grad_w_up, grad_w_down, grad_shared_*, grad_latent_*

        In Pipeline Parallelism, a scheduler can overlap Phase dW of stage i
        with Phase dX of stage i-1 (Megatron backward_dw pattern). Currently
        both phases run sequentially in this single backward() call.

        A2A overlap: combine-reverse A2A runs on side stream while routing
        gradient computes on default stream. Sync before weight grad bmm.
        """
        top_k = ctx.top_k
        max_tokens_local = ctx.max_tokens_local
        activation = ctx.activation
        gating = ctx.gating
        ep_size = ctx.ep_size
        ep_group = ctx.ep_group
        has_shared = ctx.has_shared
        has_latent = ctx.has_latent
        M = ctx.M
        send_splits_dispatch = ctx.send_splits_dispatch
        recv_splits_dispatch = ctx.recv_splits_dispatch

        saved = ctx.saved_tensors
        (hidden_states, gate_weight_c, w_up_c, w_down_c,
         topk_indices, topk_weights,
         hidden_batched, up_batched, activated_batched,
         expand_to_batch,
         sort_order_dispatch, unsorted_fwd) = saved[:12]
        idx = 12
        if has_shared:
            sw_gate, sw_up, sw_down = saved[idx], saved[idx + 1], saved[idx + 2]
            idx += 3
        if has_latent:
            lw_down_c, lw_up_c, received_latent, unsorted_latent = (
                saved[idx], saved[idx + 1], saved[idx + 2], saved[idx + 3])
            latent_dim = lw_down_c.shape[0]

        N, D = grad_output.shape
        E_local = w_up_c.shape[0]
        E = gate_weight_c.shape[0]
        device = grad_output.device

        dtype = _get_compute_dtype(grad_output.dtype)
        if grad_output.dtype != dtype:
            grad_output = grad_output.to(dtype)

        # === Shared expert backward on side stream ===
        if has_shared:
            shared_stream = _get_shared_stream()
            ready_event = torch.cuda.Event()
            ready_event.record()
            with torch.cuda.stream(shared_stream):
                shared_stream.wait_event(ready_event)
                gate_out = hidden_states @ sw_gate.T
                up_out = hidden_states @ sw_up.T
                silu_gate = F.silu(gate_out)

                grad_pre_down = grad_output @ sw_down
                grad_silu_gate = grad_pre_down * up_out
                grad_up_out = grad_pre_down * silu_gate

                sig_gate = torch.sigmoid(gate_out)
                grad_gate_out = grad_silu_gate * (silu_gate + sig_gate * (1.0 - silu_gate))

                grad_shared_w_gate = grad_gate_out.T @ hidden_states
                grad_shared_w_up = grad_up_out.T @ hidden_states
                grad_shared_w_down = grad_output.T @ (silu_gate * up_out)
                grad_hidden_shared = grad_gate_out @ sw_gate + grad_up_out @ sw_up
            shared_done_event = torch.cuda.Event()
            shared_done_event.record(shared_stream)

        # ========== Phase dX: Activation gradients (PP-separable) ==========
        # In PP, everything here can run while previous stage processes dW.

        # === Step 1: Routing weight gradient (no A2A needed) ===
        # unsorted_fwd is in D-space (after un-projection in forward if latent active)
        per_expert_out = unsorted_fwd.view(N, top_k, D)
        grad_topk_weights = (grad_output.unsqueeze(1) * per_expert_out).sum(-1)  # [N, top_k]

        # === Step 2: A2A on side stream + routing grad on default (overlapped) ===
        # Forward combine: local_output → A2A(recv→send) → unsort[sort_order]
        # Backward: grad_unsorted → sort[sort_order] → reverse A2A → grad_local_output
        grad_unsorted = (grad_output.unsqueeze(1) * topk_weights.to(dtype).unsqueeze(-1))
        grad_flat = grad_unsorted.reshape(N * top_k, D)

        # If latent active: project grad back to latent before A2A (mirror forward combine project)
        if has_latent:
            # Forward: unsorted_D = unsorted_latent @ lw_up.T
            # Backward: grad_unsorted_latent = grad_unsorted_D @ lw_up
            grad_flat_latent = grad_flat @ lw_up_c  # [N*top_k, D] @ [D, latent] = [N*top_k, latent]
            sorted_grad = grad_flat_latent[sort_order_dispatch]
        else:
            sorted_grad = grad_flat[sort_order_dispatch]

        # Launch A2A on side stream (overlaps with routing grad on default stream)
        a2a_stream = _get_a2a_stream()
        a2a_ready = torch.cuda.Event()
        a2a_ready.record()
        with torch.cuda.stream(a2a_stream):
            a2a_stream.wait_event(a2a_ready)
            recv_grad = _get_fc().all_to_all_single(
                sorted_grad.contiguous(), recv_splits_dispatch, send_splits_dispatch, ep_group
            ).clone()
        a2a_done = torch.cuda.Event()
        a2a_done.record(a2a_stream)

        # === Step 3: Routing gradient on default stream (overlapped with A2A) ===
        # Hand-written routing backward — matches non-EP path (lines 558-576).
        # ~10 ops instead of hundreds from PyTorch autograd.
        if gating == "softmax":
            s = topk_weights  # [N, top_k] float32 from _apply_routing
            sg = (grad_topk_weights * s).sum(-1, keepdim=True)
            grad_tl = s * (grad_topk_weights - sg)
        elif gating == "sigmoid":
            logits = hidden_states @ gate_weight_c.T
            scores = torch.sigmoid(logits.float())
            topk_scores = scores.gather(1, topk_indices)
            topk_scores_sum = topk_scores.sum(-1, keepdim=True)
            g_dot_w = (grad_topk_weights * topk_weights).sum(-1, keepdim=True)
            grad_topk_scores = (grad_topk_weights - g_dot_w) / (topk_scores_sum + _WEIGHT_EPS)
            grad_tl = grad_topk_scores * topk_scores * (1.0 - topk_scores)

        grad_logits = torch.zeros(N, E, dtype=torch.float32, device=device)
        grad_logits.scatter_add_(1, topk_indices, grad_tl)
        gl = grad_logits.to(dtype)
        grad_hidden_routing = gl @ gate_weight_c
        grad_gate_weight = gl.T @ hidden_states

        # (Latent weight grads computed in Step 5 below, after A2A sync)

        # ========== Phase dW: Weight gradients (PP-separable) ==========
        # In PP, everything below here can overlap with previous stage's dX.

        # === Step 4: Sync A2A, then local expert weight gradients via bmm ===
        torch.cuda.current_stream().wait_event(a2a_done)

        # recv_grad is in latent space if latent active; un-project to D for weight grads
        if has_latent:
            recv_grad_latent = recv_grad  # save for lw_down weight grad
            # Forward: recv_D = recv_latent @ lw_up.T
            # Backward: grad_recv_latent → un-project → grad_recv_D
            recv_grad = recv_grad_latent @ lw_up_c.T  # [M, latent] @ [latent, D] = [M, D]

        # Scatter received gradient into batched layout
        weighted_go_batched = torch.zeros(
            E_local * max_tokens_local, D, dtype=dtype, device=device,
        )
        weighted_go_batched[expand_to_batch] = recv_grad
        weighted_go_batched = weighted_go_batched.view(E_local, max_tokens_local, D)

        # Down-proj backward (for activation grad needed by up-proj weight grad)
        grad_activated_batched = torch.bmm(weighted_go_batched, w_down_c)
        grad_up_input_batched = activation_backward(grad_activated_batched, up_batched, activation)

        # Weight gradients via bmm (scaled by 1/ep_size for gradient averaging)
        grad_w_down = torch.bmm(
            weighted_go_batched.transpose(1, 2), activated_batched
        ) * (1.0 / ep_size)
        grad_w_up = torch.bmm(
            grad_up_input_batched.transpose(1, 2), hidden_batched
        ) * (1.0 / ep_size)

        # === Step 5: Latent projection weight gradients ===
        if has_latent:
            # Two differentiable matmul sites contribute to latent weight grads:
            #
            # (1) Combine un-project: unsorted_D = unsorted_latent @ lw_up.T
            #     grad_lw_up += grad_flat.T @ unsorted_latent
            #     (unsorted_latent saved in forward)
            #
            # (2) Recv un-project: recv_D = recv_latent @ lw_up.T
            #     grad_lw_up += recv_grad_D.T @ received_latent
            #     (received_latent saved in forward, recv_grad_D is recv_grad after un-project)
            #
            # For lw_down, the differentiable sites are:
            # (3) Combine project: local_out_latent = local_out_D @ lw_down.T
            #     grad_lw_down += recv_grad_latent.T @ local_out_D
            #     (recv_grad_latent saved above, local_out_D needs recompute from batched output)
            #
            # (4) Dispatch project: dispatch_latent = hidden @ lw_down.T
            #     No backward contribution (non-differentiable dispatch).
            #
            # lw_up grad from both un-project sites:
            grad_lw_up = (grad_flat.T @ unsorted_latent +
                          recv_grad.T @ received_latent) * (1.0 / ep_size)

            # lw_down grad from combine project site:
            # Recompute local_out_D from saved batched expert tensors
            down_batched_recomp = torch.bmm(activated_batched, w_down_c.transpose(1, 2))
            local_output_D = fused_gather_reduce(
                down_batched_recomp.view(E_local * max_tokens_local, D),
                expand_to_batch, None, M, 1,
            )
            grad_lw_down = (recv_grad_latent.T @ local_output_D) * (1.0 / ep_size)

        # === Step 6: Combine routing + shared expert hidden grads ===
        grad_hidden = grad_hidden_routing
        if has_shared:
            torch.cuda.current_stream().wait_event(shared_done_event)
            grad_hidden = grad_hidden + grad_hidden_shared
        else:
            grad_shared_w_gate = None
            grad_shared_w_up = None
            grad_shared_w_down = None

        if not has_latent:
            grad_lw_down = None
            grad_lw_up = None

        # Returns: grad for (hidden_states, gate_weight, w_up, w_down,
        #          top_k, activation, gating, max_tokens_per_expert,
        #          shared_w_gate, shared_w_up, shared_w_down,
        #          latent_w_down, latent_w_up)
        return (grad_hidden, grad_gate_weight, grad_w_up, grad_w_down,
                None, None, None, None,
                grad_shared_w_gate, grad_shared_w_up, grad_shared_w_down,
                grad_lw_down, grad_lw_up)


@torch.compiler.disable
def set_ep_group(group, backend="nccl"):
    """Store EP process group in module global for compile-compatible EP.

    Must be called before fused_moe() in compiled code. ProcessGroup can't be
    a dynamo graph node, so this compiler-disabled function stashes it in a global
    that the EP autograd function reads directly.

    Usage in model forward():
        set_ep_group(self.ep_group)          # compiler-disabled, causes graph break once
        output, losses = fused_moe(...)      # no ep_group arg needed, reads from global
    """
    global _EP_GROUP, _EP_ACTIVE, _EP_BACKEND
    _EP_GROUP = group
    _EP_ACTIVE = group is not None
    _EP_BACKEND = backend if group is not None else "nccl"


@torch.compiler.disable
def _fused_moe_ep(hidden_states, gate_weight, w_up, w_down,
                  top_k, activation, gating, max_tokens_per_expert,
                  shared_expert_weights=None, latent_weights=None):
    """Expert-parallel fused MoE: routing → dispatch → expert compute → combine.

    Routing forward + backward both run inside the autograd function.
    Forward: gate matmul + fused topk. Backward: recomputed routing + hand-written grad.

    @torch.compiler.disable because A2A collectives can't be traced by dynamo.
    The routing ops (gate matmul + Triton topk) run inside the disabled region
    since splitting them out adds graph break overhead that outweighs the benefit.
    """
    if shared_expert_weights is not None:
        sw_gate, sw_up, sw_down = shared_expert_weights
    else:
        sw_gate = sw_up = sw_down = None
    if latent_weights is not None:
        lw_down, lw_up = latent_weights
    else:
        lw_down = lw_up = None
    return _FusedMoEEPAutograd.apply(
        hidden_states, gate_weight,
        w_up, w_down,
        top_k, activation, gating, max_tokens_per_expert,
        sw_gate, sw_up, sw_down,
        lw_down, lw_up,
    )


def fused_moe(
    hidden_states: torch.Tensor,  # [N, D]
    gate_weight: torch.Tensor,    # [E, D]
    w_up: torch.Tensor,           # [E, ffn_dim, D]
    w_down: torch.Tensor,         # [E, D, ffn_dim//2]
    top_k: int = 2,
    activation: str = "swiglu",
    gating: str = "softmax",
    capacity_factor: float | None = None,
    aux_loss_coeff: float = 0.0,
    z_loss_coeff: float = 0.0,
    recompute_activations: bool = False,
    ep_group=None,
    max_tokens_per_expert: int | None = None,
    allow_dropped_tokens: bool = False,
    return_metrics: bool = False,
    shared_expert_weights: tuple | None = None,
    latent_weights: tuple | None = None,
    # Routing enhancements (DeepSeek-V2/V3 style)
    n_group: int | None = None,
    topk_group: int | None = None,
    scoring_factor: float | None = None,
    pre_softmax: bool = False,
    input_jitter: float = 0.0,
    expert_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Fused MoE: gate -> top-k route -> GLU experts -> weighted combine.

    Uses fused Triton scatter/gather kernels + cuBLAS batched GEMM.
    Supports torch.amp.autocast and torch.compile.

    Args:
        hidden_states: [N, D] input tokens
        gate_weight: [E, D] gating projection weight
        w_up: [E, ffn_dim, D] up-projection (gate+up stacked)
        w_down: [E, D, ffn_dim//2] down-projection
        top_k: number of experts per token
        activation: "swiglu", "geglu", or "relu_squared"
        gating: "softmax" or "sigmoid"
        capacity_factor: None = no dropping, float = capacity multiplier (e.g. 1.25)
        aux_loss_coeff: Switch Transformer style load-balance loss coefficient (0 = disabled)
        z_loss_coeff: Router z-loss coefficient (0 = disabled)
        recompute_activations: True = recompute 3 large tensors in backward (~35% memory savings)
        ep_group: torch.distributed ProcessGroup for expert parallelism (None = single GPU)
        max_tokens_per_expert: Buffer size per expert. If None, uses a heuristic (150% margin).
            For best performance, set to the observed maximum from a calibration run.
            The kernel safely skips overflow tokens if the buffer is exceeded.
        allow_dropped_tokens: If False (default), raises RuntimeError when routing overflow
            would silently drop tokens. If True, allows silent dropping (fast path, no sync).
        return_metrics: If True, adds routing metrics to the losses dict (tokens_per_expert,
            max/min_expert_load, expert_load_imbalance, dropped_token_count/fraction).
        shared_expert_weights: (w_gate, w_up, w_down) for shared expert (EP path only).
        latent_weights: (proj_down_weight, proj_up_weight) for A2A latent dim compression (EP path only).
            proj_down: [latent_dim, D], proj_up: [D, latent_dim]. Projects tokens to smaller dim
            before dispatch and back after combine, reducing A2A communication volume.
        n_group: Number of expert groups for group-limited routing (DeepSeek-V2/V3).
        topk_group: Number of groups to select per token.
        scoring_factor: Scale routing weights by scoring_factor/top_k after selection.
        pre_softmax: Apply softmax before top-k selection (DeepSeek-V3 style).
        input_jitter: Multiply inputs by uniform(1-jitter, 1+jitter) during training.
        expert_bias: [E] bias added to scores before top-k (aux-loss-free routing).

    Returns:
        output: [N, D] MoE output
        losses: dict with optional "aux_loss", "z_loss", and routing metrics
    """
    # Validate inputs. When EP is active via module global (_EP_ACTIVE), ep_group
    # is not passed as an arg (to avoid ProcessGroup in compiled graph).
    # Validation for EP runs inside _fused_moe_ep (compiler-disabled).
    if not _EP_ACTIVE:
        _validate_inputs(hidden_states, gate_weight, w_up, w_down, top_k, activation, gating,
                         ep_group=ep_group)

    N, D = hidden_states.shape
    E = gate_weight.shape[0]
    device = hidden_states.device
    losses = {}

    # Set routing config for autograd functions to read
    _ROUTING_CFG["n_group"] = n_group
    _ROUTING_CFG["topk_group"] = topk_group
    _ROUTING_CFG["scoring_factor"] = scoring_factor
    _ROUTING_CFG["pre_softmax"] = pre_softmax
    _ROUTING_CFG["input_jitter"] = input_jitter
    _ROUTING_CFG["expert_bias"] = expert_bias

    # --- Routing info: needed for aux_loss and/or return_metrics ---
    need_routing_info = (aux_loss_coeff > 0.0 or z_loss_coeff > 0.0 or return_metrics)

    if need_routing_info:
        # Determine dtype for logits computation
        compute_dtype = _get_compute_dtype(hidden_states.dtype)
        logits = hidden_states.to(compute_dtype) @ gate_weight.to(compute_dtype).T  # [N, E]

        # Use _apply_routing for consistent routing with enhancements
        topk_weights_aux, topk_indices_aux = _apply_routing(
            logits, gating, top_k,
            n_group=n_group, topk_group=topk_group,
            scoring_factor=scoring_factor, pre_softmax=pre_softmax,
            expert_bias=expert_bias,
        )

        flat_expert_ids_aux = topk_indices_aux.view(-1)
        tokens_per_expert_aux = torch.bincount(flat_expert_ids_aux, minlength=E).float()

        if aux_loss_coeff > 0.0:
            # f_i = fraction of tokens routed to expert i
            f = tokens_per_expert_aux / N
            # P_i = mean router probability for expert i
            if gating == "softmax":
                P = torch.zeros(E, device=device, dtype=torch.float32)
                P.scatter_add_(0, flat_expert_ids_aux, topk_weights_aux.view(-1).float())
                P = P / N
            elif gating == "sigmoid":
                scores_full = torch.sigmoid(logits.float())
                P = scores_full.mean(dim=0)
            aux_loss = aux_loss_coeff * E * (f * P).sum()
            losses["aux_loss"] = aux_loss

        if z_loss_coeff > 0.0:
            z_loss = z_loss_coeff * torch.logsumexp(logits.float(), dim=-1).square().mean()
            losses["z_loss"] = z_loss

        if return_metrics:
            losses["tokens_per_expert"] = tokens_per_expert_aux.int()
            losses["max_expert_load"] = tokens_per_expert_aux.max().int()
            losses["min_expert_load"] = tokens_per_expert_aux.min().int()
            losses["expert_load_imbalance"] = (
                tokens_per_expert_aux.max().float() /
                (tokens_per_expert_aux.float().mean() + _WEIGHT_EPS)
            )
            if capacity_factor is not None:
                expert_capacity_m = max(int((N / E) * capacity_factor), top_k)
                # Compute rank within expert to count dropped tokens
                sorted_order_m = torch.argsort(flat_expert_ids_aux, stable=True)
                expert_starts_m = torch.zeros(E + 1, dtype=torch.long, device=device)
                expert_starts_m[1:] = torch.cumsum(tokens_per_expert_aux.long(), dim=0)
                sorted_expert_ids_m = flat_expert_ids_aux[sorted_order_m]
                sorted_ranks_m = torch.arange(len(sorted_order_m), device=device)
                sorted_ranks_m = sorted_ranks_m - expert_starts_m[sorted_expert_ids_m]
                rank_within_expert_m = torch.zeros_like(flat_expert_ids_aux)
                rank_within_expert_m[sorted_order_m] = sorted_ranks_m
                dropped_count = (rank_within_expert_m >= expert_capacity_m).sum()
                losses["dropped_token_count"] = dropped_count
                losses["dropped_token_fraction"] = dropped_count.float() / (N * top_k)

    if _EP_ACTIVE or ep_group is not None:
        # EP path: routing + dispatch + expert compute + combine all inside
        # _FusedMoEEPAutograd. Routing backward is hand-written (~10 ops)
        # instead of traced through PyTorch autograd (hundreds of small kernels).
        did_set_ep = False
        if ep_group is not None and not _EP_ACTIVE:
            set_ep_group(ep_group)
            did_set_ep = True

        # Pass gate_weight directly — routing runs inside the autograd function
        output = _fused_moe_ep(
            hidden_states, gate_weight, w_up, w_down,
            top_k, activation, gating, max_tokens_per_expert,
            shared_expert_weights=shared_expert_weights,
            latent_weights=latent_weights,
        )

        if did_set_ep:
            set_ep_group(None)

        return output, losses

    output = _FusedMoEAutograd.apply(
        hidden_states, gate_weight, w_up, w_down, top_k, activation, gating,
        capacity_factor, recompute_activations, max_tokens_per_expert,
        allow_dropped_tokens, return_metrics
    )
    return output, losses
