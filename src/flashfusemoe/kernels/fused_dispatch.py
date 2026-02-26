"""Fused dispatch kernel for Expert Parallelism.

Fuses the pack step of EP dispatch: given sort_order and hidden_states,
produces packed_send [N*top_k, D+1] in a single kernel, eliminating:
1. hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D) — expand
2. sorted_hidden = hidden_expanded[sort_order] — index
3. packed_send[:, :D] = sorted_hidden; packed_send[:, D] = sorted_expert_ids — pack

Instead: for each output position i, look up sort_order[i] to find the source
token-expert pair, compute token_id = sort_order[i] // top_k, copy hidden_states[token_id]
to output[i, :D], and pack expert_id into output[i, D].
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_dispatch_pack_kernel(
    # Inputs
    hidden_ptr,          # [N, D]
    sort_order_ptr,      # [N * top_k]
    flat_indices_ptr,    # [N * top_k] expert indices
    # Output
    packed_ptr,          # [N * top_k, D + 1]
    # Dimensions
    N_times_topk,
    D: tl.constexpr,
    top_k: tl.constexpr,
    stride_h_n, stride_h_d,
    stride_p_m, stride_p_d,
    BLOCK_M: tl.constexpr,
):
    """Pack hidden_states + expert_ids into sorted buffer for A2A.

    For output row i:
      source = sort_order[i]
      token_id = source // top_k
      packed[i, :D] = hidden[token_id, :]
      packed[i, D] = flat_indices[source]  (as float for dtype compatibility)
    """
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows < N_times_topk

    # Load sort_order for this block
    source = tl.load(sort_order_ptr + rows, mask=row_mask, other=0)
    token_ids = source // top_k

    # Load expert IDs and store in last column
    expert_ids = tl.load(flat_indices_ptr + source, mask=row_mask, other=0)

    # Copy hidden_states rows and pack expert_ids
    for d in range(D):
        vals = tl.load(hidden_ptr + token_ids * stride_h_n + d * stride_h_d,
                       mask=row_mask, other=0.0)
        tl.store(packed_ptr + rows * stride_p_m + d * stride_p_d, vals, mask=row_mask)

    # Store expert_id in column D
    tl.store(packed_ptr + rows * stride_p_m + D * stride_p_d,
             expert_ids.to(tl.float16) if hidden_ptr.dtype.element_ty == tl.float16
             else expert_ids.to(tl.bfloat16) if hidden_ptr.dtype.element_ty == tl.bfloat16
             else expert_ids.to(tl.float32),
             mask=row_mask)


def fused_dispatch_pack(
    hidden_states: torch.Tensor,   # [N, D]
    sort_order: torch.Tensor,      # [N * top_k]
    flat_indices: torch.Tensor,    # [N * top_k] expert assignments
    top_k: int,
) -> torch.Tensor:
    """Fused pack: sort + expand + index + pack into [N*top_k, D+1] buffer.

    Replaces:
        sorted_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D)
        sorted_hidden = sorted_hidden[sort_order]
        sorted_expert_ids = flat_indices[sort_order]
        packed_send = torch.empty(N * top_k, D + 1, ...)
        packed_send[:, :D] = sorted_hidden
        packed_send[:, D] = sorted_expert_ids.to(dtype)

    Args:
        hidden_states: [N, D] input tokens
        sort_order: [N * top_k] permutation indices sorting by destination GPU
        flat_indices: [N * top_k] flattened expert indices
        top_k: number of experts per token

    Returns:
        packed: [N * top_k, D + 1] buffer ready for all-to-all
    """
    N, D = hidden_states.shape
    M = N * top_k
    packed = torch.empty(M, D + 1, dtype=hidden_states.dtype, device=hidden_states.device)

    BLOCK_M = 128
    grid = (triton.cdiv(M, BLOCK_M),)

    _fused_dispatch_pack_kernel[grid](
        hidden_states, sort_order, flat_indices,
        packed,
        M, D, top_k,
        hidden_states.stride(0), hidden_states.stride(1),
        packed.stride(0), packed.stride(1),
        BLOCK_M=BLOCK_M,
    )

    return packed
