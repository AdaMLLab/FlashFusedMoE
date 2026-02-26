"""Expert Parallelism for fused MoE.

Distributes experts across multiple GPUs using all-to-all communication.
Each GPU owns a subset of experts and processes only their tokens.

Architecture:
    GPU 0: experts [0,1]     GPU 1: experts [2,3]
    GPU 2: experts [4,5]     GPU 3: experts [6,7]

    Token flow:
    1. All GPUs compute routing (replicated gate_weight)
    2. All-to-all DISPATCH: send tokens to GPU owning target expert
    3. Each GPU runs fused_moe on LOCAL experts only
    4. All-to-all COMBINE: results sent back to originating GPU
    5. Weighted sum at originating GPU

Backends:
    - "nccl": NCCL-based all-to-all (default, works everywhere)
    - "deep_ep": DeepEP NVSHMEM-based fused dispatch/combine (requires deep_ep library)
    - "hybrid": Auto-selects deep_ep if available, falls back to NCCL

Optimizations (Megatron-inspired):
    - Async functional collective for count exchange (overlaps with sort)
    - DtoH transfer of split sizes on a side CUDA stream (overlaps with pack)
    - Single packed A2A: tokens + expert_ids in [M, D+1] buffer
    - DispatchHandle dataclass for clean metadata passing

Includes MockExpertParallelDispatcher for single-GPU testing.
"""

from dataclasses import dataclass
import torch

from flashfusemoe.kernels.fused_dispatch import fused_dispatch_pack

# Guarded import for DeepEP (NVSHMEM-based fused dispatch/combine)
try:
    import deep_ep
    HAS_DEEP_EP = True
except ImportError:
    deep_ep = None
    HAS_DEEP_EP = False

_fc = None  # Lazy import of torch.distributed._functional_collectives


def _get_fc():
    """Lazy import functional collectives to avoid side effects on single-GPU."""
    global _fc
    if _fc is None:
        import torch.distributed._functional_collectives as fc
        _fc = fc
    return _fc


@dataclass
class DispatchHandle:
    """Metadata from dispatch needed for combine. Replaces dict-based metadata."""
    sort_order: torch.Tensor      # [N*top_k] dispatch permutation
    send_splits: list[int]        # tokens sent to each EP rank
    recv_splits: list[int]        # tokens received from each EP rank
    N: int
    top_k: int
    local_expert_ids: torch.Tensor  # [M] local expert indices for received tokens


class _AllToAllAutograd(torch.autograd.Function):
    """Differentiable all-to-all for token tensors.

    Forward: all_to_all_single(output, input, output_split_sizes, input_split_sizes, group)
    Backward: reverse all-to-all (swap split sizes)
    """

    @staticmethod
    def forward(ctx, input_tensor, output_split_sizes, input_split_sizes, group):
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        output = _get_fc().all_to_all_single(
            input_tensor.contiguous(), output_split_sizes, input_split_sizes, group
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Reverse: swap send/recv splits
        grad_input = _get_fc().all_to_all_single(
            grad_output.contiguous(), ctx.input_split_sizes, ctx.output_split_sizes, ctx.group
        )
        return grad_input, None, None, None


class ExpertParallelDispatcher:
    """Dispatches tokens to expert-owning GPUs via all-to-all.

    Requires torch.distributed to be initialized with a process group.
    """

    def __init__(self, num_experts, ep_size, process_group=None):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_gpu = num_experts // ep_size
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"
        self.process_group = process_group
        # Side-stream for async DtoH of split sizes (Megatron pattern)
        # Created eagerly here (outside compiled graph) to avoid pin_memory lowering issues.
        self._dtoh_stream = None
        # Pre-allocated pinned CPU buffers for split sizes
        self._send_splits_cpu = torch.empty(ep_size, dtype=torch.long, pin_memory=True)
        self._recv_splits_cpu = torch.empty(ep_size, dtype=torch.long, pin_memory=True)

    def _get_group(self):
        if self.process_group is not None:
            return self.process_group
        return torch.distributed.group.WORLD

    def dispatch(self, hidden_states, topk_indices, topk_weights):
        """Send tokens to their expert's GPU via all-to-all.

        Args:
            hidden_states: [N, D] input tokens (may differ per GPU)
            topk_indices: [N, top_k] expert assignments
            topk_weights: [N, top_k] routing weights

        Returns:
            received_tokens: [M, D] tokens received for local experts
            metadata: dict with info needed for combine
        """
        N, D = hidden_states.shape
        top_k = topk_indices.shape[1]
        device = hidden_states.device
        group = self._get_group()
        rank = torch.distributed.get_rank(group)

        # Determine which GPU each token-expert pair goes to
        dest_gpu = topk_indices // self.experts_per_gpu  # [N, top_k]

        # Count tokens going to each GPU
        flat_dest = dest_gpu.view(-1)
        send_counts = torch.bincount(flat_dest.int(), minlength=self.ep_size).to(torch.long)

        # Exchange counts — each GPU may have different data so counts differ
        recv_counts = torch.zeros(self.ep_size, dtype=torch.long, device=device)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)

        # Pack tokens by destination GPU
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        # Sort by destination
        sort_order = torch.argsort(flat_dest, stable=True)
        sorted_hidden = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D)
        sorted_hidden = sorted_hidden[sort_order]
        sorted_expert_ids = flat_indices[sort_order]
        sorted_weights = flat_weights[sort_order]

        # All-to-all exchange of tokens
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        received_tokens = _get_fc().all_to_all_single(
            sorted_hidden, recv_splits, send_splits, group
        ).clone()
        received_expert_ids = _get_fc().all_to_all_single(
            sorted_expert_ids, recv_splits, send_splits, group
        ).clone()
        received_weights = _get_fc().all_to_all_single(
            sorted_weights, recv_splits, send_splits, group
        ).clone()

        # Remap expert IDs to local expert indices
        local_expert_ids = received_expert_ids - rank * self.experts_per_gpu

        metadata = {
            "sort_order": sort_order,
            "send_counts": send_counts,
            "recv_counts": recv_counts,
            "send_splits": send_splits,
            "recv_splits": recv_splits,
            "N": N,
            "top_k": top_k,
            "local_expert_ids": local_expert_ids,
            "received_weights": received_weights,
        }

        return received_tokens, metadata

    def _ensure_dtoh_stream(self, device):
        """Lazily create DtoH side stream (device-dependent, so deferred from __init__)."""
        if self._dtoh_stream is None:
            self._dtoh_stream = torch.cuda.Stream(device=device)

    def dispatch_fused(self, hidden_states, topk_indices):
        """Fused dispatch with async count exchange and DtoH overlap.

        Optimizations (Megatron-inspired):
        1. Async functional collective for count exchange (overlaps with sort)
        2. DtoH transfer on side stream with CUDA events (overlaps with pack)
        3. Single packed A2A: tokens + expert_ids in [M, D+1] buffer

        Args:
            hidden_states: [N, D] input tokens (replicated on all GPUs)
            topk_indices: [N, top_k] expert assignments

        Returns:
            received_tokens: [M, D] tokens received for local experts
            handle: DispatchHandle with metadata for combine
        """
        N, D = hidden_states.shape
        top_k = topk_indices.shape[1]
        device = hidden_states.device
        group = self._get_group()
        rank = torch.distributed.get_rank(group)

        dest_gpu = topk_indices // self.experts_per_gpu  # [N, top_k]
        flat_dest = dest_gpu.view(-1)  # [N*top_k]
        flat_indices = topk_indices.view(-1)

        # Count tokens going to each GPU
        send_counts = torch.bincount(flat_dest.int(), minlength=self.ep_size).to(torch.long)

        # Exchange counts (blocking — only 8 int64 values, negligible cost)
        recv_counts = torch.zeros(self.ep_size, dtype=torch.long, device=device)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)

        # Launch async DtoH on side stream (Megatron pattern: overlaps with sort + pack)
        self._ensure_dtoh_stream(device)
        dtoh_stream = self._dtoh_stream
        dtoh_ready = torch.cuda.Event()
        dtoh_ready.record()
        with torch.cuda.stream(dtoh_stream):
            dtoh_stream.wait_event(dtoh_ready)
            self._send_splits_cpu.copy_(send_counts, non_blocking=True)
            self._recv_splits_cpu.copy_(recv_counts, non_blocking=True)
        dtoh_event = dtoh_stream.record_event()

        # Sort by destination GPU (overlaps with DtoH above)
        sort_order = torch.argsort(flat_dest, stable=True)

        # Fused pack: single Triton kernel replaces expand + index + pack (3 ops → 1)
        packed_send = fused_dispatch_pack(
            hidden_states, sort_order, flat_indices, top_k,
        )

        # Synchronize DtoH — split sizes now available on CPU
        dtoh_event.synchronize()
        send_splits = self._send_splits_cpu.tolist()
        recv_splits = self._recv_splits_cpu.tolist()

        # Main token A2A
        packed_recv = _get_fc().all_to_all_single(
            packed_send, recv_splits, send_splits, group
        )

        # .clone() forces full materialization of async collective result
        packed_recv = packed_recv.clone()
        received_tokens = packed_recv[:, :D].contiguous()
        received_expert_ids = packed_recv[:, D].to(torch.long)
        local_expert_ids = received_expert_ids - rank * self.experts_per_gpu

        handle = DispatchHandle(
            sort_order=sort_order,
            send_splits=send_splits,
            recv_splits=recv_splits,
            N=N,
            top_k=top_k,
            local_expert_ids=local_expert_ids,
        )

        return received_tokens, handle

    def fused_combine(self, expert_outputs, handle):
        """Fused combine: reverse A2A + unsort using DispatchHandle.

        Args:
            expert_outputs: [M, D] outputs from local expert computation
            handle: DispatchHandle from dispatch_fused()

        Returns:
            returned: [N*top_k, D] unsorted results (caller applies weights + sum)
        """
        group = self._get_group()

        # Reverse splits for combine direction
        send_splits = handle.recv_splits  # we received these, now send them back
        recv_splits = handle.send_splits

        returned = _get_fc().all_to_all_single(
            expert_outputs.contiguous(), recv_splits, send_splits, group
        ).clone()

        # Unsort back to [N*top_k, D]
        unsorted = torch.empty_like(returned)
        unsorted[handle.sort_order] = returned
        return unsorted

    def dispatch_differentiable(self, hidden_states, topk_indices, topk_weights):
        """Differentiable dispatch: gradients flow back through all-to-all for tokens.

        Same logic as dispatch() but uses _AllToAllAutograd for the token tensor.
        Expert IDs and weights use non-differentiable all_to_all_single.

        Args:
            hidden_states: [N, D] input tokens (requires_grad ok)
            topk_indices: [N, top_k] expert assignments
            topk_weights: [N, top_k] routing weights

        Returns:
            received_tokens: [M, D] tokens received for local experts (grad-enabled)
            metadata: dict with info needed for combine_differentiable
        """
        N, D = hidden_states.shape
        top_k = topk_indices.shape[1]
        device = hidden_states.device
        group = self._get_group()
        rank = torch.distributed.get_rank(group)

        dest_gpu = topk_indices // self.experts_per_gpu  # [N, top_k]

        # Count tokens going to each GPU
        flat_dest = dest_gpu.view(-1)
        send_counts = torch.bincount(flat_dest.int(), minlength=self.ep_size).to(torch.long)

        # Exchange counts — each GPU may have different data so counts differ
        recv_counts = torch.zeros(self.ep_size, dtype=torch.long, device=device)
        torch.distributed.all_to_all_single(recv_counts, send_counts, group=group)
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        sort_order = torch.argsort(flat_dest, stable=True)
        hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D)
        sorted_hidden = hidden_expanded[sort_order]
        sorted_expert_ids = flat_indices[sort_order]
        sorted_weights = flat_weights[sort_order]

        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # Differentiable all-to-all for tokens
        received_tokens = _AllToAllAutograd.apply(
            sorted_hidden, recv_splits, send_splits, group
        )

        # Non-differentiable all-to-all for expert_ids and weights (metadata)
        received_expert_ids = _get_fc().all_to_all_single(
            sorted_expert_ids, recv_splits, send_splits, group
        )
        received_weights = _get_fc().all_to_all_single(
            sorted_weights, recv_splits, send_splits, group
        )

        local_expert_ids = received_expert_ids - rank * self.experts_per_gpu

        metadata = {
            "sort_order": sort_order,
            "send_counts": send_counts,
            "recv_counts": recv_counts,
            "send_splits": send_splits,
            "recv_splits": recv_splits,
            "N": N,
            "top_k": top_k,
            "local_expert_ids": local_expert_ids,
            "received_weights": received_weights,
        }

        return received_tokens, metadata

    def combine_differentiable(self, expert_outputs, metadata):
        """Differentiable combine: gradients flow back through all-to-all.

        Args:
            expert_outputs: [M, D] outputs from local expert computation
            metadata: dict from dispatch_differentiable()

        Returns:
            output: [N, D] final weighted-summed output
        """
        group = self._get_group()
        N = metadata["N"]
        top_k = metadata["top_k"]
        sort_order = metadata["sort_order"]

        send_splits = metadata["recv_splits"]  # Reversed!
        recv_splits = metadata["send_splits"]

        # Differentiable all-to-all back
        returned = _AllToAllAutograd.apply(
            expert_outputs, recv_splits, send_splits, group
        )

        # Unsort and scatter-add to original positions
        unsorted = torch.empty_like(returned)
        unsorted[sort_order] = returned

        output = unsorted.view(N, top_k, returned.shape[-1]).sum(dim=1)
        return output

    def combine(self, expert_outputs, metadata):
        """Gather expert outputs back to originating GPU via all-to-all.

        Args:
            expert_outputs: [M, D] outputs from local expert computation
            metadata: dict from dispatch()

        Returns:
            output: [N, D] final weighted-summed output
        """
        group = self._get_group()
        D = expert_outputs.shape[-1]
        device = expert_outputs.device

        send_splits = metadata["recv_splits"]  # Reversed!
        recv_splits = metadata["send_splits"]

        # Send expert outputs back
        returned = _get_fc().all_to_all_single(
            expert_outputs.contiguous(), recv_splits, send_splits, group
        ).clone()

        # Unsort and scatter-add to original positions
        N = metadata["N"]
        top_k = metadata["top_k"]
        sort_order = metadata["sort_order"]

        # Reverse the sort
        unsorted = torch.empty_like(returned)
        unsorted[sort_order] = returned

        # Reshape to [N, top_k, D] and sum
        output = unsorted.view(N, top_k, D).sum(dim=1)
        return output


class MockExpertParallelDispatcher:
    """Simulates EP on single GPU by logically partitioning experts.

    No actual communication — permutes tokens locally as if they were
    sent via all-to-all. Tracks simulated communication volume.

    This allows testing EP logic without requiring multi-GPU hardware
    or torch.distributed initialization.
    """

    def __init__(self, num_experts, ep_size):
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_gpu = num_experts // ep_size
        assert num_experts % ep_size == 0, \
            f"num_experts ({num_experts}) must be divisible by ep_size ({ep_size})"

        # Communication tracking
        self.dispatch_bytes = 0
        self.combine_bytes = 0
        self.total_dispatch_tokens = 0
        self.total_combine_tokens = 0

    def reset_stats(self):
        self.dispatch_bytes = 0
        self.combine_bytes = 0
        self.total_dispatch_tokens = 0
        self.total_combine_tokens = 0

    def dispatch(self, hidden_states, topk_indices, topk_weights):
        """Simulate dispatch: partition tokens by destination GPU.

        For each simulated GPU rank, collect the tokens destined for its experts.

        Args:
            hidden_states: [N, D]
            topk_indices: [N, top_k]
            topk_weights: [N, top_k]

        Returns:
            List of (received_tokens, metadata) tuples, one per simulated GPU.
        """
        N, D = hidden_states.shape
        top_k = topk_indices.shape[1]
        device = hidden_states.device
        dtype = hidden_states.dtype
        dtype_size = hidden_states.element_size()

        dest_gpu = topk_indices // self.experts_per_gpu  # [N, top_k]
        flat_dest = dest_gpu.view(-1)
        flat_indices = topk_indices.view(-1)
        flat_weights = topk_weights.view(-1)

        # Expand hidden to [N*top_k, D]
        hidden_expanded = hidden_states.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D)

        results = []
        for rank in range(self.ep_size):
            mask = flat_dest == rank
            count = mask.sum().item()

            received_tokens = hidden_expanded[mask]  # [count, D]
            received_expert_ids = flat_indices[mask] - rank * self.experts_per_gpu
            received_weights = flat_weights[mask]

            # Track simulated communication (tokens crossing GPU boundaries)
            self.dispatch_bytes += count * D * dtype_size
            self.total_dispatch_tokens += count

            metadata = {
                "mask": mask,
                "N": N,
                "top_k": top_k,
                "D": D,
                "rank": rank,
                "local_expert_ids": received_expert_ids,
                "received_weights": received_weights,
                "count": count,
            }
            results.append((received_tokens, metadata))

        return results

    def combine(self, all_expert_outputs, all_metadata):
        """Simulate combine: gather outputs back and sum.

        Args:
            all_expert_outputs: list of [M_rank, D] tensors per GPU
            all_metadata: list of metadata dicts per GPU

        Returns:
            output: [N, D] final weighted output
        """
        meta0 = all_metadata[0]
        N = meta0["N"]
        top_k = meta0["top_k"]
        D = meta0["D"]
        device = all_expert_outputs[0].device
        dtype = all_expert_outputs[0].dtype
        dtype_size = all_expert_outputs[0].element_size()

        # Reconstruct full [N*top_k, D] output
        flat_output = torch.zeros(N * top_k, D, dtype=dtype, device=device)

        for rank in range(self.ep_size):
            mask = all_metadata[rank]["mask"]
            expert_out = all_expert_outputs[rank]
            count = all_metadata[rank]["count"]

            # Track combine communication
            self.combine_bytes += count * D * dtype_size
            self.total_combine_tokens += count

            flat_output[mask] = expert_out

        # Sum over top_k
        output = flat_output.view(N, top_k, D).sum(dim=1)
        return output

    def get_comm_stats(self):
        """Return simulated communication statistics."""
        return {
            "dispatch_bytes": self.dispatch_bytes,
            "combine_bytes": self.combine_bytes,
            "total_bytes": self.dispatch_bytes + self.combine_bytes,
            "dispatch_tokens": self.total_dispatch_tokens,
            "combine_tokens": self.total_combine_tokens,
        }


class DeepEPDispatcher:
    """DeepEP-based fused dispatch/combine using NVSHMEM.

    Replaces separate NCCL all-to-all with DeepEP's fused kernel that merges
    token permutation with NVSHMEM communication. Uses handle-based API:
    dispatch returns a handle, combine takes that handle.

    Reference: Megatron-LM _DeepepManager in token_dispatcher.py

    Requires: deep_ep library (pip install deep_ep or build from source).
    """

    def __init__(self, num_experts, ep_size, process_group=None,
                 num_nvl_bytes=None, num_rdma_bytes=None):
        if not HAS_DEEP_EP:
            raise ImportError(
                "deep_ep library not found. Install with: pip install deep_ep"
            )
        self.num_experts = num_experts
        self.ep_size = ep_size
        self.experts_per_gpu = num_experts // ep_size
        assert num_experts % ep_size == 0
        self.process_group = process_group

        # Create DeepEP buffer — handles NVSHMEM allocation
        rank = torch.distributed.get_rank(process_group)
        self.buffer = deep_ep.Buffer(
            group=process_group,
            num_nvl_bytes=num_nvl_bytes or (256 * 1024 * 1024),  # 256MB default
            num_rdma_bytes=num_rdma_bytes or (256 * 1024 * 1024),
        )

    def dispatch_fused(self, hidden_states, topk_indices):
        """Fused dispatch using DeepEP NVSHMEM kernel.

        API matches ExpertParallelDispatcher.dispatch_fused() for drop-in use.

        Args:
            hidden_states: [N, D] input tokens
            topk_indices: [N, top_k] expert assignments

        Returns:
            received_tokens: [M, D] tokens for local experts
            handle: DispatchHandle (or DeepEP handle) for combine
        """
        N, D = hidden_states.shape
        top_k = topk_indices.shape[1]
        rank = torch.distributed.get_rank(self.process_group)

        # DeepEP dispatch: fused permute + NVSHMEM send
        # Returns (received_tokens, handle) where handle has metadata for combine
        received_tokens, deep_handle = self.buffer.dispatch(
            hidden_states, topk_indices,
            num_experts=self.num_experts,
            async_finish=True,
            allocate_on_comm_stream=True,
        )

        # Map expert IDs to local indices
        local_expert_ids = deep_handle.expert_ids - rank * self.experts_per_gpu

        # Wrap in our DispatchHandle for API compatibility
        handle = DispatchHandle(
            sort_order=deep_handle.sort_indices,
            send_splits=deep_handle.send_counts.tolist(),
            recv_splits=deep_handle.recv_counts.tolist(),
            N=N,
            top_k=top_k,
            local_expert_ids=local_expert_ids,
        )
        handle._deep_handle = deep_handle  # keep for combine

        return received_tokens, handle

    def fused_combine(self, expert_outputs, handle):
        """Fused combine using DeepEP NVSHMEM kernel.

        Args:
            expert_outputs: [M, D] outputs from local expert computation
            handle: DispatchHandle from dispatch_fused() (with _deep_handle)

        Returns:
            returned: [N*top_k, D] unsorted results
        """
        deep_handle = handle._deep_handle
        returned = self.buffer.combine(
            expert_outputs, deep_handle,
            async_finish=True,
        )

        # DeepEP combine returns [N*top_k, D] already unsorted
        return returned


def create_dispatcher(num_experts, ep_size, process_group=None, backend="nccl", **kwargs):
    """Factory for creating the appropriate EP dispatcher.

    Args:
        num_experts: Total number of experts across all GPUs
        ep_size: Number of GPUs for expert parallelism
        process_group: torch.distributed ProcessGroup
        backend: "nccl" (default), "deep_ep", or "hybrid"
        **kwargs: Additional args passed to the specific dispatcher (e.g. num_nvl_bytes)

    Returns:
        Dispatcher instance (ExpertParallelDispatcher or DeepEPDispatcher)
    """
    if backend == "deep_ep":
        if not HAS_DEEP_EP:
            raise ImportError(
                "deep_ep library not found. Install with: pip install deep_ep. "
                "Use backend='nccl' or backend='hybrid' for automatic fallback."
            )
        return DeepEPDispatcher(num_experts, ep_size, process_group, **kwargs)

    if backend == "hybrid":
        if HAS_DEEP_EP:
            return DeepEPDispatcher(num_experts, ep_size, process_group, **kwargs)
        # Fallback to NCCL
        return ExpertParallelDispatcher(num_experts, ep_size, process_group)

    # Default: NCCL
    return ExpertParallelDispatcher(num_experts, ep_size, process_group)
