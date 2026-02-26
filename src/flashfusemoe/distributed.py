"""EP utilities: weight sharding, overlapped gradient sync.

Usage:
    from flashfusemoe import shard_expert_weights, OverlappedGradSync, get_dense_params
"""

import torch
import torch.nn as nn
import torch.distributed as dist


def shard_expert_weights(w_up, w_down, ep_group):
    """Slice expert weight tensors to local shard for this rank.

    Args:
        w_up: nn.Parameter [E, ffn_dim, D]
        w_down: nn.Parameter [E, D, ffn_dim//2]
        ep_group: torch.distributed ProcessGroup

    Returns:
        (w_up_local, w_down_local) as nn.Parameters
    """
    rank = dist.get_rank(ep_group)
    ep_size = dist.get_world_size(ep_group)
    E = w_up.shape[0]
    assert E % ep_size == 0, f"n_experts ({E}) must be divisible by ep_size ({ep_size})"
    E_local = E // ep_size
    start = rank * E_local
    end = start + E_local

    w_up_local = nn.Parameter(w_up.data[start:end].contiguous())
    w_down_local = nn.Parameter(w_down.data[start:end].contiguous())
    return w_up_local, w_down_local


def get_dense_params(model, backward_order=True):
    """Get non-expert params from a model containing MoELayers.

    Detects expert params by checking for MoELayer.expert_params or
    by the "w_up"/"w_down" naming convention (excluding shared experts).

    Args:
        model: nn.Module (may contain MoELayer instances)
        backward_order: if True, returns list reversed (last layer first)

    Returns:
        list of Parameters suitable for OverlappedGradSync
    """
    # Collect expert param ids from MoELayer instances
    expert_ids = set()

    # Import here to avoid circular import at module load
    from flashfusemoe.nn import MoELayer

    for m in model.modules():
        if isinstance(m, MoELayer):
            for p in m.expert_params:
                expert_ids.add(id(p))

    # If no MoELayers found, fall back to name-based detection
    if not expert_ids:
        for name, param in model.named_parameters():
            if ("w_up" in name or "w_down" in name) and "shared_expert" not in name:
                expert_ids.add(id(param))

    dense_params = []
    seen = set()
    for _name, param in model.named_parameters():
        if not param.requires_grad or id(param) in seen:
            continue
        seen.add(id(param))
        if id(param) in expert_ids:
            continue
        dense_params.append(param)

    if backward_order:
        dense_params = list(reversed(dense_params))

    return dense_params


class OverlappedGradSync:
    """Overlap dense gradient allreduce with backward computation.

    Uses register_post_accumulate_grad_hook to fire bucketed allreduces
    on a side CUDA stream as grads become available during backward.
    Only the final unpack after backward is sequential.

    Args:
        params: list of Parameters in backward order (last layer first)
        process_group: torch.distributed ProcessGroup for allreduce
        num_buckets: number of gradient buckets (default 4)
    """

    def __init__(self, params, process_group, num_buckets=4):
        self.process_group = process_group
        self.num_buckets = num_buckets
        self._hooks = []

        if not params:
            self.num_buckets = 0
            self.buckets = []
            return

        # Assign to buckets (roughly equal numel)
        total_numel = sum(p.numel() for p in params)
        target_per_bucket = total_numel / num_buckets
        self.buckets = []
        cur_params = []
        cur_numel = 0
        for p in params:
            cur_params.append(p)
            cur_numel += p.numel()
            if cur_numel >= target_per_bucket and len(self.buckets) < num_buckets - 1:
                self.buckets.append(self._make_bucket(cur_params))
                cur_params = []
                cur_numel = 0
        if cur_params:
            self.buckets.append(self._make_bucket(cur_params))

        self.num_buckets = len(self.buckets)

        # Map param id -> bucket index
        param_to_bucket = {}
        for bi, bucket in enumerate(self.buckets):
            for p in bucket["params"]:
                param_to_bucket[id(p)] = bi

        # Create side stream for comm
        device = params[0].device
        self._comm_stream = torch.cuda.Stream(device=device)

        # Register hooks
        for p in params:
            bi = param_to_bucket[id(p)]
            hook = p.register_post_accumulate_grad_hook(
                self._make_hook(bi)
            )
            self._hooks.append(hook)

    def _make_bucket(self, params):
        total = sum(p.numel() for p in params)
        device = params[0].device
        return {
            "params": list(params),
            "flat": torch.empty(total, dtype=params[0].dtype, device=device),
            "count": 0,
            "expected": len(params),
            "done_event": None,
        }

    def _make_hook(self, bucket_idx):
        def hook(param):
            bucket = self.buckets[bucket_idx]
            bucket["count"] += 1
            if bucket["count"] == bucket["expected"]:
                # Pack grads into flat buffer
                offset = 0
                for p in bucket["params"]:
                    n = p.numel()
                    bucket["flat"][offset:offset + n] = p.grad.view(-1)
                    offset += n
                # Record event on default stream (pack is done)
                ready = torch.cuda.Event()
                ready.record()
                # Launch allreduce on side stream
                with torch.cuda.stream(self._comm_stream):
                    self._comm_stream.wait_event(ready)
                    dist.all_reduce(bucket["flat"], op=dist.ReduceOp.AVG,
                                    group=self.process_group)
                    ev = torch.cuda.Event()
                    ev.record()
                    bucket["done_event"] = ev
        return hook

    def finish(self):
        """Wait for all bucket allreduces and unpack back to .grad tensors.

        Call this right after loss.backward() returns.
        """
        if self.num_buckets == 0:
            return
        # Wait for all comm to finish on default stream
        for bucket in self.buckets:
            if bucket["done_event"] is not None:
                torch.cuda.current_stream().wait_event(bucket["done_event"])
        # Unpack
        for bucket in self.buckets:
            offset = 0
            for p in bucket["params"]:
                n = p.numel()
                p.grad.copy_(bucket["flat"][offset:offset + n].view_as(p.grad))
                offset += n
        # Reset counters for next step
        self._reset()

    def _reset(self):
        for bucket in self.buckets:
            bucket["count"] = 0
            bucket["done_event"] = None

    def remove_hooks(self):
        """Cleanup hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
