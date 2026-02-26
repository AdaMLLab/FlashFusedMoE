# Expert Parallelism

Expert Parallelism (EP) distributes experts across GPUs. Each GPU owns `E/ep_size` experts and processes only their tokens.

## How it works

```
GPU 0: experts [0,1,2,3]     GPU 1: experts [4,5,6,7]

Forward:
1. All GPUs compute routing (gate weight is replicated)
2. DISPATCH: all-to-all sends tokens to expert-owning GPU
3. Each GPU runs fused_moe on LOCAL experts only
4. COMBINE: all-to-all sends results back to originating GPU
5. Weighted sum at originating GPU
```

### Dispatch

The `ExpertParallelDispatcher` handles token routing:

1. Count tokens going to each GPU (bincount on destination)
2. Exchange counts (blocking all-to-all, just 8 int64 values)
3. Sort tokens by destination GPU (argsort)
4. Pack tokens + expert IDs into single buffer via `fused_dispatch_pack` Triton kernel (3 ops fused into 1)
5. Token all-to-all exchange
6. Remap expert IDs to local indices

The `DispatchHandle` dataclass carries metadata needed for combine (sort order, split sizes, local expert IDs).

### Combine

Reverse all-to-all sends expert outputs back to originating GPUs. Unsort restores original token order. Caller applies routing weights and sums over top-k.

## Memory savings

EP=2 uses 44% less peak memory than unfused EP=2:

| Mode | EP | Peak Memory | Activation/layer |
|------|-----|-------------|-----------------|
| Unfused BF16 | 2 | 47 GB | 740 MB |
| Fused BF16 | 2 | 27 GB | 158 MB |

The gap comes from the custom autograd function saving only essential tensors. PyTorch autograd saves every intermediate for every op in the unfused path (per-expert loop intermediates, expanded tensors, shared expert intermediates).

## Throughput scaling

H100 80GB, BF16, torch.compile:

| EP | p50 (ms) | Agg Tok/s | Peak Memory |
|-----|----------|-----------|-------------|
| 1 | 137 | 57k | 35 GB |
| 2 | 196 | 85k | 27 GB |

## Setup

### With MoELayer

```python
from flashfusemoe import MoELayer

moe = MoELayer(d_model=1024, ffn_dim=4096, num_experts=8, top_k=2)
moe.shard_experts(ep_group, ep_backend="nccl")
```

### With fused_moe() directly

```python
from flashfusemoe import shard_expert_weights, set_ep_group

w_up_local, w_down_local = shard_expert_weights(w_up, w_down, ep_group)
set_ep_group(ep_group, backend="nccl")

# Now fused_moe() will use EP path automatically
output, losses = fused_moe(x, gate_weight, w_up_local, w_down_local, top_k=2)
```

### With DeepSeek model (examples/)

```python
from deepseek.model import DeepSeekTransformer, DeepSeekConfig, shard_experts

model = DeepSeekTransformer(config).cuda()
shard_experts(model, ep_group, ep_backend="nccl")
```

## EP backward

The EP backward is hand-written inside `_FusedMoEEPAutograd`:

1. Routing weight gradient (no A2A needed, ~10 ops)
2. A2A combine-reverse on side CUDA stream, overlapped with routing grad on default stream
3. Weight grads via bmm on default stream after A2A sync

This replaces the traced-through-autograd backward (hundreds of small PyTorch kernels) with ~10 explicit ops. The routing backward alone saved ~28ms per step.

## EP backends

- `"nccl"` (default): standard NCCL all-to-all
- `"deep_ep"`: DeepEP NVSHMEM-based fused dispatch/combine (requires deep_ep library)
- `"hybrid"`: auto-selects deep_ep if available, falls back to NCCL
