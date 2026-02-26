# FlashFuseMoE

Fused gate-to-output MoE layer with Expert Parallelism. Triton kernels, torch.compile support, overlapped communication.

## Install

```bash
pip install -e .
```

Requires PyTorch >= 2.4 and Triton >= 3.0.

## Quick Start

### MoELayer (nn.Module)

```python
from flashfusemoe import MoELayer

moe = MoELayer(
    d_model=1024, ffn_dim=4096, num_experts=8, top_k=2,
    activation="swiglu", num_shared_experts=1,
)
output, losses = moe(x)  # x: [B, T, D] -> ([B, T, D], dict)
```

### Low-level fused_moe()

```python
from flashfusemoe import fused_moe

output, losses = fused_moe(
    hidden_states,   # [N, D]
    gate_weight,     # [E, D]
    w_up, w_down,    # [E, ffn_dim, D], [E, D, ffn_dim//2]
    top_k=2,
    activation="swiglu",
    aux_loss_coeff=0.01,
)
```

### Expert Parallelism

```python
from flashfusemoe import MoELayer, OverlappedGradSync, get_dense_params

moe = MoELayer(d_model=1024, ffn_dim=4096, num_experts=8, top_k=2)
moe.shard_experts(ep_group)  # shard weights across GPUs

# Overlapped dense gradient sync during backward
dense_params = get_dense_params(model, backward_order=True)
grad_syncer = OverlappedGradSync(dense_params, ep_group, num_buckets=4)
loss.backward()
grad_syncer.finish()
```

## Features

- **Fused Triton kernels**: routing scatter/gather, GLU activations, topk+scoring all fused. 5 kernel launches per layer vs 23 unfused.
- **cuBLAS batched GEMM**: all expert matmuls via `torch.bmm` on padded `[E, max_tokens, D]` layout. No Python expert loops.
- **torch.compile**: non-EP path is `@allow_in_graph` compatible. EP path uses `@compiler.disable` (NCCL can't be compiled).
- **Expert Parallelism**: fused dispatch/combine with async DtoH, packed A2A, hand-written routing backward (~10 ops vs hundreds).
- **Communication overlap**: shared expert on side stream during dispatch A2A, A2A combine on side stream during routing backward, bucketed dense grad allreduce during backward.
- **Shared experts**: always-active expert overlapped with dispatch A2A in EP forward.
- **Latent projection**: project D to smaller dim before A2A to reduce communication volume.
- **Auxiliary losses**: Switch Transformer load-balance loss, router z-loss.
- **Activation checkpointing**: `recompute_activations=True` saves ~35% activation memory.
- **DeepEP support**: optional NVSHMEM-based fused dispatch/combine via `deep_ep` library.

## Benchmarks

H100 80GB, 2 GPUs, ~500M active / ~1.1B total MoE model, BF16, torch.compile:

| Mode | EP | p50 (ms) | Agg Tok/s | Peak Memory |
|------|-----|----------|-----------|-------------|
| Fused BF16 | 1 | 137 | 57k | 35 GB |
| Fused BF16 | 2 | 196 | 85k | 27 GB |

Fused EP=2 uses 44% less memory than unfused (27 GB vs 47 GB) due to custom autograd saving only essential tensors (158 MB/layer vs 740 MB/layer).

## Docs

- [docs/fused_moe.md](docs/fused_moe.md) -- the fused MoE kernel
- [docs/expert_parallelism.md](docs/expert_parallelism.md) -- expert parallelism
- [docs/communication_overlap.md](docs/communication_overlap.md) -- communication overlap
- [docs/kernels.md](docs/kernels.md) -- Triton kernel details
- [docs/moe_layer.md](docs/moe_layer.md) -- MoELayer nn.Module
- [docs/guide.md](docs/guide.md) -- composing everything together

## Examples

```bash
# Single GPU
python examples/full_integration.py

# Multi GPU EP
torchrun --nproc_per_node=2 examples/full_integration.py --ep_size 2
```

## License

GPL-3.0
