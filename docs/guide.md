# Composing Everything Together

This guide shows how to build a model with FlashFuseMoE from scratch.

## Single-GPU quickstart

```python
import torch
import torch.nn as nn
from flashfusemoe import MoELayer

class MyBlock(nn.Module):
    def __init__(self, d_model, ffn_dim, num_experts, top_k):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.moe = MoELayer(d_model, ffn_dim, num_experts, top_k,
                            aux_loss_coeff=0.01)

    def forward(self, x):
        moe_out, losses = self.moe(self.norm(x))
        return x + moe_out, losses

block = MyBlock(1024, 4096, 8, 2).cuda()
x = torch.randn(4, 128, 1024, device="cuda", dtype=torch.bfloat16)
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    out, losses = block(x)
```

## torch.compile

The MoE layer works with torch.compile. The non-EP path is `@allow_in_graph` compatible:

```python
model = torch.compile(model)
```

## Multi-GPU EP setup

```python
import torch.distributed as dist
from flashfusemoe import MoELayer, OverlappedGradSync, get_dense_params

dist.init_process_group("nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

# Create model
model = MyModel().cuda()

# Shard experts
ep_group = dist.new_group(ranks=list(range(world_size)))
for module in model.modules():
    if isinstance(module, MoELayer):
        module.shard_experts(ep_group)

# Optimizer (after sharding)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Overlapped gradient sync
dense_params = get_dense_params(model, backward_order=True)
grad_syncer = OverlappedGradSync(dense_params, ep_group, num_buckets=4)

# Training loop
for batch in data:
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output, losses = model(batch)
    loss = compute_loss(output) + losses.get("aux_loss", 0)
    loss.backward()
    grad_syncer.finish()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

grad_syncer.remove_hooks()
```

## Adding custom optimizers

The library is optimizer-agnostic. Use any optimizer. For Muon or other distributed optimizers, just pass the right parameter groups:

```python
from flashfusemoe import get_dense_params

dense_params = get_dense_params(model)
expert_params = []
for m in model.modules():
    if isinstance(m, MoELayer):
        expert_params.extend(m.expert_params)

# Muon for dense, AdamW for experts
muon_opt = Muon(dense_params, lr=0.02)
adam_opt = torch.optim.AdamW(expert_params, lr=3e-4)
```

## MTP compatibility

The library doesn't know about prediction heads. MTP or any other auxiliary prediction scheme works by just connecting MoE layers in your model however you want. The `(output, losses)` return convention means MoE loss terms compose naturally with CE loss and MTP loss.

## Full working example

See `examples/full_integration.py` for a complete runnable example with everything above.

```bash
# Single GPU
python examples/full_integration.py

# Multi GPU
torchrun --nproc_per_node=2 examples/full_integration.py --ep_size 2
```

## Benchmark numbers

H100 80GB, 2 GPUs, ~500M active / ~1.1B total params, BF16:

| Config | p50 (ms) | Agg Tok/s | Peak Memory |
|--------|----------|-----------|-------------|
| Fused EP=1 | 137 | 57k | 35 GB |
| Fused EP=2 | 196 | 85k | 27 GB |
| Fused EP=2, compile | 196 | 85k | 27 GB |
| + OverlappedGradSync | comm 3.6ms (1.2% of step) | | |
