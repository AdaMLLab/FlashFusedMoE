# MoELayer nn.Module

`MoELayer` is a drop-in replacement for an FFN block. It wraps `fused_moe()` with proper weight initialization, shared expert support, and EP integration.

## API

```python
from flashfusemoe import MoELayer

moe = MoELayer(
    d_model=1024,          # input/output dimension
    ffn_dim=4096,          # expert FFN dimension (gate+up stacked)
    num_experts=8,         # total number of routed experts
    top_k=2,               # experts per token
    activation="swiglu",   # "swiglu", "geglu", or "relu_squared"
    gating="softmax",      # "softmax" or "sigmoid"
    num_shared_experts=0,  # always-active shared experts (0 = none)
    latent_dim=None,       # latent dim for A2A compression (None = off)
    capacity_factor=None,  # None = no dropping, float = capacity multiplier
    max_tokens_per_expert=None,  # buffer size per expert (None = heuristic)
    allow_dropped_tokens=True,   # allow silent token dropping on overflow
    aux_loss_coeff=0.0,    # Switch Transformer load-balance loss
    z_loss_coeff=0.0,      # router z-loss coefficient
    recompute_activations=False,  # recompute activations in backward
)

output, losses = moe(x)  # x: [*, D] -> ([*, D], dict)
```

## Return value

Always returns `(output, losses)` tuple. `losses` is a dict that contains:
- `"aux_loss"`: load-balance loss (when `aux_loss_coeff > 0`)
- `"z_loss"`: router z-loss (when `z_loss_coeff > 0`)

Empty dict when no losses are enabled.

## Shared experts

Set `num_shared_experts > 0` to add an always-active shared expert. In EP mode, the shared expert forward runs on a side CUDA stream during the dispatch all-to-all, hiding its latency.

```python
moe = MoELayer(d_model=1024, ffn_dim=4096, num_experts=8, top_k=2,
               num_shared_experts=1)
```

## Latent projection

Set `latent_dim` to project tokens to a smaller dimension before A2A dispatch, reducing communication volume. Useful at high EP sizes.

```python
moe = MoELayer(d_model=7168, ffn_dim=18432, num_experts=64, top_k=6,
               latent_dim=2048)  # 3.5x less A2A traffic
```

## Expert Parallelism

Call `shard_experts()` before optimizer creation:

```python
moe = MoELayer(d_model=1024, ffn_dim=4096, num_experts=8, top_k=2)
moe = moe.cuda()
moe.shard_experts(ep_group, ep_backend="nccl")
optimizer = torch.optim.AdamW(moe.parameters())
```

This slices `w_up` and `w_down` to the local shard and sets the EP group for the fused_moe path.

## Expert params

Use `moe.expert_params` to get the list of expert parameters (sharded across EP ranks). Useful for excluding them from dense gradient sync:

```python
expert_ids = {id(p) for p in moe.expert_params}
dense_params = [p for p in model.parameters() if id(p) not in expert_ids]
```

Or use the convenience function:

```python
from flashfusemoe import get_dense_params
dense_params = get_dense_params(model)
```

## Weight initialization

- Gate weight: kaiming_uniform scaled by 0.01
- Expert weights (w_up, w_down): kaiming_uniform per expert
