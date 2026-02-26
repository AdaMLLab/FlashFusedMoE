# Communication Overlap

FlashFuseMoE uses 4 overlap techniques to hide communication behind computation.

## 1. Shared expert during dispatch A2A (forward)

When EP is active and a shared expert exists, the shared expert forward runs on a side CUDA stream while the dispatch all-to-all runs on the default stream.

```
Default stream:  [dispatch A2A] -----> [local expert compute] --> [combine A2A]
Side stream:     [shared expert fwd] ->
```

The shared expert weights are passed into `fused_moe()` so both can execute concurrently.

## 2. Combine A2A during routing backward (backward)

The combine-reverse all-to-all (sending gradients back to originating GPUs) runs on a side stream while the routing gradient computation runs on the default stream.

```
Default stream:  [routing grad (~10 ops)] -----> [wait A2A] -> [weight grads]
Side stream:     [combine-reverse A2A] --------->
```

This saves ~5ms per step because routing grad and A2A run concurrently.

## 3. Weight grads during combine A2A (backward)

Weight gradients for local experts compute on the default stream while the combine A2A completes on the side stream. After syncing, latent projection weight grads compute.

## 4. Bucketed dense grad sync (training loop)

`OverlappedGradSync` uses `register_post_accumulate_grad_hook` to fire bucketed allreduces on a side CUDA stream as gradients become available during backward.

```python
from flashfusemoe import OverlappedGradSync, get_dense_params

dense_params = get_dense_params(model, backward_order=True)
syncer = OverlappedGradSync(dense_params, ep_group, num_buckets=4)

loss.backward()   # hooks fire allreduces as buckets fill
syncer.finish()   # wait for all allreduces, unpack back to .grad
```

Buckets are ordered in reverse module order (last layer first) so the first bucket fills earliest and its allreduce overlaps maximally with remaining backward work.

### Phase timing breakdown (EP=2, H100 80GB)

| Phase | Before overlap | After overlap |
|-------|---------------|--------------|
| Pack dense grads | 2.4 ms | (hidden) |
| All-reduce | 3.3 ms | (hidden) |
| Unpack | 1.7 ms | 1.7 ms |
| **Total comm** | **7.4 ms** | **3.6 ms** |

The residual 3.6ms is because `torch.compile` batches backward graph execution, limiting the overlap window. Only the final unpack remains sequential.

## Using OverlappedGradSync

```python
# Get dense (non-expert) params in backward order
dense_params = get_dense_params(model, backward_order=True)

# Create syncer (registers hooks)
syncer = OverlappedGradSync(dense_params, process_group, num_buckets=4)

# Training loop
for batch in data:
    loss = model(batch).loss
    loss.backward()       # hooks fire allreduces during backward
    syncer.finish()       # wait + unpack
    optimizer.step()
    optimizer.zero_grad()

# Cleanup
syncer.remove_hooks()
```

`get_dense_params()` automatically detects expert parameters (from `MoELayer.expert_params` or by name convention) and excludes them.
