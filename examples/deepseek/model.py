"""DeepSeek-V3 style transformer at ~500M scale.

Key features adapted from DeepSeek-V3:
- Fine-grained MoE with shared expert (always-active)
- Sigmoid gating with normalized top-k weights
- First-k dense layers (rest MoE)
- RoPE position embeddings
- RMSNorm pre-norm
- SwiGLU activation

Two configs:
- DENSE_500M: ~475M params, standard dense FFN
- MOE_500M: ~475M active / ~1.1B total, MoE FFN with shared expert
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfusemoe.fused_moe import fused_moe, set_ep_group
from flashfusemoe.expert_parallel import ExpertParallelDispatcher


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DeepSeekConfig:
    vocab_size: int = 50304       # GPT-2 rounded to 64
    d_model: int = 1024
    n_heads: int = 16
    n_layers: int = 24
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # Dense FFN (used for dense model and first_k_dense layers)
    dense_ffn_dim: int = 8192     # gate+up combined; intermediate = 4096

    # MoE
    use_moe: bool = False
    first_k_dense: int = 1        # first N layers use dense FFN
    n_routed_experts: int = 8
    n_shared_experts: int = 1
    moe_ffn_dim: int = 3072       # per expert; intermediate = 1536
    top_k: int = 2
    scoring_func: str = "softmax"  # "softmax" or "sigmoid"
    moe_backend: str = "unfused"   # "unfused" (PyTorch loops) or "fused" (Triton kernels)
    max_tokens_per_expert: int | None = None  # Buffer size per expert; None = auto
    allow_dropped_tokens: bool = True  # True for training (accept heuristic buffer)

    # Expert parallelism
    ep_group: object = None  # torch.distributed ProcessGroup (None = single GPU)
    ep_size: int = 1         # number of GPUs for expert parallelism
    moe_latent_dim: int | None = None  # Latent dim for A2A compression (None = no projection)
    ep_backend: str = "nccl"  # EP comm backend: "nccl", "deep_ep", "hybrid"

    # Multi-Token Prediction (MTP)
    mtp_num_layers: int = 0       # number of MTP prediction heads (0 = disabled)
    mtp_loss_factor: float = 0.1  # weight of MTP loss relative to main CE loss


# Pre-built configs
DENSE_500M = DeepSeekConfig(use_moe=False)

MOE_UNFUSED = DeepSeekConfig(
    use_moe=True,
    first_k_dense=1,
    n_routed_experts=8,
    n_shared_experts=1,
    moe_ffn_dim=3072,
    top_k=2,
    scoring_func="softmax",
    moe_backend="unfused",
)

MOE_FUSED = DeepSeekConfig(
    use_moe=True,
    first_k_dense=1,
    n_routed_experts=8,
    n_shared_experts=1,
    moe_ffn_dim=3072,
    top_k=2,
    scoring_func="softmax",
    moe_backend="fused",
)

MOE_FUSED_EP = DeepSeekConfig(
    use_moe=True,
    first_k_dense=1,
    n_routed_experts=8,
    n_shared_experts=1,
    moe_ffn_dim=3072,
    top_k=2,
    scoring_func="softmax",
    moe_backend="fused",
    # ep_group and ep_size set at runtime by shard_experts()
)


def count_params(model):
    """Count total and active parameters."""
    total = sum(p.numel() for p in model.parameters())
    return total



# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() / rms * self.weight.float()).to(x.dtype)


def precompute_rope_freqs(dim, max_seq_len, theta=10000.0, device=None):
    """Precompute RoPE frequency tensor."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(max_seq_len, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x, freqs_cis):
    """Apply rotary embeddings. x: [B, n_heads, T, head_dim]"""
    T = x.shape[2]
    freqs = freqs_cis[:T].unsqueeze(0).unsqueeze(0)  # [1, 1, T, head_dim/2]
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(*x.shape)
    return x_rotated.type_as(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, x, freqs_cis):
        B, T, D = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)  # [B, n_heads, T, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope(q, freqs_cis)
        k = apply_rope(k, freqs_cis)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """Standard SwiGLU FFN: gate+up combined, then down."""
    def __init__(self, d_model, ffn_dim):
        super().__init__()
        intermediate = ffn_dim // 2
        self.w_gate = nn.Linear(d_model, intermediate, bias=False)
        self.w_up = nn.Linear(d_model, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class DeepSeekMoEFFN(nn.Module):
    """DeepSeek-V3 style MoE FFN with shared expert + routed experts.

    Supports two backends:
    - "unfused": PyTorch loop over experts (reference implementation)
    - "fused": Triton fused routing + grouped GEMM kernels

    When ep_group is set, stores only local expert weights (E_local = E / ep_size).
    Gate weight is always replicated (all GPUs need full routing).
    """
    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.d_model = config.d_model
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.top_k
        self.moe_ffn_dim = config.moe_ffn_dim
        self.half_ffn = config.moe_ffn_dim // 2
        self.scoring_func = config.scoring_func
        self.backend = config.moe_backend
        self.max_tokens_per_expert = config.max_tokens_per_expert
        self.allow_dropped_tokens = config.allow_dropped_tokens
        self.ep_group = config.ep_group
        self.ep_size = config.ep_size

        # Shared expert (always active, replicated)
        if config.n_shared_experts > 0:
            self.shared_expert = SwiGLUFFN(config.d_model, config.moe_ffn_dim)
        else:
            self.shared_expert = None

        # Gate weight: [E, D] always replicated (all GPUs need full routing)
        self.gate = nn.Linear(config.d_model, config.n_routed_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        self.gate.weight.data *= 0.01

        # Expert weights: [E, ffn_dim, D] (gate+up stacked)
        # When EP is active, these are initialized as full size;
        # shard_experts() will replace them with local slices
        self.w_up = nn.Parameter(torch.empty(config.n_routed_experts, config.moe_ffn_dim, config.d_model))
        self.w_down = nn.Parameter(torch.empty(config.n_routed_experts, config.d_model, self.half_ffn))
        self._init_expert_weights()

        # Latent dimension projection for A2A compression (Megatron pattern)
        # Projects D→latent before dispatch, latent→D after combine, reducing A2A volume
        self.latent_dim = config.moe_latent_dim
        if self.latent_dim is not None:
            self.proj_down = nn.Linear(config.d_model, self.latent_dim, bias=False)
            self.proj_up = nn.Linear(self.latent_dim, config.d_model, bias=False)

    def _init_expert_weights(self):
        for e in range(self.n_routed_experts):
            nn.init.kaiming_uniform_(self.w_up.data[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_down.data[e], a=math.sqrt(5))

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(B * T, D)

        # Routing + expert computation
        if self.backend == "fused":
            # When EP is active and shared expert exists, pass shared expert
            # weights so the autograd function can overlap shared expert compute
            # with the dispatch all-to-all.
            shared_weights = None
            if self.ep_group is not None and self.shared_expert is not None:
                shared_weights = (
                    self.shared_expert.w_gate.weight,
                    self.shared_expert.w_up.weight,
                    self.shared_expert.w_down.weight,
                )
            latent_weights = None
            if self.latent_dim is not None:
                latent_weights = (self.proj_down.weight, self.proj_up.weight)
            routed_out, _losses = fused_moe(
                x_flat, self.gate.weight, self.w_up, self.w_down, self.top_k,
                gating=self.scoring_func,
                max_tokens_per_expert=self.max_tokens_per_expert,
                allow_dropped_tokens=self.allow_dropped_tokens,
                shared_expert_weights=shared_weights,
                latent_weights=latent_weights,
            )
            # If shared expert was computed inside fused_moe (EP path),
            # skip separate computation. Otherwise compute here.
            if shared_weights is not None:
                return routed_out.view(B, T, D)
        elif self.ep_group is not None:
            routed_out = self._route_and_compute_ep(x_flat)
        else:
            routed_out = self._route_and_compute(x_flat)

        # Shared expert (computed separately for non-EP or unfused paths)
        shared_out = self.shared_expert(x_flat) if self.shared_expert is not None else 0

        out = shared_out + routed_out
        if isinstance(out, (int, float)):
            out = routed_out
        return out.view(B, T, D)

    def _route_and_compute(self, x):
        """Route tokens to experts and compute weighted output (PyTorch loop)."""
        N, D = x.shape

        logits = x @ self.gate.weight.T  # [N, E]

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
            _, topk_indices = torch.topk(scores, self.top_k, dim=-1)
            topk_weights = scores.gather(1, topk_indices)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        topk_weights = topk_weights.to(x.dtype)

        # Flatten top-k: [N*top_k] expert ids, weights, token ids
        flat_expert_ids = topk_indices.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        flat_token_ids = torch.arange(N, device=x.device).repeat_interleave(self.top_k)

        output = torch.zeros(N, D, dtype=x.dtype, device=x.device)
        for e in range(self.n_routed_experts):
            mask = flat_expert_ids == e
            if not mask.any():
                continue
            token_ids = flat_token_ids[mask]
            w = flat_weights[mask].unsqueeze(-1)
            tokens = x[token_ids]
            gate_out = tokens @ self.w_up[e, :self.half_ffn, :].T
            up_out = tokens @ self.w_up[e, self.half_ffn:, :].T
            down_out = (F.silu(gate_out) * up_out) @ self.w_down[e].T
            output.index_add_(0, token_ids, w * down_out)

        return output

    def _route_and_compute_ep(self, x):
        """Route tokens to experts across GPUs and compute weighted output (EP + PyTorch loop).

        All GPUs compute identical routing (replicated gate_weight).
        Tokens dispatched via all-to-all to expert-owning GPU.
        Each GPU loops over its local experts only.
        Results combined via all-to-all back.
        """
        N, D = x.shape
        E = self.n_routed_experts
        E_local = self.w_up.shape[0]  # sharded after shard_experts()

        # Replicated routing (identical on all GPUs)
        logits = x @ self.gate.weight.T  # [N, E]

        if self.scoring_func == "sigmoid":
            scores = torch.sigmoid(logits)
            _, topk_indices = torch.topk(scores, self.top_k, dim=-1)
            topk_weights = scores.gather(1, topk_indices)
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-6)
        else:
            topk_logits, topk_indices = torch.topk(logits, self.top_k, dim=-1)
            topk_weights = F.softmax(topk_logits, dim=-1, dtype=torch.float32)

        topk_weights = topk_weights.to(x.dtype)

        # Dispatch tokens to expert-owning GPUs
        dispatcher = ExpertParallelDispatcher(E, self.ep_size, self.ep_group)
        received_tokens, metadata = dispatcher.dispatch(
            x, topk_indices, topk_weights,
        )

        local_expert_ids = metadata["local_expert_ids"]  # [M], values in [0, E_local)
        received_weights = metadata["received_weights"]   # [M]
        M = received_tokens.shape[0]

        # Compute on local experts (PyTorch loop)
        local_output = torch.zeros(M, D, dtype=x.dtype, device=x.device)
        for e in range(E_local):
            mask = local_expert_ids == e
            if not mask.any():
                continue
            tokens = received_tokens[mask]  # [n_e, D]
            w = received_weights[mask].unsqueeze(-1)  # [n_e, 1]

            gate_out = tokens @ self.w_up[e, :self.half_ffn, :].T
            up_out = tokens @ self.w_up[e, self.half_ffn:, :].T
            down_out = (F.silu(gate_out) * up_out) @ self.w_down[e].T
            local_output[mask] = w * down_out

        # Combine: send results back to originating GPUs
        output = dispatcher.combine(local_output, metadata)  # [N, D]
        return output


class DeepSeekBlock(nn.Module):
    def __init__(self, config: DeepSeekConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # Use MoE or dense FFN based on layer index
        use_moe = config.use_moe and layer_idx >= config.first_k_dense
        if use_moe:
            self.ffn = DeepSeekMoEFFN(config)
        else:
            self.ffn = SwiGLUFFN(config.d_model, config.dense_ffn_dim)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MTPModule(nn.Module):
    """Multi-Token Prediction module (DeepSeek-V3 style).

    Takes the previous hidden state and the embedding of the future token,
    projects them together, runs through a single transformer block, and
    predicts the next-next token.

    For K MTP modules chained:
      MTP-1 predicts token t+2 given hidden(t+1) + embed(t+2)
      MTP-2 predicts token t+3 given hidden_mtp1(t+2) + embed(t+3)
      ...

    At inference time, MTP enables speculative decoding: generate K+1 tokens
    in one forward pass by using MTP heads as draft predictors.
    """

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        D = config.d_model
        self.proj = nn.Linear(2 * D, D, bias=False)
        self.norm = RMSNorm(D, config.rms_norm_eps)
        self.block = DeepSeekBlock(config, config.n_layers)  # Use next layer index
        self.head = nn.Linear(D, config.vocab_size, bias=False)
        nn.init.normal_(self.proj.weight, std=0.02)

    def forward(self, prev_hidden, future_embed, freqs_cis):
        """
        Args:
            prev_hidden: [B, T, D] hidden from previous layer/MTP module
            future_embed: [B, T, D] embedding of the token we're predicting
            freqs_cis: RoPE frequencies

        Returns:
            logits: [B, T, V] prediction logits
            hidden: [B, T, D] hidden state for next MTP module
        """
        # Concatenate and project: [B, T, 2D] → [B, T, D]
        combined = torch.cat([prev_hidden, future_embed], dim=-1)
        x = self.proj(combined)
        x = self.norm(x)
        x = self.block(x, freqs_cis)
        logits = self.head(x)
        return logits, x


class DeepSeekTransformer(nn.Module):
    """DeepSeek-V3 style transformer at ~500M scale."""

    def __init__(self, config: DeepSeekConfig):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            DeepSeekBlock(config, i) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Tie embeddings
        self.lm_head.weight = self.tok_emb.weight

        # Multi-Token Prediction modules
        self.mtp_modules = nn.ModuleList([
            MTPModule(config) for _ in range(config.mtp_num_layers)
        ]) if config.mtp_num_layers > 0 else None
        self.mtp_loss_factor = config.mtp_loss_factor

        # Precompute RoPE
        head_dim = config.d_model // config.n_heads
        self.register_buffer(
            'freqs_cis',
            precompute_rope_freqs(head_dim, config.max_seq_len, config.rope_theta),
            persistent=False,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, std=0.02)

    def forward(self, input_ids, targets=None):
        """input_ids: [B, T] -> logits: [B, T, V] (+ MTP losses if targets provided)

        When targets is provided AND mtp_modules exist:
          Returns (logits, mtp_losses_dict) where mtp_losses_dict contains
          per-MTP-head CE losses and the combined weighted MTP loss.

        When targets is None or no MTP modules:
          Returns logits only (backward compatible).
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)
        freqs_cis = self.freqs_cis[:T]

        for layer in self.layers:
            x = layer(x, freqs_cis)

        hidden = self.norm(x)
        logits = self.lm_head(hidden)

        # Multi-Token Prediction
        if self.mtp_modules is not None and targets is not None and T > 1:
            mtp_losses = {}
            total_mtp_loss = 0.0
            prev_hidden = hidden

            for k, mtp in enumerate(self.mtp_modules):
                # For MTP-k, we predict token at position t+k+2
                # We need: prev_hidden[:, :-1, :] and embed of targets[:, k+1:]
                shift = k + 1
                if T - shift < 1:
                    break

                # Future token embeddings: embed(targets[:, shift:-1]) if available
                # targets is [B, T] with targets[t] = next token for position t
                # MTP-k predicts targets[:, shift:] using hidden[:, :-shift]
                if shift >= T:
                    break

                future_tokens = targets[:, shift:]  # [B, T-shift]
                future_embed = self.tok_emb(future_tokens)  # [B, T-shift, D]

                # Truncate prev_hidden to match
                prev_hidden_trunc = prev_hidden[:, :T - shift, :]  # [B, T-shift, D]
                freqs_trunc = self.freqs_cis[:T - shift]

                mtp_logits, mtp_hidden = mtp(prev_hidden_trunc, future_embed, freqs_trunc)

                # CE loss for this MTP head: predict targets[:, shift+1:]
                # But we only have T-shift positions, predicting the token AFTER each
                if shift + 1 < T:
                    mtp_targets = targets[:, shift + 1:]  # [B, T-shift-1]
                    # Trim mtp_logits to match: positions [0, T-shift-2] predict targets[shift+1:]
                    mtp_logits_trim = mtp_logits[:, :-1, :]  # [B, T-shift-1, V]
                    if mtp_targets.shape[1] > 0:
                        loss_k = F.cross_entropy(
                            mtp_logits_trim.reshape(-1, mtp_logits_trim.size(-1)),
                            mtp_targets.reshape(-1),
                        )
                        mtp_losses[f"mtp_loss_{k}"] = loss_k
                        total_mtp_loss = total_mtp_loss + loss_k

                # Update prev_hidden for next MTP module (use normalized mtp_hidden)
                prev_hidden = mtp_hidden

            if len(mtp_losses) > 0:
                mtp_losses["mtp_loss"] = self.mtp_loss_factor * total_mtp_loss
            return logits, mtp_losses

        return logits


def create_2d_process_groups(world_size, ep_size):
    """Create 2D process group mesh for FSDP + EP.

    Splits world_size ranks into:
      - ep_groups: groups of ep_size ranks for expert parallelism
      - dp_groups: groups of (world_size // ep_size) ranks for data parallelism

    Example with world_size=8, ep_size=4:
      EP groups: [0,1,2,3], [4,5,6,7]
      DP groups: [0,4], [1,5], [2,6], [3,7]

    Returns:
        ep_group: ProcessGroup for this rank's EP group
        dp_group: ProcessGroup for this rank's DP group
    """
    import torch.distributed as dist

    assert world_size % ep_size == 0, \
        f"world_size ({world_size}) must be divisible by ep_size ({ep_size})"
    dp_size = world_size // ep_size
    rank = dist.get_rank()

    # EP groups: contiguous ranks
    ep_group = None
    for i in range(dp_size):
        ranks = list(range(i * ep_size, (i + 1) * ep_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            ep_group = group

    # DP groups: strided ranks
    dp_group = None
    for i in range(ep_size):
        ranks = list(range(i, world_size, ep_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            dp_group = group

    return ep_group, dp_group


def wrap_with_fsdp(model, dp_group):
    """Wrap model with FSDP for dense parameter sharding across DP group.

    Expert weights (w_up, w_down in DeepSeekMoEFFN) are already sharded via EP
    and should NOT be FSDP-sharded. We wrap each non-MoE layer with FSDP.

    Args:
        model: DeepSeekTransformer (already expert-sharded via shard_experts)
        dp_group: ProcessGroup for data parallelism

    Returns:
        FSDP-wrapped model
    """
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy

    # Wrap entire model with FSDP, but only shard dense params.
    # Expert params won't be re-sharded because they're already local slices.
    model = FSDP(
        model,
        process_group=dp_group,
        auto_wrap_policy=ModuleWrapPolicy({DeepSeekBlock}),
        use_orig_params=True,
    )

    return model


def shard_experts(model, ep_group, ep_backend="nccl"):
    """Shard expert weights across GPUs for expert parallelism.

    Takes a full model (with all experts on each GPU) and replaces expert
    weights with local slices. Gate weights and shared experts stay replicated.

    Args:
        model: DeepSeekTransformer with full expert weights
        ep_group: torch.distributed ProcessGroup
        ep_backend: EP comm backend ("nccl", "deep_ep", "hybrid")

    Returns:
        model with sharded expert weights (modified in-place)
    """
    rank = torch.distributed.get_rank(ep_group)
    ep_size = torch.distributed.get_world_size(ep_group)

    for layer in model.layers:
        ffn = layer.ffn
        if not isinstance(ffn, DeepSeekMoEFFN):
            continue

        E = ffn.n_routed_experts
        assert E % ep_size == 0, f"n_routed_experts ({E}) must be divisible by ep_size ({ep_size})"
        E_local = E // ep_size
        start = rank * E_local
        end = start + E_local

        # Slice and replace expert weights with local portion
        ffn.w_up = nn.Parameter(ffn.w_up.data[start:end].contiguous())
        ffn.w_down = nn.Parameter(ffn.w_down.data[start:end].contiguous())

        # Set EP group on the layer
        ffn.ep_group = ep_group
        ffn.ep_size = ep_size

    # Set module-level EP group for compile-compatible fused_moe path.
    # This is called once at init, so no graph break during forward().
    set_ep_group(ep_group, backend=ep_backend)

    return model
