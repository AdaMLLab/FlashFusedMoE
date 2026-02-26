"""MoELayer: drop-in nn.Module wrapping fused_moe().

Model-agnostic MoE layer that replaces an FFN block. Supports expert
parallelism, shared experts, latent projection, and torch.compile.

Usage:
    from flashfusemoe import MoELayer

    moe = MoELayer(d_model=1024, ffn_dim=4096, num_experts=8, top_k=2)
    output, losses = moe(x)  # x: [B, T, D]
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from flashfusemoe.fused_moe import fused_moe, set_ep_group


class _GLUFFN(nn.Module):
    """GLU FFN for shared experts. Supports swiglu/geglu/relu_squared."""

    def __init__(self, d_model, ffn_dim, activation="swiglu"):
        super().__init__()
        intermediate = ffn_dim // 2
        self.w_gate = nn.Linear(d_model, intermediate, bias=False)
        self.w_up = nn.Linear(d_model, intermediate, bias=False)
        self.w_down = nn.Linear(intermediate, d_model, bias=False)
        self.activation = activation

    def forward(self, x):
        gate_out = self.w_gate(x)
        up_out = self.w_up(x)
        if self.activation == "swiglu":
            activated = F.silu(gate_out) * up_out
        elif self.activation == "geglu":
            activated = F.gelu(gate_out) * up_out
        elif self.activation == "relu_squared":
            activated = F.relu(gate_out).square() * up_out
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
        return self.w_down(activated)


class MoELayer(nn.Module):
    """Fused Mixture-of-Experts layer.

    Drop-in replacement for an FFN block. Supports expert parallelism,
    shared experts, latent projection, and torch.compile.

    Returns (output, losses) where losses is a dict (empty if no aux/z loss).
    """

    def __init__(
        self,
        d_model: int,
        ffn_dim: int,
        num_experts: int,
        top_k: int = 2,
        activation: str = "swiglu",
        gating: str = "softmax",
        num_shared_experts: int = 0,
        latent_dim: int | None = None,
        capacity_factor: float | None = None,
        max_tokens_per_expert: int | None = None,
        allow_dropped_tokens: bool = True,
        aux_loss_coeff: float = 0.0,
        z_loss_coeff: float = 0.0,
        recompute_activations: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.activation = activation
        self.gating = gating
        self.half_ffn = ffn_dim // 2
        self.max_tokens_per_expert = max_tokens_per_expert
        self.allow_dropped_tokens = allow_dropped_tokens
        self.aux_loss_coeff = aux_loss_coeff
        self.z_loss_coeff = z_loss_coeff
        self.capacity_factor = capacity_factor
        self.recompute_activations = recompute_activations

        # EP state (set by shard_experts())
        self._ep_group = None
        self._ep_size = 1

        # Gate weight: [E, D] always replicated
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))
        self.gate.weight.data *= 0.01

        # Expert weights: [E, ffn_dim, D] (gate+up stacked) and [E, D, ffn_dim//2]
        self.w_up = nn.Parameter(torch.empty(num_experts, ffn_dim, d_model))
        self.w_down = nn.Parameter(torch.empty(num_experts, d_model, self.half_ffn))
        self._init_expert_weights()

        # Shared expert(s)
        if num_shared_experts > 0:
            self.shared_expert = _GLUFFN(d_model, ffn_dim, activation)
        else:
            self.shared_expert = None

        # Latent projection for A2A compression
        self.latent_dim = latent_dim
        if latent_dim is not None:
            self.proj_down = nn.Linear(d_model, latent_dim, bias=False)
            self.proj_up = nn.Linear(latent_dim, d_model, bias=False)

    def _init_expert_weights(self):
        for e in range(self.w_up.shape[0]):
            nn.init.kaiming_uniform_(self.w_up.data[e], a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.w_down.data[e], a=math.sqrt(5))

    def forward(self, x):
        """x: [*, D] -> ([*, D], dict)"""
        leading = x.shape[:-1]
        D = x.shape[-1]
        x_flat = x.reshape(-1, D)

        # Build shared expert weights tuple for EP overlap
        shared_weights = None
        if self._ep_group is not None and self.shared_expert is not None:
            shared_weights = (
                self.shared_expert.w_gate.weight,
                self.shared_expert.w_up.weight,
                self.shared_expert.w_down.weight,
            )

        latent_weights = None
        if self.latent_dim is not None:
            latent_weights = (self.proj_down.weight, self.proj_up.weight)

        routed_out, losses = fused_moe(
            x_flat, self.gate.weight, self.w_up, self.w_down, self.top_k,
            activation=self.activation,
            gating=self.gating,
            capacity_factor=self.capacity_factor,
            aux_loss_coeff=self.aux_loss_coeff,
            z_loss_coeff=self.z_loss_coeff,
            recompute_activations=self.recompute_activations,
            max_tokens_per_expert=self.max_tokens_per_expert,
            allow_dropped_tokens=self.allow_dropped_tokens,
            shared_expert_weights=shared_weights,
            latent_weights=latent_weights,
        )

        # If shared expert was computed inside fused_moe (EP path), skip separate computation
        if shared_weights is not None:
            return routed_out.reshape(*leading, D), losses

        # Shared expert (computed separately for non-EP path)
        if self.shared_expert is not None:
            shared_out = self.shared_expert(x_flat)
            routed_out = routed_out + shared_out

        return routed_out.reshape(*leading, D), losses

    def shard_experts(self, ep_group, ep_backend="nccl"):
        """Shard expert weights for EP. Call before optimizer creation."""
        from flashfusemoe.distributed import shard_expert_weights
        self.w_up, self.w_down = shard_expert_weights(
            self.w_up, self.w_down, ep_group,
        )
        self._ep_group = ep_group
        self._ep_size = torch.distributed.get_world_size(ep_group)
        set_ep_group(ep_group, backend=ep_backend)

    @property
    def expert_params(self):
        """Params sharded across EP ranks (skip in dense grad sync)."""
        return [self.w_up, self.w_down]
