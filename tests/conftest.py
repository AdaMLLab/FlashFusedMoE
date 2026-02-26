"""Shared fixtures and reference implementations for FlashFuseMoE tests."""

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(autouse=True)
def cuda_device():
    """Ensure CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def seed():
    """Set deterministic seed."""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    return 42


def make_moe_weights(N, D, ffn_dim, E, dtype=torch.float16, device="cuda"):
    """Create random MoE weights.

    Returns:
        gate_weight: [E, D]
        w_up: [E, ffn_dim, D]  (gate+up stacked: first half gate, second half up)
        w_down: [E, D, ffn_dim//2]
    """
    gate_weight = torch.randn(E, D, dtype=dtype, device=device) * 0.01
    w_up = torch.randn(E, ffn_dim, D, dtype=dtype, device=device) * (2.0 / D) ** 0.5
    w_down = torch.randn(E, D, ffn_dim // 2, dtype=dtype, device=device) * (2.0 / (ffn_dim // 2)) ** 0.5
    return gate_weight, w_up, w_down


# Standard test configurations: (N, D, ffn_dim, E, top_k)
SMALL_CONFIGS = [
    (32, 128, 256, 4, 2),
    (64, 256, 512, 8, 2),
    (128, 512, 1024, 8, 2),
]

MEDIUM_CONFIGS = [
    (256, 1024, 2048, 8, 2),
    (512, 2048, 5632, 16, 2),
]

LARGE_CONFIGS = [
    (1024, 4096, 14336, 8, 2),
]

ALL_CONFIGS = SMALL_CONFIGS + MEDIUM_CONFIGS + LARGE_CONFIGS

_WEIGHT_EPS = 1e-6


def unfused_moe(hidden_states, gate_weight, w_up, w_down, top_k,
                renormalize=True, activation="swiglu", gating="softmax",
                capacity_factor=None, aux_loss_coeff=0.0, z_loss_coeff=0.0):
    """Pure PyTorch unfused MoE -- correctness oracle for tests."""
    N, D = hidden_states.shape
    E = gate_weight.shape[0]
    ffn_dim = w_up.shape[1]
    half_ffn = ffn_dim // 2
    device = hidden_states.device
    losses = {}

    logits = hidden_states @ gate_weight.T

    if gating == "softmax":
        topk_weights, topk_indices = torch.topk(logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32)
        if not renormalize:
            all_weights = F.softmax(logits, dim=-1, dtype=torch.float32)
            topk_weights = all_weights.gather(1, topk_indices)
    elif gating == "sigmoid":
        scores = torch.sigmoid(logits.float())
        _, topk_indices = torch.topk(scores, top_k, dim=-1)
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + _WEIGHT_EPS)

    flat_expert_ids = topk_indices.view(-1)
    tokens_per_expert = torch.bincount(flat_expert_ids, minlength=E).float()

    if aux_loss_coeff > 0.0:
        f = tokens_per_expert / N
        if gating == "softmax":
            P = torch.zeros(E, device=device, dtype=torch.float32)
            P.scatter_add_(0, flat_expert_ids, topk_weights.view(-1).float())
            P = P / N
        elif gating == "sigmoid":
            scores_full = torch.sigmoid(logits.float())
            P = scores_full.mean(dim=0)
        losses["aux_loss"] = aux_loss_coeff * E * (f * P).sum()

    if z_loss_coeff > 0.0:
        losses["z_loss"] = z_loss_coeff * torch.logsumexp(logits.float(), dim=-1).square().mean()

    if capacity_factor is not None:
        expert_capacity = max(int((N / E) * capacity_factor), top_k)
        sorted_order = torch.argsort(flat_expert_ids, stable=True)
        expert_starts = torch.zeros(E + 1, dtype=torch.long, device=device)
        expert_starts[1:] = torch.cumsum(tokens_per_expert.long(), dim=0)
        rank_within_expert = torch.zeros_like(flat_expert_ids)
        sorted_expert_ids = flat_expert_ids[sorted_order]
        sorted_ranks = torch.arange(len(sorted_order), device=device)
        sorted_ranks = sorted_ranks - expert_starts[sorted_expert_ids]
        rank_within_expert[sorted_order] = sorted_ranks
        keep_mask = rank_within_expert < expert_capacity
        topk_weights = topk_weights * keep_mask.view(N, top_k).float()
        weight_sum = topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights / (weight_sum + _WEIGHT_EPS)

    output = torch.zeros(N, D, dtype=hidden_states.dtype, device=device)
    for k_idx in range(top_k):
        expert_ids = topk_indices[:, k_idx]
        weights = topk_weights[:, k_idx]
        for e in range(E):
            mask = expert_ids == e
            if not mask.any():
                continue
            tokens = hidden_states[mask]
            w_gate_e = w_up[e, :half_ffn, :]
            w_up_e = w_up[e, half_ffn:, :]
            gate_out = tokens @ w_gate_e.T
            up_out = tokens @ w_up_e.T
            if activation == "swiglu":
                activated = F.silu(gate_out) * up_out
            elif activation == "geglu":
                activated = F.gelu(gate_out, approximate="tanh") * up_out
            elif activation == "relu_squared":
                activated = F.relu(gate_out).square() * up_out
            down_out = activated @ w_down[e].T
            output[mask] += weights[mask].unsqueeze(-1).to(down_out.dtype) * down_out

    return output, topk_indices, topk_weights, losses
