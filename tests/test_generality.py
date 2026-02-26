"""Session 11: Generality tests — large D, multiple activations, sigmoid gating.

Tests fused_moe against unfused_moe reference in fp64 for correctness across
all parameter combinations.
"""

import pytest
import torch

from flashfusemoe.fused_moe import fused_moe
from conftest import unfused_moe


def _make_inputs(N, D, ffn_dim, E, dtype, device="cuda"):
    """Create random MoE inputs."""
    hidden = torch.randn(N, D, dtype=dtype, device=device)
    gate_w = torch.randn(E, D, dtype=dtype, device=device) * 0.01
    w_up = torch.randn(E, ffn_dim, D, dtype=dtype, device=device) * (2.0 / D) ** 0.5
    w_down = torch.randn(E, D, ffn_dim // 2, dtype=dtype, device=device) * (2.0 / (ffn_dim // 2)) ** 0.5
    return hidden, gate_w, w_up, w_down


# ---------------------------------------------------------------------------
# Forward correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", [64, 128, 256, 512, 1024, 2048, 4096, 7168])
@pytest.mark.parametrize("activation", ["swiglu", "geglu", "relu_squared"])
@pytest.mark.parametrize("gating", ["softmax", "sigmoid"])
def test_forward_correctness(D, activation, gating):
    """Fused output matches unfused reference in fp64."""
    torch.manual_seed(42)
    N, E, top_k = 64, 8, 2
    ffn_dim = D * 3
    # Ensure ffn_dim is even
    if ffn_dim % 2 != 0:
        ffn_dim += 1

    hidden, gate_w, w_up, w_down = _make_inputs(N, D, ffn_dim, E, torch.float64)

    out_fused, _ = fused_moe(hidden, gate_w, w_up, w_down, top_k,
                              activation=activation, gating=gating)
    out_ref, _, _, _ = unfused_moe(hidden, gate_w, w_up, w_down, top_k,
                                    activation=activation, gating=gating)

    rel_err = ((out_fused - out_ref).float().norm() /
               (out_ref.float().norm() + 1e-12)).item()
    assert rel_err < 1e-5, (
        f"Forward mismatch: rel_err={rel_err:.2e} "
        f"(D={D}, act={activation}, gating={gating})"
    )


# ---------------------------------------------------------------------------
# Gradient correctness
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation", ["swiglu", "geglu", "relu_squared"])
@pytest.mark.parametrize("gating", ["softmax", "sigmoid"])
def test_gradient_correctness(activation, gating):
    """Compare analytical gradients (fused) vs PyTorch autograd (unfused reference)."""
    torch.manual_seed(42)
    N, D, ffn_dim, E, top_k = 16, 64, 128, 4, 2
    device = "cuda"
    dtype = torch.float64

    hidden_data = torch.randn(N, D, dtype=dtype, device=device)
    gate_data = torch.randn(E, D, dtype=dtype, device=device) * 0.1
    w_up_data = torch.randn(E, ffn_dim, D, dtype=dtype, device=device) * 0.1
    w_down_data = torch.randn(E, D, ffn_dim // 2, dtype=dtype, device=device) * 0.1
    grad_out = torch.randn(N, D, dtype=dtype, device=device)

    # --- Fused backward ---
    h1 = hidden_data.clone().requires_grad_(True)
    g1 = gate_data.clone().requires_grad_(True)
    wu1 = w_up_data.clone().requires_grad_(True)
    wd1 = w_down_data.clone().requires_grad_(True)
    out1, _ = fused_moe(h1, g1, wu1, wd1, top_k, activation=activation, gating=gating)
    out1.backward(grad_out)

    # --- Reference (unfused) with PyTorch autograd ---
    h2 = hidden_data.clone().requires_grad_(True)
    g2 = gate_data.clone().requires_grad_(True)
    wu2 = w_up_data.clone().requires_grad_(True)
    wd2 = w_down_data.clone().requires_grad_(True)
    out2, _, _, _ = unfused_moe(h2, g2, wu2, wd2, top_k,
                                activation=activation, gating=gating)
    out2.backward(grad_out)

    # Forward match (different computation paths = small numerical diff)
    torch.testing.assert_close(out1, out2, atol=1e-6, rtol=1e-5)

    # Gradient comparisons
    torch.testing.assert_close(wu1.grad, wu2.grad, atol=1e-5, rtol=1e-4,
                               msg=f"w_up gradient mismatch (act={activation}, gating={gating})")
    torch.testing.assert_close(wd1.grad, wd2.grad, atol=1e-5, rtol=1e-4,
                               msg=f"w_down gradient mismatch (act={activation}, gating={gating})")


# ---------------------------------------------------------------------------
# Large D test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("D", [4096, 7168, 8192])
def test_large_D(D):
    """Large D values work in bf16 without NaN/Inf."""
    torch.manual_seed(42)
    N, E, top_k = 32, 8, 2
    ffn_dim = D * 3
    if ffn_dim % 2 != 0:
        ffn_dim += 1

    hidden, gate_w, w_up, w_down = _make_inputs(N, D, ffn_dim, E, torch.bfloat16)
    hidden.requires_grad_(True)

    out, _ = fused_moe(hidden, gate_w, w_up, w_down, top_k)
    assert torch.isfinite(out).all(), f"Forward has NaN/Inf for D={D}"

    loss = out.sum()
    loss.backward()


# ---------------------------------------------------------------------------
# Various E and top_k
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("E,top_k", [(4, 1), (8, 2), (16, 4), (64, 8)])
def test_expert_configs(E, top_k):
    """Various expert/top_k combos produce correct results in fp64."""
    torch.manual_seed(42)
    N, D, ffn_dim = 128, 512, 1024

    hidden, gate_w, w_up, w_down = _make_inputs(N, D, ffn_dim, E, torch.float64)

    out_fused, _ = fused_moe(hidden, gate_w, w_up, w_down, top_k)
    out_ref, _, _, _ = unfused_moe(hidden, gate_w, w_up, w_down, top_k)

    rel_err = ((out_fused - out_ref).float().norm() /
               (out_ref.float().norm() + 1e-12)).item()
    assert rel_err < 1e-5, f"Forward mismatch: rel_err={rel_err:.2e} (E={E}, top_k={top_k})"


# ---------------------------------------------------------------------------
# BF16 stability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("activation", ["swiglu", "geglu", "relu_squared"])
def test_bf16_no_nan(activation):
    """BF16 via autocast forward+backward produces no NaN/Inf."""
    torch.manual_seed(42)
    N, D, ffn_dim, E, top_k = 256, 1024, 2048, 8, 2

    # fp32 inputs + autocast (how bf16 is used in practice)
    hidden = torch.randn(N, D, dtype=torch.float32, device="cuda") * 0.1
    gate_w = torch.randn(E, D, dtype=torch.float32, device="cuda") * 0.01
    w_up = torch.randn(E, ffn_dim, D, dtype=torch.float32, device="cuda") * 0.05
    w_down = torch.randn(E, D, ffn_dim // 2, dtype=torch.float32, device="cuda") * 0.05
    hidden.requires_grad_(True)
    w_up.requires_grad_(True)
    w_down.requires_grad_(True)

    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        out, _ = fused_moe(hidden, gate_w, w_up, w_down, top_k, activation=activation)

    assert torch.isfinite(out).all(), f"Forward NaN/Inf with {activation}"

    loss = out.float().sum()
    loss.backward()

    assert torch.isfinite(hidden.grad).all(), f"hidden grad NaN/Inf with {activation}"
    assert torch.isfinite(w_up.grad).all(), f"w_up grad NaN/Inf with {activation}"
    assert torch.isfinite(w_down.grad).all(), f"w_down grad NaN/Inf with {activation}"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_validation_bad_shapes():
    """Wrong dimensions raise AssertionError with helpful message."""
    device = "cuda"
    dtype = torch.float32
    N, D, ffn_dim, E, top_k = 16, 64, 128, 4, 2

    hidden, gate_w, w_up, w_down = _make_inputs(N, D, ffn_dim, E, dtype)

    # 3D hidden_states
    with pytest.raises(AssertionError, match="hidden_states must be 2D"):
        fused_moe(hidden.unsqueeze(0), gate_w, w_up, w_down, top_k)

    # Wrong gate_weight shape (wrong D dimension)
    with pytest.raises(AssertionError, match="gate_weight must be"):
        bad_gate = torch.randn(E, D + 1, dtype=dtype, device=device)
        fused_moe(hidden, bad_gate, w_up, w_down, top_k)

    # Wrong w_up shape (wrong D dimension)
    with pytest.raises(AssertionError, match="w_up must be"):
        bad_w_up = torch.randn(E, ffn_dim, D + 1, dtype=dtype, device=device)
        fused_moe(hidden, gate_w, bad_w_up, w_down, top_k)

    # Wrong w_down shape (wrong half_ffn dimension)
    with pytest.raises(AssertionError, match="w_down must be"):
        bad_w_down = torch.randn(E, D, ffn_dim // 2 + 1, dtype=dtype, device=device)
        fused_moe(hidden, gate_w, w_up, bad_w_down, top_k)

    # Bad top_k
    with pytest.raises(AssertionError, match="top_k must be in"):
        fused_moe(hidden, gate_w, w_up, w_down, 0)
    with pytest.raises(AssertionError, match="top_k must be in"):
        fused_moe(hidden, gate_w, w_up, w_down, E + 1)

    # Bad activation
    with pytest.raises(AssertionError, match="Unknown activation"):
        fused_moe(hidden, gate_w, w_up, w_down, top_k, activation="gelu")

    # Bad gating
    with pytest.raises(AssertionError, match="Unknown gating"):
        fused_moe(hidden, gate_w, w_up, w_down, top_k, gating="tanh")
