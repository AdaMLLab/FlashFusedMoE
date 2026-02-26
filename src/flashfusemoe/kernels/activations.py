"""Fused GLU activation forward + backward Triton kernels.

Supports SwiGLU, GeGLU, and ReLU² activations via a constexpr ACTIVATION
parameter — Triton compiles separate kernels per value, so there is zero
runtime overhead from the dispatch.

Replaces 6-9 separate PyTorch kernel launches (silu, mul, cat, etc.) with
2 fused Triton kernels. The backward kernel writes gate and up gradients
directly into a single output tensor, eliminating the torch.cat allocation.
"""

import torch
import triton
import triton.language as tl


# Activation codes: passed as tl.constexpr to kernels
_ACTIVATION_CODES = {"swiglu": 0, "geglu": 1, "relu_squared": 2}


@triton.jit
def _triton_tanh(x):
    """tanh(x) = 2*sigmoid(2x) - 1  (tl.math.tanh unavailable in Triton 3.6)."""
    return 2.0 * tl.sigmoid(2.0 * x) - 1.0


@triton.jit
def _glu_activation_forward_kernel(
    input_ptr,   # [M, ffn_dim]  (gate[:half] | up[half:])
    output_ptr,  # [M, half_ffn]
    M,
    half_ffn,
    stride_in_m, stride_in_d,
    stride_out_m, stride_out_d,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused GLU activation forward: output = act(gate) * up.

    gate = input[:, :half_ffn], up = input[:, half_ffn:]
    ACTIVATION: 0=SwiGLU, 1=GeGLU, 2=ReLU²
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < half_ffn
    mask = mask_m[:, None] & mask_n[None, :]

    # Load gate (first half) and up (second half)
    gate_ptrs = input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_d
    up_ptrs = input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + half_ffn) * stride_in_d

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    if ACTIVATION == 0:  # SwiGLU
        sig = tl.sigmoid(gate)
        result = gate * sig * up
    elif ACTIVATION == 1:  # GeGLU (tanh approximation)
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        inner = SQRT_2_OVER_PI * (gate + 0.044715 * gate * gate * gate)
        result = 0.5 * gate * (1.0 + _triton_tanh(inner)) * up
    elif ACTIVATION == 2:  # ReLU²
        relu_gate = tl.maximum(gate, 0.0)
        result = relu_gate * relu_gate * up

    # Store
    out_ptrs = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_d
    tl.store(out_ptrs, result.to(output_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _glu_activation_backward_kernel(
    grad_act_ptr,   # [M, half_ffn]  grad of activated output
    input_ptr,      # [M, ffn_dim]   original input (gate | up)
    grad_input_ptr, # [M, ffn_dim]   output: grad_gate to [:half], grad_up to [half:]
    M,
    half_ffn,
    stride_ga_m, stride_ga_d,
    stride_in_m, stride_in_d,
    stride_gi_m, stride_gi_d,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused GLU activation backward: writes grad_gate and grad_up directly.

    No torch.cat needed — writes to output[:, :half] and output[:, half:]
    ACTIVATION: 0=SwiGLU, 1=GeGLU, 2=ReLU²
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < half_ffn
    mask = mask_m[:, None] & mask_n[None, :]

    # Load grad_activated
    ga_ptrs = grad_act_ptr + offs_m[:, None] * stride_ga_m + offs_n[None, :] * stride_ga_d
    grad_activated = tl.load(ga_ptrs, mask=mask, other=0.0).to(tl.float32)

    # Load gate and up from original input
    gate_ptrs = input_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_d
    up_ptrs = input_ptr + offs_m[:, None] * stride_in_m + (offs_n[None, :] + half_ffn) * stride_in_d

    gate = tl.load(gate_ptrs, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptrs, mask=mask, other=0.0).to(tl.float32)

    if ACTIVATION == 0:  # SwiGLU
        sig = tl.sigmoid(gate)
        silu_gate = gate * sig
        grad_up = grad_activated * silu_gate
        grad_gate = grad_activated * up * (sig * (1.0 + gate * (1.0 - sig)))
    elif ACTIVATION == 1:  # GeGLU
        SQRT_2_OVER_PI: tl.constexpr = 0.7978845608028654
        inner = SQRT_2_OVER_PI * (gate + 0.044715 * gate * gate * gate)
        tanh_inner = _triton_tanh(inner)
        gelu = 0.5 * gate * (1.0 + tanh_inner)
        grad_up = grad_activated * gelu
        dtanh = 1.0 - tanh_inner * tanh_inner
        d_inner = SQRT_2_OVER_PI * (1.0 + 3.0 * 0.044715 * gate * gate)
        d_gelu = 0.5 * (1.0 + tanh_inner) + 0.5 * gate * dtanh * d_inner
        grad_gate = grad_activated * up * d_gelu
    elif ACTIVATION == 2:  # ReLU²
        relu_gate = tl.maximum(gate, 0.0)
        grad_up = grad_activated * relu_gate * relu_gate
        grad_gate = grad_activated * up * 2.0 * relu_gate

    # Write grad_gate to [:, :half_ffn] and grad_up to [:, half_ffn:]
    grad_gate_ptrs = grad_input_ptr + offs_m[:, None] * stride_gi_m + offs_n[None, :] * stride_gi_d
    grad_up_ptrs = grad_input_ptr + offs_m[:, None] * stride_gi_m + (offs_n[None, :] + half_ffn) * stride_gi_d

    tl.store(grad_gate_ptrs, grad_gate.to(grad_input_ptr.dtype.element_ty), mask=mask)
    tl.store(grad_up_ptrs, grad_up.to(grad_input_ptr.dtype.element_ty), mask=mask)


def activation_forward(up_batched: torch.Tensor, activation: str = "swiglu") -> torch.Tensor:
    """Fused GLU activation forward.

    Args:
        up_batched: [E, max_tokens, ffn_dim] or [M, ffn_dim]
            First half is gate, second half is up.
        activation: one of "swiglu", "geglu", "relu_squared"

    Returns:
        activated: same leading dims, last dim = ffn_dim // 2
    """
    code = _ACTIVATION_CODES[activation]
    orig_shape = up_batched.shape
    ffn_dim = orig_shape[-1]
    half_ffn = ffn_dim // 2

    # Flatten to 2D
    inp = up_batched.reshape(-1, ffn_dim)
    M = inp.shape[0]

    output = torch.empty(M, half_ffn, dtype=up_batched.dtype, device=up_batched.device)

    BLOCK_M = 64
    BLOCK_N = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(half_ffn, BLOCK_N))

    _glu_activation_forward_kernel[grid](
        inp, output,
        M, half_ffn,
        inp.stride(0), inp.stride(1),
        output.stride(0), output.stride(1),
        ACTIVATION=code,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    # Reshape output to match input leading dims
    out_shape = list(orig_shape)
    out_shape[-1] = half_ffn
    return output.reshape(out_shape)


def activation_backward(grad_activated: torch.Tensor, up_batched: torch.Tensor,
                         activation: str = "swiglu") -> torch.Tensor:
    """Fused GLU activation backward.

    Args:
        grad_activated: [E, max_tokens, half_ffn] or [M, half_ffn]
        up_batched: [E, max_tokens, ffn_dim] or [M, ffn_dim]
            Original input (gate | up).
        activation: one of "swiglu", "geglu", "relu_squared"

    Returns:
        grad_input: same shape as up_batched [E, max_tokens, ffn_dim]
            [:, :, :half] = grad_gate, [:, :, half:] = grad_up
    """
    code = _ACTIVATION_CODES[activation]
    orig_shape = up_batched.shape
    ffn_dim = orig_shape[-1]
    half_ffn = ffn_dim // 2

    ga = grad_activated.reshape(-1, half_ffn)
    inp = up_batched.reshape(-1, ffn_dim)
    M = inp.shape[0]

    grad_input = torch.empty(M, ffn_dim, dtype=up_batched.dtype, device=up_batched.device)

    BLOCK_M = 64
    BLOCK_N = 128

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(half_ffn, BLOCK_N))

    _glu_activation_backward_kernel[grid](
        ga, inp, grad_input,
        M, half_ffn,
        ga.stride(0), ga.stride(1),
        inp.stride(0), inp.stride(1),
        grad_input.stride(0), grad_input.stride(1),
        ACTIVATION=code,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        num_warps=4,
    )

    return grad_input.reshape(orig_shape)
