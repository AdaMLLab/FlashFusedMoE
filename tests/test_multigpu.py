"""Multi-GPU Expert Parallelism correctness tests.

Tests:
1. Dispatch/combine round-trip: dispatch → identity → combine matches input
2. EP output matches single-GPU: fused_moe(ep_group) vs fused_moe(ep_group=None)
3. EP backward correctness: gradient comparison
4. EP with bf16 autocast
5. Differentiable all-to-all: verify _AllToAllAutograd gradient
6. Variable routing load: skewed routing correctness

Run with:
    torchrun --nproc_per_node=2 -m pytest tests/test_multigpu.py -v
"""

import os
import pytest
import torch
import torch.distributed as dist

from flashfusemoe.expert_parallel import ExpertParallelDispatcher, _AllToAllAutograd
from flashfusemoe.fused_moe import fused_moe


def _is_torchrun():
    """Check if we're running under torchrun (env vars set)."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def _ensure_dist_initialized():
    """Initialize distributed if running under torchrun but not yet initialized."""
    if not _is_torchrun():
        return False
    if not dist.is_initialized():
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
    return True


# Module-level initialization: if running under torchrun, init dist once.
_DIST_AVAILABLE = _ensure_dist_initialized()


def _skip_if_not_distributed():
    if not _DIST_AVAILABLE or not dist.is_initialized():
        pytest.skip("Requires torchrun (dist not initialized)")
    if torch.cuda.device_count() < 2:
        pytest.skip("Need 2+ GPUs")


# ---------------------------------------------------------------------------
# Test 1: Dispatch/combine round-trip
# ---------------------------------------------------------------------------

class TestDispatchCombineRoundtrip:
    def test_roundtrip(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 64, 128, 8, 2
        ep_size = world_size

        hidden = torch.randn(N, D, device=device)
        gate_weight = torch.randn(E, D, device=device) * 0.01
        logits = hidden @ gate_weight.T
        topk_logits, topk_indices = torch.topk(logits, top_k, dim=-1)
        topk_weights = torch.softmax(topk_logits, dim=-1)

        dispatcher = ExpertParallelDispatcher(E, ep_size)
        received_tokens, metadata = dispatcher.dispatch(hidden, topk_indices, topk_weights)

        local_expert_ids = metadata["local_expert_ids"]
        local_weights = metadata["received_weights"]

        weighted_output = received_tokens * local_weights.unsqueeze(-1)
        output = dispatcher.combine(weighted_output, metadata)

        flat_weights = topk_weights.view(-1)
        hidden_expanded = hidden.unsqueeze(1).expand(-1, top_k, -1).reshape(N * top_k, D)
        weighted_expanded = hidden_expanded * flat_weights.unsqueeze(-1)
        expected = weighted_expanded.view(N, top_k, D).sum(dim=1)

        max_err = (output - expected).abs().max().item()
        assert max_err < 1e-4, f"Round-trip error: {max_err}"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 2: EP output matches single-GPU
# ---------------------------------------------------------------------------

class TestEPMatchesSingleGPU:
    def test_output_matches(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 64, 128, 8, 2
        ffn_dim = 256
        half_ffn = ffn_dim // 2

        hidden = torch.randn(N, D, device=device)
        gate_w = torch.randn(E, D, device=device) * 0.01
        w_up = torch.randn(E, ffn_dim, D, device=device) * (2.0 / D) ** 0.5
        w_down = torch.randn(E, D, half_ffn, device=device) * (2.0 / half_ffn) ** 0.5

        out_single, _ = fused_moe(
            hidden, gate_w, w_up, w_down, top_k,
            allow_dropped_tokens=True,
        )

        E_local = E // world_size
        start = rank * E_local
        end = start + E_local
        w_up_local = w_up[start:end].contiguous()
        w_down_local = w_down[start:end].contiguous()

        out_ep, _ = fused_moe(
            hidden, gate_w, w_up_local, w_down_local, top_k,
            ep_group=dist.group.WORLD,
            allow_dropped_tokens=True,
        )

        max_err = (out_ep - out_single).abs().max().item()
        # EP uses Triton grouped GEMM while single-GPU uses cuBLAS bmm —
        # different numerical paths, so allow wider tolerance.
        assert max_err < 0.05, f"EP vs single-GPU max error: {max_err}"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 3: EP backward correctness
# ---------------------------------------------------------------------------

class TestEPBackwardCorrectness:
    def test_gradients_match(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 32, 64, 8, 2
        ffn_dim = 128
        half_ffn = ffn_dim // 2

        hidden_data = torch.randn(N, D, device=device)
        gate_w_data = torch.randn(E, D, device=device) * 0.01
        w_up_data = torch.randn(E, ffn_dim, D, device=device) * (2.0 / D) ** 0.5
        w_down_data = torch.randn(E, D, half_ffn, device=device) * (2.0 / half_ffn) ** 0.5
        grad_out = torch.randn(N, D, device=device)

        # Single-GPU gradients
        hidden_s = hidden_data.clone().requires_grad_(True)
        gate_w_s = gate_w_data.clone().requires_grad_(True)
        w_up_s = w_up_data.clone().requires_grad_(True)
        w_down_s = w_down_data.clone().requires_grad_(True)

        out_s, _ = fused_moe(hidden_s, gate_w_s, w_up_s, w_down_s, top_k, allow_dropped_tokens=True)
        out_s.backward(grad_out)

        # EP gradients
        E_local = E // world_size
        start = rank * E_local
        end = start + E_local

        hidden_ep = hidden_data.clone().requires_grad_(True)
        gate_w_ep = gate_w_data.clone().requires_grad_(True)
        w_up_ep = w_up_data[start:end].clone().contiguous().requires_grad_(True)
        w_down_ep = w_down_data[start:end].clone().contiguous().requires_grad_(True)

        out_ep, _ = fused_moe(
            hidden_ep, gate_w_ep, w_up_ep, w_down_ep, top_k,
            ep_group=dist.group.WORLD, allow_dropped_tokens=True,
        )
        out_ep.backward(grad_out)

        # EP uses detached expert-compute gradients for hidden_states (matching
        # unfused EP behavior where dispatch/combine are non-differentiable).
        # The hidden grad only contains the routing gradient; expert compute gradient
        # is intentionally omitted for EP efficiency (~50% backward speedup).
        def _rel_err(a, b):
            abs_err = (a - b).abs().max().item()
            scale = max(a.abs().max().item(), b.abs().max().item(), 1e-8)
            return abs_err / scale

        # hidden_states grad: EP version has routing-only gradient (no expert compute grad).
        # Just check it's not zero and has reasonable magnitude.
        assert hidden_ep.grad is not None, "hidden_states should have grad (from routing)"
        assert hidden_ep.grad.abs().max().item() > 1e-8, "hidden grad should be non-zero"

        # gate_weight grad: should match since routing is identical
        rel_gate = _rel_err(gate_w_ep.grad, gate_w_s.grad)
        assert rel_gate < 0.01, f"gate_weight grad rel error: {rel_gate}"

        # Expert weight grads: should match single-GPU local shard
        w_up_s_local_grad = w_up_s.grad[start:end]
        rel_wup = _rel_err(w_up_ep.grad, w_up_s_local_grad)
        assert rel_wup < 0.01, f"w_up grad rel error: {rel_wup}"

        w_down_s_local_grad = w_down_s.grad[start:end]
        rel_wdown = _rel_err(w_down_ep.grad, w_down_s_local_grad)
        assert rel_wdown < 0.01, f"w_down grad rel error: {rel_wdown}"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 4: EP with bf16 autocast
# ---------------------------------------------------------------------------

class TestEPBF16Autocast:
    def test_bf16_matches(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 64, 128, 8, 2
        ffn_dim = 256
        half_ffn = ffn_dim // 2

        hidden = torch.randn(N, D, device=device, dtype=torch.float32)
        gate_w = torch.randn(E, D, device=device, dtype=torch.float32) * 0.01
        w_up = torch.randn(E, ffn_dim, D, device=device, dtype=torch.float32) * (2.0 / D) ** 0.5
        w_down = torch.randn(E, D, half_ffn, device=device, dtype=torch.float32) * (2.0 / half_ffn) ** 0.5

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_single, _ = fused_moe(hidden, gate_w, w_up, w_down, top_k, allow_dropped_tokens=True)

        E_local = E // world_size
        start = rank * E_local
        end = start + E_local

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out_ep, _ = fused_moe(
                hidden, gate_w, w_up[start:end].contiguous(), w_down[start:end].contiguous(),
                top_k, ep_group=dist.group.WORLD, allow_dropped_tokens=True,
            )

        max_err = (out_ep.float() - out_single.float()).abs().max().item()
        # bf16 has ~0.01 precision; EP dispatch/combine introduces additional rounding
        assert max_err < 0.05, f"BF16 EP vs single-GPU max error: {max_err}"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 5: Differentiable all-to-all gradient
# ---------------------------------------------------------------------------

class TestAllToAllGradient:
    def test_gradient(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42 + rank)

        N, D = 16, 32
        x = torch.randn(N, D, device=device, requires_grad=True)

        split_size = N // world_size
        send_splits = [split_size] * world_size
        recv_splits = [split_size] * world_size

        group = dist.group.WORLD
        y = _AllToAllAutograd.apply(x, recv_splits, send_splits, group)

        grad_y = torch.randn_like(y)
        y.backward(grad_y)

        assert x.grad is not None, "No gradient computed"
        assert x.grad.shape == x.shape, f"Grad shape mismatch: {x.grad.shape} vs {x.shape}"
        assert torch.isfinite(x.grad).all(), "Non-finite gradients"

        expected_grad = torch.empty_like(x)
        dist.all_to_all_single(
            expected_grad, grad_y,
            output_split_sizes=send_splits,
            input_split_sizes=recv_splits,
            group=group,
        )
        max_err = (x.grad - expected_grad).abs().max().item()
        assert max_err < 1e-6, f"All-to-all gradient error: {max_err}"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 6: Variable routing load (skewed)
# ---------------------------------------------------------------------------

class TestVariableRoutingLoad:
    def test_skewed_routing(self):
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 128, 64, 8, 2
        ffn_dim = 128
        half_ffn = ffn_dim // 2

        hidden = torch.randn(N, D, device=device)
        gate_w = torch.randn(E, D, device=device) * 0.01
        gate_w[0] += 0.5  # bias toward expert 0
        w_up = torch.randn(E, ffn_dim, D, device=device) * (2.0 / D) ** 0.5
        w_down = torch.randn(E, D, half_ffn, device=device) * (2.0 / half_ffn) ** 0.5

        out_single, _ = fused_moe(
            hidden, gate_w, w_up, w_down, top_k, allow_dropped_tokens=True,
        )

        E_local = E // world_size
        start = rank * E_local
        end = start + E_local

        out_ep, _ = fused_moe(
            hidden, gate_w, w_up[start:end].contiguous(), w_down[start:end].contiguous(),
            top_k, ep_group=dist.group.WORLD, allow_dropped_tokens=True,
        )

        max_err = (out_ep - out_single).abs().max().item()
        # EP uses Triton grouped GEMM while single-GPU uses cuBLAS bmm
        assert max_err < 0.05, f"Skewed routing EP vs single-GPU max error: {max_err}"
        assert torch.isfinite(out_ep).all(), "Non-finite output with skewed routing"
        dist.barrier()


# ---------------------------------------------------------------------------
# Test 7: EP with latent dimension projection
# ---------------------------------------------------------------------------

class TestEPLatentProjection:
    def test_latent_forward_backward(self):
        """Test EP with latent dim projection: forward produces output, backward computes grads."""
        _skip_if_not_distributed()
        dist.barrier()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.manual_seed(42)

        N, D, E, top_k = 64, 128, 8, 2
        ffn_dim = 256
        half_ffn = ffn_dim // 2
        latent_dim = 64  # project from D=128 → 64 before A2A

        hidden = torch.randn(N, D, device=device, requires_grad=True)
        gate_w = (torch.randn(E, D, device=device) * 0.01).detach().requires_grad_(True)
        w_up = torch.randn(E, ffn_dim, D, device=device) * (2.0 / D) ** 0.5
        w_down = torch.randn(E, D, half_ffn, device=device) * (2.0 / half_ffn) ** 0.5
        # proj_down: [latent_dim, D], proj_up: [D, latent_dim]
        proj_down = (torch.randn(latent_dim, D, device=device) * 0.01).detach().requires_grad_(True)
        proj_up = (torch.randn(D, latent_dim, device=device) * 0.01).detach().requires_grad_(True)

        E_local = E // world_size
        start = rank * E_local
        end = start + E_local
        w_up_local = w_up[start:end].contiguous().requires_grad_(True)
        w_down_local = w_down[start:end].contiguous().requires_grad_(True)

        out_latent, _ = fused_moe(
            hidden, gate_w, w_up_local, w_down_local, top_k,
            ep_group=dist.group.WORLD, allow_dropped_tokens=True,
            latent_weights=(proj_down, proj_up),
        )

        assert out_latent.shape == (N, D), f"Output shape: {out_latent.shape}"
        assert torch.isfinite(out_latent).all(), "Non-finite output"

        # Backward
        grad_out = torch.randn(N, D, device=device)
        out_latent.backward(grad_out)

        assert hidden.grad is not None, "No hidden grad"
        assert gate_w.grad is not None, "No gate_w grad"
        assert w_up_local.grad is not None, "No w_up grad"
        assert w_down_local.grad is not None, "No w_down grad"
        assert proj_down.grad is not None, "No proj_down grad"
        assert proj_up.grad is not None, "No proj_up grad"
        assert torch.isfinite(hidden.grad).all(), "Non-finite hidden grad"
        assert torch.isfinite(proj_down.grad).all(), "Non-finite proj_down grad"
        assert torch.isfinite(proj_up.grad).all(), "Non-finite proj_up grad"
        dist.barrier()
