"""Mamba-3's scan in plain PyTorch, from src/model/cfmamba/mamba3_export.py.

Two questions, and they need different evidence:

**Is the arithmetic right?** `Mamba3Sequential` uses the Triton kernel's own
arrangement -- one scaling of `K_t` by `dt_t l_t + dt_{t+1} (1 - l_{t+1})` --
rather than Proposition 1's two-term update. Those are equal, but checking the
implementation against the reasoning that produced it proves nothing. So
`_proposition_one` below is written straight from the paper's recurrence, carrying
`Bx_{t-1}` explicitly, and the two are compared. It runs on CPU, so this is the
test that guards the port for anyone without a GPU.

**Does it match the kernel?** That needs CUDA and is marked accordingly. The
CPU test above is the one that catches a regression; this one catches a
divergence from `mamba_ssm` itself, which the reference test cannot see.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from src.model.cfmamba.mamba3_export import (
    Mamba3Sequential,
    _apply_interleaved_rope,
    _rms_norm,
)

# CFMamba-Phys's own configuration, so the test exercises the shapes that ship.
CONFIG = {
    "d_model": 80, "d_state": 16, "expand": 2, "headdim": 32,
    "rope_fraction": 1.0, "is_mimo": False, "mimo_rank": 1, "chunk_size": 32,
}

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="the Mamba-3 kernel is CUDA-only"
)


def _proposition_one(layer: Mamba3Sequential, u: torch.Tensor) -> torch.Tensor:
    """Mamba-3 Proposition 1, written out with the previous step carried explicitly.

        h_t = a_t h_{t-1} + (1 - l_t) dt_t a_t Bx_{t-1} + l_t dt_t Bx_t
        y_t = <C_t, h_t> + D x_t,   then gated by silu(z_t)

    Deliberately *not* the form the module uses. `Bx_{t-1}` is kept as its own
    term and no `scale` is precomputed, so agreement between the two is evidence
    about the algebra rather than a restatement of it.
    """
    batch, seqlen, _ = u.shape
    heads, headdim, state = layer.nheads, layer.headdim, layer.d_state

    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        layer.in_proj(u), layer._split, dim=-1
    )
    z = z.unflatten(-1, (heads, headdim))
    x = x.unflatten(-1, (heads, headdim))
    A = (-F.softplus(dd_A.float())).clamp(max=-layer.A_floor)
    dt = F.softplus(dd_dt + layer.dt_bias)
    lam = torch.sigmoid(trap)

    key = _rms_norm(B, layer.B_norm.weight).unsqueeze(2) + layer.B_bias.squeeze(1)
    query = _rms_norm(C, layer.C_norm.weight).unsqueeze(2) + layer.C_bias.squeeze(1)
    theta = torch.cumsum(
        (torch.tanh(angles) * math.pi).unsqueeze(2) * dt.unsqueeze(-1), dim=1
    )
    cos, sin = torch.cos(theta), torch.sin(theta)
    query_rot = _apply_interleaved_rope(query, cos, sin)
    key_rot = _apply_interleaved_rope(key, cos, sin)

    outer = x.unsqueeze(-1) * key_rot.unsqueeze(-2)          # (B, L, H, P, N) = Bx
    state_h = u.new_zeros(batch, heads, headdim, state)
    outputs = []
    for step in range(seqlen):
        decay = torch.exp(A[:, step] * dt[:, step]).view(batch, heads, 1, 1)
        weight = (dt[:, step] * lam[:, step]).view(batch, heads, 1, 1)
        state_h = state_h * decay + weight * outer[:, step]
        if step > 0:
            previous = (
                (1.0 - lam[:, step]) * dt[:, step]
            ).view(batch, heads, 1, 1) * decay
            state_h = state_h + previous * outer[:, step - 1]
        y = torch.einsum("bhpn,bhn->bhp", state_h, query_rot[:, step])
        y = y + layer.D.view(1, heads, 1) * x[:, step]
        outputs.append(y * F.silu(z[:, step]))
    return layer.out_proj(torch.stack(outputs, dim=1).flatten(-2))


def test_the_scan_matches_proposition_one() -> None:
    """The folded `scale` form against the paper's two-term recurrence.

    The two differ only in bookkeeping, so they should agree to float32 rounding.
    A real error in the fold -- the wrong step's `l`, a missing decay on the
    previous-step term -- moves this by orders of magnitude, not by epsilon.
    """
    torch.manual_seed(0)
    layer = Mamba3Sequential(**CONFIG).eval()
    u = torch.randn(2, 64, CONFIG["d_model"])
    with torch.no_grad():
        got, expected = layer(u), _proposition_one(layer, u)
    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


def test_the_previous_step_term_carries_the_decay() -> None:
    """A guard on the correction that is easiest to get wrong and hardest to see.

    Dropping the `a_t` factor from Proposition 1's previous-step term is the error
    `rishikksh20/mamba3-pytorch` makes, and it still scores Pearson 0.97 against
    the kernel -- plausible enough to ship. This asserts the two forms are *not*
    interchangeable, so `test_the_scan_matches_proposition_one` is a real
    constraint rather than a tautology.
    """
    torch.manual_seed(0)
    layer = Mamba3Sequential(**CONFIG).eval()
    u = torch.randn(1, 48, CONFIG["d_model"])

    with torch.no_grad():
        correct = layer(u)
        wrong = _proposition_one_without_decay(layer, u)
    assert not torch.allclose(correct, wrong, atol=1e-3)


def _proposition_one_without_decay(
    layer: Mamba3Sequential, u: torch.Tensor
) -> torch.Tensor:
    """`_proposition_one` with the `a_t` dropped from the previous-step term."""
    batch, seqlen, _ = u.shape
    heads, headdim, state = layer.nheads, layer.headdim, layer.d_state
    z, x, B, C, dd_dt, dd_A, trap, angles = torch.split(
        layer.in_proj(u), layer._split, dim=-1
    )
    z = z.unflatten(-1, (heads, headdim))
    x = x.unflatten(-1, (heads, headdim))
    A = (-F.softplus(dd_A.float())).clamp(max=-layer.A_floor)
    dt = F.softplus(dd_dt + layer.dt_bias)
    lam = torch.sigmoid(trap)
    key = _rms_norm(B, layer.B_norm.weight).unsqueeze(2) + layer.B_bias.squeeze(1)
    query = _rms_norm(C, layer.C_norm.weight).unsqueeze(2) + layer.C_bias.squeeze(1)
    theta = torch.cumsum(
        (torch.tanh(angles) * math.pi).unsqueeze(2) * dt.unsqueeze(-1), dim=1
    )
    cos, sin = torch.cos(theta), torch.sin(theta)
    query_rot = _apply_interleaved_rope(query, cos, sin)
    key_rot = _apply_interleaved_rope(key, cos, sin)
    outer = x.unsqueeze(-1) * key_rot.unsqueeze(-2)
    state_h = u.new_zeros(batch, heads, headdim, state)
    outputs = []
    for step in range(seqlen):
        decay = torch.exp(A[:, step] * dt[:, step]).view(batch, heads, 1, 1)
        weight = (dt[:, step] * lam[:, step]).view(batch, heads, 1, 1)
        state_h = state_h * decay + weight * outer[:, step]
        if step > 0:
            # The missing `* decay` is the whole point of this function.
            previous = ((1.0 - lam[:, step]) * dt[:, step]).view(batch, heads, 1, 1)
            state_h = state_h + previous * outer[:, step - 1]
        y = torch.einsum("bhpn,bhn->bhp", state_h, query_rot[:, step])
        y = y + layer.D.view(1, heads, 1) * x[:, step]
        outputs.append(y * F.silu(z[:, step]))
    return layer.out_proj(torch.stack(outputs, dim=1).flatten(-2))


def test_rope_pairing_is_interleaved() -> None:
    """(0, 1), (2, 3), ... -- not split-half.

    The kernel reaches this through `reshape(..., N // 2, 2)` then `split`. Getting
    it wrong is not a crash, just a worse model: split-half pairing scores 0.99253
    against 0.99534 on the trained weights.
    """
    x = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    quarter = torch.full((1, 2), math.pi / 2)
    rotated = _apply_interleaved_rope(x, torch.cos(quarter), torch.sin(quarter))
    # A quarter turn sends (1, 0) to (0, 1) inside each adjacent pair.
    assert torch.allclose(rotated, torch.tensor([[0.0, 1.0, 0.0, 1.0]]), atol=1e-6)


def test_shape_is_preserved_and_runs_on_cpu() -> None:
    layer = Mamba3Sequential(**CONFIG).eval()
    u = torch.randn(2, 37, CONFIG["d_model"])
    assert layer(u).shape == u.shape


def test_the_unsupported_configurations_are_refused() -> None:
    """Each would load a checkpoint without complaint and then compute something else."""
    for bad in ({"is_mimo": True}, {"is_outproj_norm": True}, {"ngroups": 2}):
        with pytest.raises(ValueError):
            Mamba3Sequential(**{**CONFIG, **bad})


@cuda
def test_state_dict_is_interchangeable_with_the_kernel() -> None:
    from mamba_ssm import Mamba3

    kernel = Mamba3(**CONFIG)
    port = Mamba3Sequential(**CONFIG)
    assert set(kernel.state_dict()) == set(port.state_dict())
    for key, value in kernel.state_dict().items():
        assert value.shape == port.state_dict()[key].shape, key
    port.load_state_dict(kernel.state_dict(), strict=True)


@cuda
def test_the_scan_matches_the_kernel() -> None:
    """Pearson above 0.9999 is the gate. The kernel's own noise floor is 0.99998.

    The residual is the kernel's PTX `cos.approx.f32`, `sin.approx.f32` and
    `tanh.approx.f32`, which its own utils.py documents as trading accuracy for
    speed -- so where the two differ, this module is the more accurate one.
    """
    from mamba_ssm import Mamba3

    torch.manual_seed(0)
    kernel = Mamba3(**CONFIG).cuda().eval()
    port = Mamba3Sequential(**CONFIG).cuda().eval()
    port.load_state_dict(kernel.state_dict(), strict=True)

    u = torch.randn(1, 300, CONFIG["d_model"], device="cuda")
    with torch.no_grad():
        a, b = kernel(u).float().flatten(), port(u).float().flatten()
    pearson = torch.corrcoef(torch.stack([a, b]))[0, 1].item()
    assert pearson > 0.9999, pearson
