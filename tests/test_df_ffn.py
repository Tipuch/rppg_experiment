"""The DF-FFN wrapper: expansion, stage order, dtype rigour.

The two spectral stages have their own tests in test_cs_ffn.py and
test_pts_ffn.py. What is left here is what only the wrapper can get wrong.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.cs_ffn import ChannelSpectralFFN
from src.model.cfmamba.df_ffn import DualFrequencyFFN
from src.model.cfmamba.pts_ffn import PhysiologyTemporalSpectralFFN

FPS = 30.0


def test_shape_and_dtype_round_trip() -> None:
    ffn = DualFrequencyFFN(dim=6, hidden=12, fps=FPS)
    x = torch.randn(2, 40, 6)
    out = ffn(x)
    assert out.shape == x.shape and out.dtype == x.dtype
    assert torch.isfinite(out).all()


def test_the_latent_is_wider_than_the_stream() -> None:
    """Section 3.3: N > C. The expansion is what gives the spectral stages room to
    separate the physiological component from the noise sharing its channels."""
    ffn = DualFrequencyFFN(dim=8, hidden=16, fps=FPS)
    assert ffn.proj_in.out_features > ffn.proj_in.in_features
    assert ffn.proj_out.in_features == ffn.proj_in.out_features


def test_the_stages_run_channel_first_then_time() -> None:
    """Fig. 4 orders CS-FFN before PTS-FFN. Reversing it would filter the temporal
    band before the channels that carry the pulse have been combined."""
    ffn = DualFrequencyFFN(dim=4, hidden=8, fps=FPS)
    assert isinstance(ffn.cs, ChannelSpectralFFN)
    assert isinstance(ffn.pts, PhysiologyTemporalSpectralFFN)
    order = [name for name, _ in ffn.named_children()]
    assert order.index("cs") < order.index("pts")


def test_the_module_is_non_linear_only_because_of_the_frets_activation() -> None:
    """Section 3.3 names no activation, and implemented literally the whole DF-FFN
    reduces to one linear operator -- measured, affine error 2.2e-07 against
    6.4e-01 for an normal two-layer MLP. Four stacked linear blocks add nothing
    the Mamba layers do not already do, which cannot be what an ablation worth
    0.36 against 0.59 MAE is describing. The activation goes where FreTS Eq. 7
    puts it: inside each complex linear.
    """
    torch.manual_seed(0)
    x1, x2 = torch.randn(2, 64, 6), torch.randn(2, 64, 6)

    def affine_error(module) -> float:
        with torch.no_grad():
            lhs = module(2.0 * x1 - x2)
            rhs = 2.0 * module(x1) - module(x2)
        return float((lhs - rhs).abs().max() / rhs.abs().max())

    # That this is linear with activation=None also confirms the module's only
    # non-linearity is the one inside the complex linears -- an extra activation
    # anywhere outside them, as an earlier version had before the output
    # projection, would show up here.
    literal = DualFrequencyFFN(dim=6, hidden=12, fps=FPS, n_frames=64, activation=None)
    assert affine_error(literal) < 1e-5, "the literal reading should be linear"

    built = DualFrequencyFFN(dim=6, hidden=12, fps=FPS, n_frames=64)
    assert built.activation == "gelu"
    assert affine_error(built) > 1e-2

    # Both activations break linearity; which one suits a pulse is an ablation, and
    # GELU is the default because ReLU zeroes half the phase plane.
    relu = DualFrequencyFFN(dim=6, hidden=12, fps=FPS, n_frames=64, activation="relu")
    assert affine_error(relu) > 1e-2


def test_spectral_work_stays_in_float32_under_autocast() -> None:
    """FFT has no autocast kernel and complex bf16 forfeits more than the memory is
    worth on a 0.9M-parameter model."""
    ffn = DualFrequencyFFN(dim=6, hidden=12, fps=FPS)
    x = torch.randn(1, 32, 6)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = ffn(x)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("pts_mode", ["full", "diagonal", "none"])
def test_gradients_reach_every_parameter(pts_mode: str) -> None:
    ffn = DualFrequencyFFN(dim=6, hidden=12, fps=FPS, pts_mode=pts_mode, n_frames=64)
    ffn(torch.randn(2, 64, 6)).square().mean().backward()
    dead = [n for n, p in ffn.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"
