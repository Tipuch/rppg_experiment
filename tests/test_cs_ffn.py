"""Channel-spectral FFN, CFMamba Eqs. 9-12.

The stage mixes channels in the channel-frequency domain. Its two failure modes
are both silent: doing the transform along the wrong axis (which still returns the
right shape), and returning a complex tensor that later gets truncated.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.cs_ffn import ChannelSpectralFFN


def test_shape_and_realness_survive_the_round_trip() -> None:
    ffn = ChannelSpectralFFN(hidden=12)
    x = torch.randn(2, 40, 12)
    out = ffn(x)
    assert out.shape == x.shape
    assert out.dtype == x.dtype and not out.is_complex()
    assert torch.isfinite(out).all()


def test_an_identity_weight_reproduces_the_input() -> None:
    """FFT then iFFT with a unit complex weight must be the identity, which pins
    that the forward and inverse transforms are along the same axis.

    Run with the activation off: with it on, the round trip is deliberately not the
    identity, and this test is about the transform axes rather than FreMLP.
    """
    ffn = ChannelSpectralFFN(hidden=8, activation=None)
    with torch.no_grad():
        ffn.linear.weight_re.copy_(torch.eye(8))
        ffn.linear.weight_im.zero_()
        ffn.linear.bias_re.zero_()
        ffn.linear.bias_im.zero_()
    x = torch.randn(2, 16, 8)
    assert torch.allclose(ffn(x), x, atol=1e-5)


def test_the_activation_makes_the_stage_non_linear() -> None:
    """FreTS Eq. 7's sigma. Without it the whole DF-FFN is one linear operator --
    see complex_linear.complex_activation for why that matters."""
    torch.manual_seed(0)
    x1, x2 = torch.randn(2, 32, 8), torch.randn(2, 32, 8)

    def affine_error(module) -> float:
        with torch.no_grad():
            lhs = module(2.0 * x1 - x2)
            rhs = 2.0 * module(x1) - module(x2)
        return float((lhs - rhs).abs().max() / rhs.abs().max())

    assert affine_error(ChannelSpectralFFN(hidden=8, activation=None)) < 1e-5
    assert affine_error(ChannelSpectralFFN(hidden=8, activation="relu")) > 1e-2


def test_it_mixes_channels_and_not_timestamps() -> None:
    """The weights are shared across time (after Eq. 11), so permuting frames must
    permute the output identically. Mixing along time instead would break this and
    nothing else would notice."""
    torch.manual_seed(0)
    ffn = ChannelSpectralFFN(hidden=8, activation=None)
    x = torch.randn(1, 24, 8)
    order = torch.randperm(24)
    with torch.no_grad():
        assert torch.allclose(ffn(x)[:, order], ffn(x[:, order]), atol=1e-5)


def test_changing_one_channel_reaches_the_others() -> None:
    """Cross-channel interaction is the entire purpose of the stage."""
    torch.manual_seed(0)
    ffn = ChannelSpectralFFN(hidden=6)
    x = torch.zeros(1, 8, 6)
    with torch.no_grad():
        base = ffn(x)
        x[0, :, 2] = 1.0
        moved = ffn(x)
    changed = (moved - base).abs().amax(dim=1)[0]
    assert (changed > 1e-4).sum() >= 5, changed


def test_a_time_invariant_input_stays_time_invariant() -> None:
    ffn = ChannelSpectralFFN(hidden=6)
    x = torch.randn(1, 1, 6).repeat(1, 12, 1)
    with torch.no_grad():
        out = ffn(x)
    assert torch.allclose(out, out[:, :1].expand_as(out), atol=1e-6)


def test_gradients_reach_every_parameter() -> None:
    ffn = ChannelSpectralFFN(hidden=8)
    ffn(torch.randn(2, 16, 8)).square().mean().backward()
    dead = [n for n, p in ffn.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"


@pytest.mark.parametrize("n_frames", [1, 7, 160, 301])
def test_any_clip_length_is_accepted(n_frames: int) -> None:
    ffn = ChannelSpectralFFN(hidden=6)
    assert ffn(torch.randn(1, n_frames, 6)).shape == (1, n_frames, 6)
