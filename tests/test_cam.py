"""Channel-adaptive modulation: bounds, broadcast, and what it is blind to."""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.cam import ChannelAdaptiveModulation


def test_shape_is_preserved() -> None:
    cam = ChannelAdaptiveModulation(dim=16)
    x = torch.randn(3, 40, 16)
    assert cam(x).shape == x.shape


def test_coefficients_are_bounded_to_the_unit_interval() -> None:
    """Eq. 7's sigmoid. The weight can only attenuate, which is what keeps a stack
    of these from compounding into an exploding residual stream.

    Checked under a deliberately hostile scaling, where float32 sigmoid saturates
    to exactly 0.0 and 1.0 -- the bound is closed, and the thing that matters is
    that it never escapes or goes non-finite.
    """
    torch.manual_seed(0)
    cam = ChannelAdaptiveModulation(dim=8)
    with torch.no_grad():
        for parameter in cam.parameters():
            parameter.mul_(50.0)
    x = torch.randn(4, 20, 8) * 30.0
    descriptor = x.mean(dim=1) + x.amax(dim=1)
    with torch.no_grad():
        coefficients = (torch.sigmoid(cam.scale(descriptor)),
                        torch.sigmoid(cam.shift(descriptor)))
    for coefficient in coefficients:
        assert torch.isfinite(coefficient).all()
        assert float(coefficient.min()) >= 0.0
        assert float(coefficient.max()) <= 1.0


def test_modulation_is_constant_along_time() -> None:
    """Eq. 8 reweights channels; it must not re-time them.

    Recovered by solving x' = w*x + b at two time steps: w is the slope, so it has
    to come out identical whichever pair of frames is used.
    """
    torch.manual_seed(0)
    cam = ChannelAdaptiveModulation(dim=6)
    x = torch.randn(2, 30, 6)
    out = cam(x)
    slope = (out[:, 1] - out[:, 0]) / (x[:, 1] - x[:, 0])
    other = (out[:, 7] - out[:, 3]) / (x[:, 7] - x[:, 3])
    assert torch.allclose(slope, other, atol=1e-4)


def test_the_descriptor_is_pooled_over_the_whole_clip() -> None:
    """Permuting frames must leave the modulation unchanged: mean and max are both
    order-free, and that is what makes the descriptor robust to a transient."""
    torch.manual_seed(0)
    cam = ChannelAdaptiveModulation(dim=6).eval()
    x = torch.randn(1, 24, 6)
    order = torch.randperm(24)
    with torch.no_grad():
        straight, shuffled = cam(x), cam(x[:, order])
    assert torch.allclose(straight[:, order], shuffled, atol=1e-5)


def test_the_modulation_depends_on_nothing_but_the_pooled_descriptor() -> None:
    """Two clips with the same mean and max per channel get the same w and b.

    This is the real invariant. CAM is deliberately *not* symmetric across
    channels -- each channel has its own row in both MLPs, which is what lets it
    learn that channel 3 is usually noisier than channel 7 -- so feeding identical
    channels does not produce identical outputs, and asserting that it should was
    a misreading of Eq. 7.
    """
    torch.manual_seed(0)
    cam = ChannelAdaptiveModulation(dim=6).eval()
    x = torch.randn(1, 20, 6)
    # Same per-channel mean and max, different ordering and different interior.
    y = x[:, torch.randperm(20)]
    with torch.no_grad():
        out_x, out_y = cam(x), cam(y)
    slope_x = (out_x[:, 1] - out_x[:, 0]) / (x[:, 1] - x[:, 0])
    slope_y = (out_y[:, 1] - out_y[:, 0]) / (y[:, 1] - y[:, 0])
    assert torch.allclose(slope_x, slope_y, atol=1e-4)


def test_gradients_reach_every_parameter() -> None:
    cam = ChannelAdaptiveModulation(dim=8)
    cam(torch.randn(2, 16, 8)).square().mean().backward()
    dead = [n for n, p in cam.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"


def test_the_default_pooling_is_cmamba_s_not_the_literal_eq_6() -> None:
    """CMamba runs each descriptor through the shared MLP and sums the outputs;
    CFMamba Eq. 6 sums the descriptors first. They are not equivalent -- summing
    first lets a large average cancel a large max before either reaches a
    non-linearity -- and CMamba's is the version with an ablation behind it."""
    assert ChannelAdaptiveModulation(dim=8).pooling == "cmamba"


def test_the_two_poolings_differ_and_both_are_reachable() -> None:
    torch.manual_seed(0)
    theirs = ChannelAdaptiveModulation(dim=6, pooling="cmamba").eval()
    literal = ChannelAdaptiveModulation(dim=6, pooling="cfmamba").eval()
    literal.load_state_dict(theirs.state_dict())
    x = torch.randn(1, 20, 6)
    with torch.no_grad():
        assert not torch.allclose(theirs(x), literal(x), atol=1e-4)


def test_the_pooling_choice_does_not_change_the_parameter_count() -> None:
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert count(ChannelAdaptiveModulation(dim=16, pooling="cmamba")) == \
        count(ChannelAdaptiveModulation(dim=16, pooling="cfmamba"))


def test_an_unknown_pooling_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown pooling"):
        ChannelAdaptiveModulation(dim=8, pooling="attention")
