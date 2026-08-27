"""Physiology-guided attention: the prior, the L1 energy, and the collapse to time.

PGA is where the video stops being a video. Everything after it is a (B, T, C)
time series, so an error here is invisible downstream -- the shapes are right and
the loss still falls, it just falls onto the wrong part of the face.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.pga import PhysiologyGuidedAttention


def _mask(height: int, width: int, box: tuple[int, int, int, int]) -> torch.Tensor:
    top, left, bottom, right = box
    mask = torch.zeros(1, height, width)
    mask[:, top:bottom, left:right] = 1.0
    return mask


# --- the Gaussian prior, Eq. 1 ------------------------------------------------

def test_the_prior_peaks_at_the_skin_centroid() -> None:
    """Not at the frame centre. A prior that ignores where the face actually is
    would be a centre-crop bias wearing a physiological label."""
    pga = PhysiologyGuidedAttention()
    # Skin pushed into the top-left quadrant.
    prior = pga.gaussian_prior(_mask(64, 64, (4, 8, 20, 24)), 16, 16)
    peak = int(prior.view(-1).argmax())
    row, col = divmod(peak, 16)
    assert row < 8 and col < 8, (row, col)
    # A Gaussian sampled on a grid reaches exactly 1.0 only when the centroid
    # lands on a cell. Here it falls half a cell off in both axes, so the peak is
    # 0.82 -- the bound checks the prior is a peak, not that it is a delta.
    assert 0.75 < float(prior.max()) <= 1.0


def test_the_prior_follows_the_mask_when_it_moves() -> None:
    pga = PhysiologyGuidedAttention()
    left = pga.gaussian_prior(_mask(64, 64, (24, 4, 40, 20)), 16, 16)
    right = pga.gaussian_prior(_mask(64, 64, (24, 44, 40, 60)), 16, 16)
    assert int(left.view(-1).argmax()) % 16 < 8
    assert int(right.view(-1).argmax()) % 16 >= 8


def test_a_wider_mask_gives_a_wider_prior() -> None:
    """sigma is the mask's own second moment, so it tracks how much of the frame
    the face fills instead of being one more constant to guess."""
    pga = PhysiologyGuidedAttention()
    narrow = pga.gaussian_prior(_mask(64, 64, (28, 28, 36, 36)), 16, 16)
    wide = pga.gaussian_prior(_mask(64, 64, (8, 8, 56, 56)), 16, 16)
    assert float(wide.mean()) > float(narrow.mean())


def test_an_empty_mask_degrades_to_a_centre_bias_rather_than_a_nan() -> None:
    """A clip whose skin mask came back empty must still train."""
    pga = PhysiologyGuidedAttention()
    prior = pga.gaussian_prior(torch.zeros(1, 64, 64), 16, 16)
    assert torch.isfinite(prior).all()
    row, col = divmod(int(prior.view(-1).argmax()), 16)
    assert 6 <= row <= 9 and 6 <= col <= 9, (row, col)


def test_the_prior_is_per_item_in_the_batch() -> None:
    pga = PhysiologyGuidedAttention()
    skin = torch.cat([
        _mask(32, 32, (2, 2, 10, 10)),
        _mask(32, 32, (22, 22, 30, 30)),
    ])
    prior = pga.gaussian_prior(skin, 8, 8)
    assert prior.shape == (2, 1, 8, 8)
    assert int(prior[0].view(-1).argmax()) < int(prior[1].view(-1).argmax())


def test_the_prior_rejects_a_wrongly_shaped_mask() -> None:
    pga = PhysiologyGuidedAttention()
    with pytest.raises(ValueError, match="expected skin"):
        pga.gaussian_prior(torch.zeros(1, 1, 8, 8), 4, 4)


# --- attention and pooling, Eqs. 2-5 -----------------------------------------

def test_space_is_collapsed_into_channels() -> None:
    pga = PhysiologyGuidedAttention()
    x = torch.rand(2, 12, 10, 16, 16) + 0.5
    out = pga(x, torch.ones(2, 128, 128))
    assert out.shape == (2, 12, 10)
    assert torch.isfinite(out).all()


def test_attention_energy_is_normalised_per_channel_and_frame() -> None:
    """Eq. 4 fixes the L1 energy of every (channel, frame) map at H*W/2.

    Without it a channel could dominate the pooled average by sheer magnitude
    rather than by where it is looking, which is the opposite of the intent.
    """
    pga = PhysiologyGuidedAttention()
    x = torch.rand(2, 5, 4, 8, 8) + 0.5
    prior = pga.gaussian_prior(torch.ones(2, 64, 64), 8, 8)
    feat = x / (pga.eps + x.mean(dim=(3, 4), keepdim=True)) * pga.gamma
    combined = feat * prior.unsqueeze(1)
    attention = 64 * combined / (2.0 * combined.abs().sum(dim=(3, 4), keepdim=True))
    energy = attention.abs().sum(dim=(3, 4))
    assert torch.allclose(energy, torch.full_like(energy, 64 / 2), atol=1e-4)


def test_the_prior_actually_reweights_the_pooled_output() -> None:
    """A left-biased mask and a right-biased mask must not pool to the same thing.

    If they do, the prior is wired up but inert -- the failure mode that would
    make the PGA ablation come out flat.
    """
    pga = PhysiologyGuidedAttention()
    x = torch.rand(1, 4, 3, 16, 16) + 0.5
    left = pga(x, _mask(128, 128, (48, 8, 80, 40)))
    right = pga(x, _mask(128, 128, (48, 88, 80, 120)))
    assert not torch.allclose(left, right, atol=1e-3)


def test_a_uniform_input_gives_a_prior_shaped_map() -> None:
    """With a flat input, Eq. 2 is constant and all the structure is the prior."""
    pga = PhysiologyGuidedAttention()
    x = torch.ones(1, 3, 2, 8, 8)
    out = pga(x, torch.ones(1, 64, 64))
    prior = pga.gaussian_prior(torch.ones(1, 64, 64), 8, 8)
    expected = float((prior * 64 / (2 * prior.abs().sum())).mean())
    assert out.flatten().tolist() == pytest.approx([expected] * 6, rel=1e-4)


def test_gradients_flow_back_through_the_pooling() -> None:
    pga = PhysiologyGuidedAttention()
    x = (torch.rand(1, 6, 4, 8, 8) + 0.5).requires_grad_(True)
    pga(x, torch.ones(1, 64, 64)).square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.any()
