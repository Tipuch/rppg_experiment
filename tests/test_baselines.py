"""POS and CHROM, the floor a learned model has to clear.

These replace the predict-the-training-mean baseline, which is meaningful for a
scalar regression and meaningless for a waveform. Both publish ~4.06-4.08 bpm MAE
on UBFC-rPPG in every table in both source papers, so they are also a check that
this project's preprocessing has not broken something the classical methods need.

The tests use a synthetic clip because it is the only case where the answer is
known rather than estimated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.model.baselines import METHODS, chrom, pos, skin_rgb_trace
from src.model.postprocess import heart_rate

FPS = 30.0


def _clip(bpm: float, n: int = 300, side: int = 32) -> torch.Tensor:
    """A uniform skin patch modulated at `bpm`, strongest in green.

    Green because haemoglobin absorbs it most, which is why every classical
    estimator weights it heaviest -- a synthetic clip that pulsed equally in all
    three channels would be a poor test of a chrominance method.
    """
    t = np.arange(n) / FPS
    pulse = 0.01 * np.sin(2 * math.pi * (bpm / 60.0) * t)
    frames = np.empty((n, 3, side, side), dtype=np.float32)
    for channel, (base, gain) in enumerate(((0.55, 0.3), (0.45, 1.0), (0.40, 0.2))):
        frames[:, channel] = base + gain * pulse[:, None, None]
    return torch.from_numpy(frames)


@pytest.mark.parametrize("name", sorted(METHODS))
@pytest.mark.parametrize("bpm", [55.0, 72.0, 110.0])
def test_a_synthetic_pulse_is_recovered(name: str, bpm: float) -> None:
    estimate = METHODS[name](_clip(bpm), torch.ones(32, 32), FPS)
    assert heart_rate(estimate, FPS) == pytest.approx(bpm, abs=4.0)


@pytest.mark.parametrize("name", sorted(METHODS))
def test_a_static_clip_yields_no_cardiac_peak_at_the_wrong_rate(name: str) -> None:
    """A method that reports a confident rate from a still image is reporting noise
    -- which is exactly what made MCD-rPPG's 4.4% pass rate look like a result."""
    still = _clip(72.0)[:1].repeat(300, 1, 1, 1)
    estimate = METHODS[name](still, torch.ones(32, 32), FPS)
    assert not np.isfinite(heart_rate(estimate, FPS)) or estimate.std() < 1e-6


def test_the_trace_is_taken_over_skin_pixels_only() -> None:
    """Averaging in the background dilutes a signal that is already a fraction of
    a percent. The mask is the whole reason the crops are segmented."""
    frames = _clip(72.0, side=32)
    frames[:, :, :16] = 0.0                       # top half is not skin
    skin = torch.ones(32, 32)
    skin[:16] = 0.0

    masked = skin_rgb_trace(frames, skin)
    unmasked = skin_rgb_trace(frames, None)
    assert masked.shape == (300, 1, 1, 3)
    # The unmasked trace is halved by the dead region; the masked one is not.
    assert masked[:, 0, 0, 1].mean() == pytest.approx(2 * unmasked[:, 0, 0, 1].mean(), rel=0.02)


def test_an_empty_mask_falls_back_to_the_whole_frame() -> None:
    """A clip whose segmentation failed should still produce a baseline number."""
    trace = skin_rgb_trace(_clip(72.0), torch.zeros(32, 32))
    assert trace.shape == (300, 1, 1, 3)
    assert np.isfinite(trace).all()


def test_reducing_to_one_pixel_is_equivalent_to_float32_summation_order() -> None:
    """Both estimators use nothing but each frame's spatial mean, so the 1x1
    reduction is an optimisation, not an approximation.

    "Equivalent" here means to float32 summation order, not bit-exactly: gathering
    1024 skin pixels and averaging along one axis builds a different pairwise-sum
    tree than averaging over two axes, and the two land ~6e-6 apart on a value of
    0.55. Worth stating rather than hiding behind a loose tolerance, because the
    pulse being measured is only 0.01 in the same units -- the disagreement is
    0.06% of the signal, which is tolerable, and would not be if it were 10x worse.
    """
    frames = _clip(72.0)
    trace = skin_rgb_trace(frames, torch.ones(32, 32))
    reference = frames.mean(dim=(2, 3)).numpy()[:, None, None, :]
    assert np.allclose(trace, reference, rtol=1e-4, atol=1e-5)
    # And the residual must stay far below the modulation it sits on.
    pulse_amplitude = 0.01
    assert np.abs(trace - reference).max() < 0.01 * pulse_amplitude


def test_the_two_methods_disagree_at_least_somewhat() -> None:
    """They project onto different directions; identical output would mean one of
    them is not running."""
    frames, skin = _clip(72.0), torch.ones(32, 32)
    a, b = pos(frames, skin, FPS), chrom(frames, skin, FPS)
    length = min(len(a), len(b))
    assert not np.allclose(a[:length], b[:length], atol=1e-6)
