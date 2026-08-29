"""The learnable Gaussian cardiac band mask, CFMamba Eqs. 14-15.

Two of these are ported from the deleted tests/test_head.py. Each guarded a bug in
the previous model that raised nothing and returned credible numbers: a frequency
band selected by rFFT bin index, so the same code meant 45-202 bpm at T=160 and
24-108 bpm at T=300; and a spectral peak point-sampled out of existence while
downsampling. This is the module where a bin index would be incorrect for a
frequency again, so this is where they live now.
"""

from __future__ import annotations

import pytest
import torch

from src.model.cfmamba.band_mask import (
    BW_MAX_HZ,
    BW_MIN_HZ,
    FC_MAX_HZ,
    FC_MIN_HZ,
    GaussianBandMask,
)

FPS = 30.0


@pytest.mark.parametrize("n_frames", [100, 160, 300, 450])
def test_mask_centre_is_a_frequency_not_a_bin(n_frames: int) -> None:
    """The ported regression: the peak must land at the same Hz for every T.

    FFT bin k means k*fps/T. A mask built from bin numbers keeps its shape and
    raises nothing while meaning a different filter at every clip length.
    """
    mask = GaussianBandMask(fps=FPS)
    with torch.no_grad():
        mask.theta_fc.fill_(0.5)
        values = mask(n_frames, torch.device("cpu"), torch.float32)
    freqs = torch.fft.fftfreq(n_frames, d=1.0 / FPS)
    peak_hz = abs(float(freqs[int(values.argmax())]))
    assert peak_hz == pytest.approx(float(mask.centre_hz.detach()), abs=FPS / n_frames)


def test_mask_is_symmetric_about_zero_frequency() -> None:
    """Eq. 15's two lobes exist because a real signal's spectrum is symmetric.

    A single lobe would keep +f_c and discard -f_c, halving the energy and leaving
    the inverse transform complex.
    """
    mask = GaussianBandMask(fps=FPS)
    with torch.no_grad():
        values = mask(160, torch.device("cpu"), torch.float32)
    # fftfreq bin i and bin -i are equal and opposite, so the mask must match.
    assert torch.allclose(values[1:], values.flip(0)[:-1], atol=1e-6)


@pytest.mark.parametrize("theta", [-40.0, -3.0, 0.0, 3.0, 40.0])
def test_the_band_cannot_leave_the_physiological_range(theta: float) -> None:
    """Eq. 14's sigmoid is a hard constraint, and that is the point of it.

    An unconstrained centre frequency would be free to resolve on a motion artifact
    or a mains-lighting flicker, which is exactly what the prior exists to rule out.
    """
    mask = GaussianBandMask(fps=FPS)
    with torch.no_grad():
        mask.theta_fc.fill_(theta)
        mask.theta_bw.fill_(theta)
    assert FC_MIN_HZ <= float(mask.centre_hz.detach()) <= FC_MAX_HZ
    assert BW_MIN_HZ <= float(mask.bandwidth_hz.detach()) <= BW_MAX_HZ


def test_the_mask_keeps_a_cardiac_tone_and_suppresses_an_out_of_band_one() -> None:
    mask = GaussianBandMask(fps=FPS)              # centre 1.625 Hz, bw 0.6 Hz
    with torch.no_grad():
        values = mask(300, torch.device("cpu"), torch.float32)
    freqs = torch.fft.fftfreq(300, d=1.0 / FPS)

    def gain_at(hz: float) -> float:
        return float(values[int((freqs - hz).abs().argmin())])

    # Attenuation is relative: what matters is the cardiac peak's margin over the
    # noise beside it, not an absolute gain. At the initial 1.625 Hz / 0.6 Hz the
    # mask passes 6.9% of a 0.15 Hz drift -- a 14x margin, and training sharpens it.
    assert gain_at(1.6) > 0.9                                # 96 bpm, dead centre
    assert gain_at(0.15) / gain_at(1.6) < 0.1                # slow drift
    assert gain_at(6.0) / gain_at(1.6) < 1e-6                # above any heart rate


def test_the_whole_physiological_band_is_reachable() -> None:
    """45 bpm and 150 bpm must both be attainable centres, or the prior excludes
    subjects it is meant to cover -- UBFC alone spans 49 to 153 bpm."""
    mask = GaussianBandMask(fps=FPS)
    for theta, expected in ((-20.0, FC_MIN_HZ), (20.0, FC_MAX_HZ)):
        with torch.no_grad():
            mask.theta_fc.fill_(theta)
        assert float(mask.centre_hz.detach()) == pytest.approx(expected, abs=1e-6)


def test_gradients_reach_both_band_parameters() -> None:
    mask = GaussianBandMask(fps=FPS)
    mask(64, torch.device("cpu"), torch.float32).sum().backward()
    assert mask.theta_fc.grad is not None and mask.theta_fc.grad.abs() > 0
    assert mask.theta_bw.grad is not None and mask.theta_bw.grad.abs() > 0
