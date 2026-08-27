"""Physiology-aware temporal-spectral FFN, CFMamba Eqs. 13-18."""

from __future__ import annotations

import math

import pytest
import torch

from src.model.cfmamba.pts_ffn import MODES, PhysiologyTemporalSpectralFFN

FPS = 30.0


def _tone(n_frames: int, bpm: float, channels: int = 1) -> torch.Tensor:
    t = torch.arange(n_frames, dtype=torch.float32) / FPS
    return torch.sin(2 * math.pi * (bpm / 60.0) * t).view(1, n_frames, 1).repeat(
        1, 1, channels
    )


def test_the_gain_phase_is_odd_in_frequency() -> None:
    """A real filter obeys H(-f) = conj(H(f)); an even phase is a dead parameter.

    With an even imaginary part the whole imaginary-gain contribution lands in the
    imaginary half of the inverse transform and `.real` discards it, so `gain_im`
    receives exactly zero gradient for the entire run while everything looks fine.
    This assertion is what makes that loud.
    """
    ffn = PhysiologyTemporalSpectralFFN(hidden=4, fps=FPS, mode="diagonal")
    with torch.no_grad():
        ffn.gain_im.normal_()
    freqs = torch.fft.fftfreq(64, d=1.0 / FPS)
    gain = ffn.frequency_gain(freqs)
    assert torch.allclose(gain[1:], torch.conj(gain.flip(0)[:-1]), atol=1e-6)


@pytest.mark.parametrize("n_frames", [63, 64])
def test_self_conjugate_bins_stay_real(n_frames: int) -> None:
    """DC always, and Nyquist whenever T is even: both are their own conjugate
    partner, so a complex gain there is silently discarded."""
    ffn = PhysiologyTemporalSpectralFFN(hidden=4, fps=FPS, mode="diagonal")
    with torch.no_grad():
        ffn.gain_im.normal_()
    freqs = torch.fft.fftfreq(n_frames, d=1.0 / FPS)
    with torch.no_grad():
        gain = ffn.frequency_gain(freqs)
    assert float(gain[0].imag.abs()) < 1e-6
    if n_frames % 2 == 0:
        assert float(gain[n_frames // 2].imag.abs()) < 1e-6


def test_a_narrow_in_band_peak_is_not_sampled_into_oblivion() -> None:
    """The second ported regression from the deleted test_head.py.

    Its ancestor point-sampled a spectrum while downsampling it, and a pure
    sinusoid at an odd bin index came out as 1e-15. The gain curve here is
    interpolated rather than pooled, which is correct only because it samples a
    smooth function -- so assert the tone still comes out where it went in.
    """
    ffn = PhysiologyTemporalSpectralFFN(hidden=1, fps=FPS, mode="diagonal")
    freqs = torch.fft.rfftfreq(160, d=1.0 / FPS)
    for bpm in (60.0, 72.0, 97.5, 138.0):
        with torch.no_grad():
            out = ffn(_tone(160, bpm))[0, :, 0]
        peak_bpm = float(freqs[int(torch.fft.rfft(out - out.mean()).abs().argmax())]) * 60
        assert peak_bpm == pytest.approx(bpm, abs=8.0), bpm


def test_out_of_band_energy_is_attenuated_relative_to_in_band() -> None:
    """The whole point of Eq. 16: a slow illumination drift must lose to a pulse."""
    ffn = PhysiologyTemporalSpectralFFN(hidden=1, fps=FPS, mode="none")
    with torch.no_grad():
        drift = ffn(_tone(300, 9.0))                 # 0.15 Hz
        pulse = ffn(_tone(300, 96.0))                # 1.6 Hz, band centre
    assert float(drift.std()) < 0.2 * float(pulse.std())


@pytest.mark.parametrize("mode", MODES)
def test_shape_and_realness_are_preserved(mode: str) -> None:
    ffn = PhysiologyTemporalSpectralFFN(hidden=6, fps=FPS, mode=mode, n_frames=160)
    x = torch.randn(2, 160, 6)
    out = ffn(x)
    assert out.shape == x.shape and not out.is_complex()
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("mode", MODES)
def test_gradients_reach_every_parameter(mode: str) -> None:
    ffn = PhysiologyTemporalSpectralFFN(hidden=6, fps=FPS, mode=mode, n_frames=64)
    ffn(torch.randn(2, 64, 6)).square().mean().backward()
    dead = [n for n, p in ffn.named_parameters() if p.grad is None or not p.grad.any()]
    assert not dead, f"no gradient reached {dead}"


@pytest.mark.parametrize("n_frames", [80, 160, 301])
def test_diagonal_mode_accepts_any_clip_length(n_frames: int) -> None:
    ffn = PhysiologyTemporalSpectralFFN(hidden=4, fps=FPS, mode="diagonal")
    assert ffn(torch.randn(1, n_frames, 4)).shape == (1, n_frames, 4)


def test_full_mode_refuses_a_length_it_was_not_built_for() -> None:
    """Silence here would mean a shape error deep inside a complex matmul."""
    ffn = PhysiologyTemporalSpectralFFN(hidden=4, fps=FPS, mode="full", n_frames=160)
    with pytest.raises(ValueError, match="cannot accept"):
        ffn(torch.randn(1, 300, 4))


def test_an_unknown_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        PhysiologyTemporalSpectralFFN(hidden=4, mode="banded")
