"""Signal derivation: the estimators the whole audit rests on."""

from __future__ import annotations

import numpy as np
import pytest

from src.aggregation.signals import (
    br_from_wave,
    detect_beats,
    hr_from_wave,
    hrv_from_wave,
)

FS = 30.0


def synth(bpm: float, seconds: float = 20.0, fs: float = FS, noise: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * fs)) / fs
    wave = np.sin(2 * np.pi * (bpm / 60.0) * t)
    if noise:
        wave = wave + np.random.default_rng(0).normal(0, noise, wave.shape)
    return wave


@pytest.mark.parametrize("bpm", [50, 72, 95, 120, 150])
def test_hr_recovers_known_rate(bpm: float) -> None:
    assert abs(hr_from_wave(synth(bpm), FS) - bpm) < 2.0


def test_hr_survives_noise() -> None:
    # Real traces are far noisier than the pulse; the estimator must not need a
    # clean sinusoid to work.
    assert abs(hr_from_wave(synth(72, noise=0.5), FS) - 72) < 5.0


def test_hr_interpolates_between_fft_bins() -> None:
    """Without parabolic interpolation every estimate snaps to a bin multiple.

    A 10 s window at 30 Hz has 0.125 Hz bins = 7.5 bpm, which quantised HR badly
    enough to make it useless as a regression target.
    """
    estimates = [hr_from_wave(synth(bpm, seconds=10.0), FS) for bpm in (71, 72, 73, 74)]
    on_grid = sum(abs(e / 7.5 - round(e / 7.5)) < 1e-6 for e in estimates)
    assert on_grid <= 1, f"estimates snapped to the 7.5 bpm grid: {estimates}"


def test_br_recovers_breathing_rate() -> None:
    assert abs(br_from_wave(synth(15, seconds=60.0), FS) - 15) < 2.0


def test_flat_signal_returns_none() -> None:
    assert hr_from_wave(np.ones(300), FS) is None


def test_beats_detected_at_expected_count() -> None:
    beats = detect_beats(synth(60, seconds=10.0), FS)
    assert 8 <= len(beats) <= 12          # ~10 beats in 10 s at 60 bpm


def test_hrv_needs_a_long_enough_record() -> None:
    """Short records give unstable SDNN; returning None beats a precise-looking lie."""
    assert hrv_from_wave(synth(72, seconds=5.0), FS) == (None, None)


def test_hrv_returns_values_on_long_record() -> None:
    sdnn, rmssd = hrv_from_wave(synth(72, seconds=40.0), FS)
    assert sdnn is not None and sdnn >= 0
    assert rmssd is not None and rmssd >= 0
