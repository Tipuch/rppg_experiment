"""Spectral audit: the tool that determines whether a dataset is usable at all.

It has to reject a spectrum with no cardiac peak. Getting that wrong in the
permissive direction would wave through a corpus with no signal -- which is the
mistake that cost this project a day of training runs.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.model.audit import BAND_EDGE_BPM, analyse_trace

FS = 30.0


def clean_pulse(bpm: float, seconds: float = 10.0) -> np.ndarray:
    t = np.arange(int(seconds * FS)) / FS
    return np.sin(2 * np.pi * (bpm / 60.0) * t)


def drift_only(seconds: float = 10.0) -> np.ndarray:
    """1/f drift with no cardiac component -- what an auto-exposure ramp looks like."""
    rng = np.random.default_rng(0)
    n = int(seconds * FS)
    white = rng.normal(0, 1, n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, 1 / FS)
    spectrum[1:] /= freqs[1:] ** 1.5
    return np.fft.irfft(spectrum, n)


@pytest.mark.parametrize("bpm", [55, 72, 95, 130])
def test_clean_pulse_is_detected(bpm: float) -> None:
    r = analyse_trace(clean_pulse(bpm), bpm)
    assert r["has_pulse"]
    assert not r["band_edge"]
    assert r["abs_err"] < 10.0


def test_pure_drift_is_rejected() -> None:
    """A monotonically falling spectrum peaks at the bottom of the search band.
    That is the fingerprint of no signal, not of a 42 bpm heart rate."""
    r = analyse_trace(drift_only(), 72.0)
    assert r["band_edge"]
    assert not r["has_pulse"]
    assert r["peak_bpm"] <= BAND_EDGE_BPM + 1.0


def test_band_edge_is_rejected_even_when_prominent() -> None:
    """58 real clips had prominence >= 50 while pinned to the band edge. Prominence
    alone must never be enough to pass."""
    trace = drift_only() * 100.0          # very prominent, still no peak
    r = analyse_trace(trace, 72.0)
    assert r["band_edge"] and not r["has_pulse"]


def test_pulse_at_wrong_rate_is_rejected() -> None:
    """A real peak that conflicts with the label is not a usable clip."""
    r = analyse_trace(clean_pulse(60.0), 120.0)
    assert not r["has_pulse"]
    assert r["abs_err"] > 10.0


def test_flat_trace_is_rejected() -> None:
    r = analyse_trace(np.ones(300), 72.0)
    assert not r["has_pulse"]


def test_short_trace_is_rejected() -> None:
    r = analyse_trace(clean_pulse(72.0, seconds=1.0)[:32], 72.0)
    assert not r["has_pulse"]
