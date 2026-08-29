"""Post-processing: does a known tone come back as the rate it went in as?

Every heart-rate number this project reports passes through here, so an error is
an error in every result at once. The synthetic cases below are the only place it
can be checked against an answer that is known rather than estimated.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.model.postprocess import (
    BUTTER_ORDER,
    HIGH_HZ,
    LOW_HZ,
    bandpass,
    compare,
    heart_rate,
    snr,
)

FPS = 30.0


def _tone(bpm: float, seconds: float = 10.0, fps: float = FPS) -> np.ndarray:
    t = np.arange(int(seconds * fps)) / fps
    return np.sin(2 * math.pi * (bpm / 60.0) * t)


@pytest.mark.parametrize("bpm", [50.0, 60.0, 72.0, 96.0, 120.0, 145.0])
def test_a_synthetic_tone_reads_back_as_itself(bpm: float) -> None:
    assert heart_rate(_tone(bpm), FPS) == pytest.approx(bpm, abs=2.0)


@pytest.mark.parametrize("seconds", [5.33, 10.0, 30.0])
def test_the_answer_does_not_depend_on_clip_length(seconds: float) -> None:
    """A rate estimator whose answer moves with T is measuring the window."""
    assert heart_rate(_tone(72.0, seconds), FPS) == pytest.approx(72.0, abs=3.0)


def test_the_band_matches_the_papers_not_the_toolbox_default() -> None:
    """The vendored wrapper hardcodes 1st-order 0.6-3.3 Hz; both papers specify
    2nd-order 0.75-2.5 Hz. Inheriting the default would admit a 36 bpm drift and a
    198 bpm harmonic into every estimate."""
    assert (LOW_HZ, HIGH_HZ, BUTTER_ORDER) == (0.75, 2.5, 2)


def test_a_slow_drift_is_removed_rather_than_reported_as_a_pulse() -> None:
    """The failure that made 76% of MCD-rPPG look peakless: a 1/f slope with no
    cardiac peak, whose argmax fixes to the bottom of the search band."""
    t = np.arange(300) / FPS
    drift = 5.0 * np.sin(2 * math.pi * 0.1 * t)
    pulse = _tone(72.0)
    assert heart_rate(drift + pulse, FPS) == pytest.approx(72.0, abs=3.0)
    assert bandpass(drift, FPS).std() < 0.15 * drift.std()


def test_a_high_frequency_component_is_removed() -> None:
    t = np.arange(300) / FPS
    assert heart_rate(_tone(72.0) + 3.0 * np.sin(2 * math.pi * 8.0 * t), FPS) == \
        pytest.approx(72.0, abs=3.0)


def test_snr_prefers_a_clean_tone_to_a_noisy_one() -> None:
    rng = np.random.default_rng(0)
    clean = _tone(72.0, 20.0)
    noisy = clean + 3.0 * rng.standard_normal(clean.size)
    assert snr(clean, 72.0, FPS) > snr(noisy, 72.0, FPS)


def test_snr_counts_the_second_harmonic_as_signal() -> None:
    """CFMamba Eq. 27 sums power around f_gt and 2*f_gt. A real pulse has a
    dicrotic notch, so its harmonic is signal; scoring it as noise would penalise
    the predictions that got the waveform shape right."""
    t = np.arange(600) / FPS
    fundamental = np.sin(2 * math.pi * 1.0 * t)
    with_harmonic = fundamental + 0.4 * np.sin(2 * math.pi * 2.0 * t)
    assert snr(with_harmonic, 60.0, FPS) > snr(fundamental + 0.4 *
                                               np.sin(2 * math.pi * 1.6 * t), 60.0, FPS)


def test_compare_reads_the_truth_rate_off_the_waveform() -> None:
    """Not off a label column. Five UBFC subjects have a broken HR readout and an
    intact waveform (DATASETS.md), so the signal is the more trustworthy source."""
    result = compare(_tone(80.0, 20.0), _tone(80.0, 20.0), FPS)
    assert result["hr_true"] == pytest.approx(80.0, abs=2.0)
    assert result["hr_pred"] == pytest.approx(80.0, abs=2.0)
    assert result["macc"] > 0.9


def test_compare_separates_a_wrong_prediction_from_a_right_one() -> None:
    right = compare(_tone(80.0, 20.0), _tone(80.0, 20.0), FPS)
    wrong = compare(_tone(120.0, 20.0), _tone(80.0, 20.0), FPS)
    assert abs(right["hr_pred"] - right["hr_true"]) < 3.0
    assert abs(wrong["hr_pred"] - wrong["hr_true"]) > 30.0
    assert wrong["macc"] < right["macc"]


def test_degenerate_input_returns_nan_rather_than_raising() -> None:
    """A dead prediction early in training must not crash an evaluation pass."""
    assert math.isnan(heart_rate(np.zeros(300), FPS))
    assert math.isnan(heart_rate(np.array([1.0, 2.0]), FPS))
    assert math.isnan(heart_rate(np.full(300, np.nan), FPS))
