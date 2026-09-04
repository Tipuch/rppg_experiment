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
    beats,
    compare,
    heart_rate,
    interval_hr,
    refine,
    snr,
    spectral_hr,
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


def test_spectral_hr_resolves_a_rate_that_falls_between_bins() -> None:
    """`heart_rate` quantises to the periodogram grid: at 300 frames the FFT is
    padded to 512 and the bins are 3.516 bpm apart, so a rate landing midway
    between two of them cannot be returned. Interpolating the vertex of the peak
    recovers it."""
    between = 68.5      # bins either side are 66.80 and 70.31
    assert spectral_hr(_tone(between), FPS) == pytest.approx(between, abs=0.5)
    assert abs(heart_rate(_tone(between), FPS) - between) > 1.0


@pytest.mark.parametrize("seconds", [5.33, 10.0, 30.0])
def test_spectral_hr_does_not_depend_on_clip_length(seconds: float) -> None:
    """Same requirement `heart_rate` carries: an estimator whose answer moves with
    T is measuring the window. Padding interpolates the grid, it does not shift
    the peak."""
    assert spectral_hr(_tone(72.0, seconds), FPS) == pytest.approx(72.0, abs=1.0)


@pytest.mark.parametrize("window", ["hann", "boxcar"])
def test_spectral_hr_recovers_a_clean_tone_under_either_window(window: str) -> None:
    """The window is a swept parameter, not a settled one -- on real clips Hann
    fixed one and broke another. Both must at least read a clean tone back, so
    changing the default cannot silently break the readout."""
    assert spectral_hr(_tone(96.0), FPS, window=window) == pytest.approx(96.0, abs=1.0)


def test_spectral_hr_is_amplitude_and_offset_invariant() -> None:
    """Nothing fixed the scale of a predicted BVP: Eq. 19's temporal term is
    negative Pearson, which is invariant to a positive scale factor."""
    wave = _tone(84.0)
    assert spectral_hr(wave * 0.001 + 50.0, FPS) == pytest.approx(
        spectral_hr(wave, FPS), abs=1e-6
    )


def test_spectral_hr_returns_nan_on_degenerate_input() -> None:
    """Mirrors `heart_rate`: a dead prediction early in training must not crash an
    evaluation pass."""
    assert math.isnan(spectral_hr(np.zeros(300), FPS))
    assert math.isnan(spectral_hr(np.array([1.0, 2.0]), FPS))
    assert math.isnan(spectral_hr(np.full(300, np.nan), FPS))


def test_heart_rate_still_matches_the_vendored_toolbox() -> None:
    """`spectral_hr` exists alongside `heart_rate` rather than replacing it. Every
    number in README.md and every comparison against the rPPG-Toolbox tables comes
    through `heart_rate`, so it must keep returning the toolbox's own answer: a
    bare argmax over a rectangular periodogram padded to the next power of two."""
    wave = _tone(68.5)
    bins = np.fft.rfftfreq(512, d=1.0 / FPS) * 60.0
    assert heart_rate(wave, FPS) in bins
    assert heart_rate(wave, FPS) == pytest.approx(66.80, abs=0.01)


def _tone_with_harmonic() -> np.ndarray:
    """72 bpm fundamental under a stronger second harmonic -- a real pulse has one.

    The spectral argmax lands on 70.31, the nearest bin to the fundamental. Beat
    timing recovers 72.00, because the harmonic rides on the beats rather than
    displacing them.
    """
    t = np.arange(300) / FPS
    return np.sin(2 * math.pi * 1.2 * t) + 1.4 * np.sin(2 * math.pi * 2.4 * t)


def test_interval_hr_reads_the_fundamental_through_a_stronger_harmonic() -> None:
    assert interval_hr(_tone_with_harmonic(), FPS) == pytest.approx(72.0, abs=0.5)
    assert heart_rate(_tone_with_harmonic(), FPS) == pytest.approx(70.31, abs=0.5)


def test_compare_reports_the_interval_readout() -> None:
    """Swept over 1569 labelled test windows, the interval readout cut RMSE from
    8.18 to 6.60 bpm and raised rho from 0.793 to 0.857 against the spectral
    argmax, at the cost of 0.35 bpm of MAE -- fewer large misses, slightly more
    small error. README.md records the sweep."""
    wave = _tone_with_harmonic()
    result = compare(wave, wave, FPS)
    assert result["hr_pred"] == pytest.approx(interval_hr(wave, FPS), abs=1e-9)
    assert result["hr_true"] == pytest.approx(interval_hr(wave, FPS), abs=1e-9)


def test_beats_never_finds_two_peaks_inside_one_cycle():
    """The spacing floor is the period of 2.5 Hz, the top of the reporting band."""
    trace = _tone(150, 300)
    found = beats(trace)
    assert len(found) >= 2
    assert np.diff(found).min() >= FPS / 2.5 - 1


def test_beats_returns_empty_on_a_degenerate_trace():
    assert len(beats(np.zeros(4))) == 0
    assert len(beats(np.full(300, np.nan))) == 0


def test_refine_locates_a_peak_between_frames() -> None:
    """An integer peak index quantises the interval by one frame, which at 30 fps
    is 3.4 bpm at 110 and 6 bpm at 150. The parabola through the three samples
    around a peak recovers where it actually sits, and the same formula does the
    same job for a spectral peak between bins -- so both call one helper."""
    trace = np.array([0.0, 1.0, 3.0, 2.0, 0.0])
    peaks = np.array([2])
    refined = refine(trace, peaks)
    # Vertex of the parabola through (1, 1.0), (2, 3.0), (3, 2.0).
    assert refined == pytest.approx([2.0 + 0.5 * (1.0 - 2.0) / (1.0 - 6.0 + 2.0)])
    assert -0.5 <= refined[0] - 2.0 <= 0.5


def test_refine_leaves_a_peak_at_the_edge_alone() -> None:
    """A peak on the first or last sample has no left or right neighbour, so there
    is no parabola to take a vertex from."""
    trace = np.array([3.0, 1.0, 0.0, 1.0, 2.0])
    assert refine(trace, np.array([0, 4])) == pytest.approx([0.0, 4.0])


def _pulse_with_dicrotic_notch(
    bpm: float = 66.0, seconds: float = 10.0, notch: float = 0.35,
    at: float = 0.5, fps: float = FPS,
) -> np.ndarray:
    """A pulse whose diastolic decay carries a second bump, as a real one does.

    `notch` is the bump's height as a fraction of the systolic peak and `at` its
    position within the cycle. Both are in the range a contact PPG shows: the
    dicrotic notch is a closing aortic valve, not an artefact, and it grows and
    shrinks within one recording as perfusion changes.

    This is the shape that produced test-split heart rates of 107-118 bpm against
    a 64-68 bpm pulse -- once the bump is a local maximum more than 12 frames from
    the systolic peak, `find_peaks` counts it as a beat, and more than half the
    intervals are then half-cycles.
    """
    period = 60.0 / bpm * fps
    t = np.arange(int(seconds * fps))
    phase = np.mod(t, period) / period
    systolic = np.exp(-((phase - 0.12) ** 2) / (2 * 0.055**2))
    dicrotic = notch * np.exp(-((phase - at) ** 2) / (2 * 0.055**2))
    return systolic + dicrotic


def test_beats_counts_one_peak_per_cycle_through_a_dicrotic_notch() -> None:
    """11 cycles in 10 s at 66 bpm, not 22. A notch counted as a beat halves the
    median interval, which reads as a rate 1.7x the truth."""
    trace = bandpass(_pulse_with_dicrotic_notch(), FPS)
    assert len(beats(trace, FPS)) == pytest.approx(11, abs=1)


def test_interval_hr_reads_the_pulse_rate_not_the_notch_rate() -> None:
    """The MR-NIRP and MCD failure, as a number: 66 bpm read back as 115."""
    trace = _pulse_with_dicrotic_notch(bpm=66.0)
    assert interval_hr(trace, FPS) == pytest.approx(66.0, abs=3.0)


def test_the_notch_guard_leaves_a_clean_trace_untouched() -> None:
    """Regression guard. The guard is a repair, so it must not fire on a trace
    that needs none: 4478 of the 4482 test windows are already consistent, and a
    readout that moves them is a worse readout whatever it does to the other 4."""
    for bpm in (50.0, 72.0, 96.0, 145.0):
        clean = bandpass(_tone(bpm), FPS)
        assert interval_hr(clean, FPS, filtered=True) == pytest.approx(bpm, abs=2.0)


def test_a_genuinely_fast_pulse_is_not_halved() -> None:
    """Regression guard. The guard compares beat timing against the spectrum, so a
    fast rate the spectrum agrees with must survive it."""
    fast = bandpass(_pulse_with_dicrotic_notch(bpm=140.0), FPS)
    assert interval_hr(fast, FPS, filtered=True) == pytest.approx(140.0, abs=5.0)
