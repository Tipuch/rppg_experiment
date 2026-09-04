"""Post-processing: does a known tone come back as the rate it went in as?

Every heart-rate number this project reports passes through here, so an error is
an error in every result at once. The synthetic cases below are the only place it
can be checked against an answer that is known rather than estimated.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from src.model.msptd import msptd
from src.model.postprocess import (
    BUTTER_ORDER,
    DETECTION_HIGH_HZ,
    DETECTION_LOW_HZ,
    DETECTION_ORDER,
    HIGH_HZ,
    HRV_MIN_SECONDS,
    LOW_HZ,
    RESP_HIGH_HZ,
    RESP_LOW_HZ,
    RESP_MIN_SECONDS,
    align_to_peaks,
    bandpass,
    beats,
    compare,
    detection_bandpass,
    heart_rate,
    hrv,
    interval_hr,
    refine,
    reported_hr,
    respiratory_rate,
    snr,
    spectral_hr,
    template_match,
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


@pytest.mark.parametrize("ratio", [0.2, 0.4, 0.6, 0.8, 1.0])
def test_interval_hr_reads_the_fundamental_through_a_harmonic(ratio: float) -> None:
    """A real pulse has a second harmonic, and beat timing must not follow it.

    The detection band runs to 8 Hz, so unlike the reporting band it does not simply
    delete the harmonic -- it has to survive it. MSPTD does, up to a harmonic as
    strong as the fundamental.
    """
    t = np.arange(300) / FPS
    wave = np.sin(2 * math.pi * 1.2 * t) + ratio * np.sin(2 * math.pi * 2.4 * t)
    assert interval_hr(wave, FPS) == pytest.approx(72.0, abs=1.0)


def test_a_harmonic_stronger_than_the_fundamental_is_read_as_the_rate() -> None:
    """The cost of detecting beats in 0.5-8 Hz instead of 0.75-2.5 Hz, stated.

    At 1.4x the fundamental the second harmonic is the largest oscillation in the
    signal, and MSPTD returns its rate. The old narrow-band detector read 72 bpm here
    for a reason that was not robustness -- 2.4 Hz is outside 0.75-2.5 Hz only just,
    and the filter removed the harmonic rather than the detector rejecting it.

    A pulse whose second harmonic exceeds its fundamental is not a pulse shape, so
    this is a limit worth having rather than one worth patching. The spectral member
    of `reported_hr` fails the same way on the same signal, which is why the vote
    cannot rescue it either.
    """
    assert interval_hr(_tone_with_harmonic(), FPS) > 120.0
    assert heart_rate(_tone_with_harmonic(), FPS) == pytest.approx(70.31, abs=0.5)


def test_compare_reports_the_same_readout_everything_else_does() -> None:
    """`compare` feeds every evaluation table, so it has to read the rate the way
    `predict` does or the tables and the plots would quote different numbers for the
    same window. Both sides go through `reported_hr`: the middle of the spectral
    peak, the median inter-beat interval and the mean inter-beat interval, which over
    1569 labelled test windows beat all three of its members at once (MAE 3.29,
    RMSE 6.23, rho 0.870). README.md records the sweep."""
    wave = _pulse_with_dicrotic_notch(bpm=72.0, seconds=20.0)
    result = compare(wave, wave, FPS)
    assert result["hr_pred"] == pytest.approx(reported_hr(wave, FPS), abs=1e-9)
    assert result["hr_true"] == pytest.approx(reported_hr(wave, FPS), abs=1e-9)
    assert result["hr_pred"] == pytest.approx(72.0, abs=2.0)


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
    at: float = 0.4, fps: float = FPS,
) -> np.ndarray:
    """A pulse whose diastolic decay carries a second bump, as a real one does.

    `notch` is the bump's height as a fraction of the systolic peak and `at` its
    position within the cycle. Both are in the range a contact PPG shows: the
    dicrotic notch is a closing aortic valve, not an artefact, and it grows and
    shrinks within one recording as perfusion changes. 0.4 of the cycle is where one
    falls; `tests/test_msptd.py` covers 0.5, which is degenerate for any scale-based
    detector and is tested there as a stated limit rather than used as the default.

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


def _pulse_with_one_extra_beat(
    bpm: float = 72.0, seconds: float = 10.0, extra: float = 0.6, fps: float = FPS,
) -> np.ndarray:
    """A clean pulse train with a single spurious peak inside one cycle.

    One extra peak, not the periodic notch `_pulse_with_dicrotic_notch` builds. The
    notch guard cannot repair this one: it fires on beat timing reading 1.4x the
    spectrum, and one bad interval in eleven does not move the median at all, so the
    guard never sees a disagreement. This is the case that separates the two
    aggregates -- the median is defined by the ten good intervals and the mean is not.
    """
    period = 60.0 / bpm * fps
    t = np.arange(int(seconds * fps))
    phase = np.mod(t, period) / period
    systolic = np.exp(-((phase - 0.12) ** 2) / (2 * 0.055**2))
    # Half a cycle past the sixth systolic peak: clear of the 12-frame spacing floor,
    # and wide enough to survive the band-pass.
    centre = 5 * period + 0.12 * period + period / 2
    spurious = extra * np.exp(-((t - centre) ** 2) / (2 * (0.055 * period) ** 2))
    return systolic + spurious


@pytest.mark.parametrize("bpm", [50.0, 72.0, 96.0, 145.0])
def test_the_mean_aggregate_reads_a_clean_tone_back(bpm: float) -> None:
    """The aggregate is a choice between two estimators of the same quantity. On a
    trace with no bad interval they must agree, or the comparison downstream is
    measuring something other than robustness."""
    clean = bandpass(_tone(bpm), FPS)
    assert interval_hr(clean, FPS, filtered=True, aggregate=np.mean) == pytest.approx(
        bpm, abs=2.0
    )


def test_one_spurious_beat_moves_the_mean_and_not_the_median() -> None:
    """The whole of the difference between the two aggregates, as a number.

    Eleven intervals at 72 bpm, one of them split in two by a spurious peak. The
    median is the sixth of twelve sorted intervals and is still a whole cycle; the
    mean loses a twelfth of its total, which is 6 bpm at this rate.
    """
    trace = _pulse_with_one_extra_beat(bpm=72.0)
    assert interval_hr(trace, FPS) == pytest.approx(72.0, abs=1.0)
    assert interval_hr(trace, FPS, aggregate=np.mean) > 76.0


def test_the_aggregate_does_not_change_which_peaks_are_counted() -> None:
    """The double-beat guard inside `beats` stays on the median whatever the readout
    aggregates with. It is a detector, not a readout: a guard whose threshold moved
    with the caller's choice would repair a different set of windows per variant and
    the sweep would no longer be comparing readouts on identical peaks."""
    trace = bandpass(_pulse_with_dicrotic_notch(bpm=66.0), FPS)
    assert interval_hr(trace, FPS, filtered=True, aggregate=np.mean) == pytest.approx(
        66.0, abs=4.0
    )


def _beat_train(
    beat_times: np.ndarray, seconds: float, fps: float = FPS
) -> np.ndarray:
    """A wave with one peak at each of `beat_times`, in seconds.

    Built by interpolating phase between the beat times and taking its cosine, so
    every peak sits exactly on a beat and the wave carries no harmonics. A train of
    narrow bumps would be simpler to write and wrong to use: its second harmonic
    falls at 2.4 Hz, inside the 0.75-2.5 Hz band, and the band-pass rings it into a
    second maximum between beats. That splits every long interval in two and reads a
    70 bpm train back as 134 bpm.
    """
    grid = np.arange(int(seconds * fps)) / fps
    phase = np.interp(grid, beat_times, 2 * np.pi * np.arange(len(beat_times)))
    return np.cos(phase)


def _respiratory_modulated_train(
    seconds: float = 60.0, bpm: float = 72.0, brpm: float = 15.0,
    depth: float = 0.08, fps: float = FPS,
) -> np.ndarray:
    """A pulse whose rate rises and falls once per breath: respiratory sinus arrhythmia.

    RSA is the mechanism a respiratory rate is read through here. The pulse itself
    carries no respiratory component in this signal, only its timing does, which is
    the case that matters: the model's DF-FFN mask is constrained to 0.75-2.5 Hz, so
    a breath at 0.25 Hz cannot reach the output as amplitude.
    """
    interval = 60.0 / bpm
    times: list[float] = []
    t = 0.0
    while t < seconds:
        times.append(t)
        t += interval / (1.0 + depth * math.sin(2 * math.pi * (brpm / 60.0) * t))
    return _beat_train(np.array(times), seconds, fps)


def test_a_metronomic_pulse_has_no_variability() -> None:
    """Constant intervals, so both dispersion measures are zero. Anything else is
    the readout's own jitter being reported as the subject's."""
    result = hrv(bandpass(_tone(72.0, seconds=90.0), FPS), FPS, filtered=True)
    assert result["sdnn_ms"] == pytest.approx(0.0, abs=6.0)
    assert result["rmssd_ms"] == pytest.approx(0.0, abs=8.0)


def test_the_mean_interval_agrees_with_the_reported_rate() -> None:
    """mean_nn and the heart rate are one quantity in two units. A discrepancy means
    the two are counting different peaks."""
    trace = bandpass(_tone(72.0, seconds=90.0), FPS)
    result = hrv(trace, FPS, filtered=True)
    assert result["mean_nn_ms"] == pytest.approx(60_000.0 / 72.0, abs=25.0)


def _alternating_train() -> np.ndarray:
    """800 and 900 ms intervals alternating. Every successive difference is 100 ms,
    so the true RMSSD is 100 ms, the true SDNN is 50 ms and pNN50 is 1.0."""
    # Long enough to clear HRV_MIN_SECONDS: 80 intervals is about 68 s.
    intervals = np.array([0.8, 0.9] * 40)
    times = np.concatenate([[1.0], 1.0 + np.cumsum(intervals)])
    return _beat_train(times, seconds=times[-1] + 1.0)


def test_a_known_interval_series_reads_back_as_itself() -> None:
    """The readout itself, measured on the waveform it was given."""
    result = hrv(_alternating_train(), FPS, filtered=True)
    assert result["rmssd_ms"] == pytest.approx(100.0, abs=10.0)
    assert result["sdnn_ms"] == pytest.approx(50.0, abs=5.0)
    assert result["pnn50"] == pytest.approx(1.0, abs=0.05)


def test_the_cardiac_band_pass_shrinks_the_variability_it_passes() -> None:
    """A caveat, pinned as a number so it cannot drift unnoticed.

    Beat-to-beat variability is itself a modulation, and an alternation between two
    intervals sits near 0.6 Hz -- below the 0.75 Hz corner every reported waveform is
    filtered through. The filter attenuates it. The same 100 ms alternation reads
    92 ms on the raw wave and 64 ms once band-passed, a third of it gone, and the
    bias is toward a calmer subject than the recording holds.

    Nothing here corrects for it: the correction depends on where in the band the
    subject's own variability sits, which is not known per window. The test states
    the direction and the rough size so a caller can read `hrv` knowing both.
    """
    raw = hrv(_alternating_train(), FPS, filtered=True)["rmssd_ms"]
    filtered = hrv(bandpass(_alternating_train(), FPS), FPS, filtered=True)["rmssd_ms"]
    assert filtered < 0.8 * raw
    assert filtered == pytest.approx(64.0, abs=10.0)


def test_variability_is_not_reported_from_too_few_beats() -> None:
    """RMSSD needs two successive differences to be a root mean square of anything.
    A single interval is a heart rate, not a variability."""
    result = hrv(bandpass(_tone(72.0, seconds=2.0), FPS), FPS, filtered=True)
    assert math.isnan(result["rmssd_ms"])
    assert math.isnan(result["sdnn_ms"])


def test_the_respiratory_band_is_six_to_thirty_breaths_a_minute() -> None:
    assert (RESP_LOW_HZ * 60.0, RESP_HIGH_HZ * 60.0) == (6.0, 30.0)


def test_a_breath_is_read_off_the_beat_timing() -> None:
    """15 breaths a minute, present only as respiratory sinus arrhythmia."""
    trace = bandpass(_respiratory_modulated_train(seconds=60.0, brpm=15.0), FPS)
    result = respiratory_rate(trace, FPS, filtered=True)
    assert result["brpm_rsa"] == pytest.approx(15.0, abs=1.5)


@pytest.mark.parametrize("brpm", [10.0, 20.0])
def test_the_breath_rate_read_back_is_the_one_that_went_in(brpm: float) -> None:
    trace = bandpass(_respiratory_modulated_train(seconds=90.0, brpm=brpm), FPS)
    assert respiratory_rate(trace, FPS, filtered=True)["brpm_rsa"] == pytest.approx(
        brpm, abs=2.0
    )


def test_a_short_window_returns_no_breath_rate_rather_than_a_guess() -> None:
    """The slowest breath in the band is one per 10 s. A 10 s window holds one cycle
    of it, which is not enough to place a peak, and the periodogram would return the
    bottom of the band whatever the subject was doing. This is the length `predict`
    runs by default, so the gate is what stops a fabricated number reaching the
    plot."""
    trace = bandpass(_respiratory_modulated_train(seconds=10.0, brpm=15.0), FPS)
    result = respiratory_rate(trace, FPS, filtered=True)
    assert math.isnan(result["brpm_rsa"])
    assert math.isnan(result["brpm"])
    assert result["seconds"] < RESP_MIN_SECONDS


def test_a_clean_pulse_matches_its_own_template() -> None:
    """TMCC is each beat correlated against the average beat. On a signal whose beats
    are identical every correlation is 1."""
    assert template_match(bandpass(_tone(72.0, 20.0), FPS), FPS, filtered=True) == \
        pytest.approx(1.0, abs=0.02)


def test_template_matching_separates_a_noisy_pulse_from_a_clean_one() -> None:
    """The property the metric exists for. Charlton et al. 2025 Fig 4 shows SNR
    failing at this: a high-quality trace with an irregular rhythm scores 0.0 dB and
    a noisy medium-quality one scores 8.2 dB. TMCC tracks the pulse wave instead of
    the spectrum, which is why it is the gate used here."""
    rng = np.random.default_rng(0)
    clean = _tone(72.0, 20.0)
    noisy = clean + 1.5 * rng.standard_normal(clean.size)
    assert template_match(bandpass(noisy, FPS), FPS, filtered=True) < \
        template_match(bandpass(clean, FPS), FPS, filtered=True)


def test_template_matching_returns_nan_when_there_are_too_few_beats() -> None:
    """A template averaged from one beat is that beat, and correlating a beat with
    itself returns 1 whatever its quality."""
    assert math.isnan(template_match(bandpass(_tone(72.0, 1.5), FPS), FPS, filtered=True))
    assert math.isnan(template_match(np.zeros(300), FPS))


def test_template_matching_does_not_separate_a_prediction_from_noise() -> None:
    """Why there is no TMCC floor, pinned so nobody reintroduces one.

    Over the 1569 windows of the test dump, template-matching correlation through the
    detection band has a median of 0.853 on predicted waveforms and 0.829 on white
    noise. The distributions overlap across their whole range. Any threshold that
    admits most predictions also admits most noise, so gating on it would exclude
    nothing while reading as though it had.

    The metric is not broken -- on contact PPG the median is 0.949, which is the
    signal Charlton et al. 2025 measured it on. It is the predicted waveform it cannot
    grade.
    """
    rng = np.random.default_rng(1)
    noise = np.array(
        [template_match(rng.standard_normal(300), FPS) for _ in range(60)]
    )
    tone = np.array(
        [template_match(_tone(72.0, 10.0) + 0.6 * rng.standard_normal(300), FPS)
         for _ in range(60)]
    )
    assert np.median(tone) - np.median(np.array(noise)) < 0.25


def test_variability_is_reported_with_its_quality_beside_it() -> None:
    """`tmcc` is returned so a reader can judge the numbers next to it, and gates
    nothing. Noise long enough to clear the duration gate still gets a variability,
    which is the honest behaviour given the distributions above: the alternative was
    a floor that let the same noise through while implying it had not."""
    rng = np.random.default_rng(1)
    result = hrv(rng.standard_normal(2700), FPS)
    assert np.isfinite(result["tmcc"])
    assert np.isfinite(result["rmssd_ms"])


def test_the_narrow_band_inflates_the_quality_score() -> None:
    """Why `template_match` filters in the detection band rather than reusing
    `bandpass`. A 0.75-2.5 Hz filter shapes white noise into a near-sinusoid, and a
    near-sinusoid has beats that all match each other, so the score rises toward 1
    for a signal with no beats in it at all."""
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(900)
    assert (
        template_match(bandpass(noise, FPS), FPS, filtered=True)
        > template_match(noise, FPS)
    )


def test_the_reported_readout_is_the_middle_of_three() -> None:
    """`reported_hr` returns the median of the spectral peak and the two interval
    aggregates, so it always equals one of them and never a value none of them gave."""
    wave = bandpass(_tone(84.0, 20.0), FPS)
    members = [
        spectral_hr(wave, FPS, filtered=True, pad=8, window="boxcar"),
        interval_hr(wave, FPS, filtered=True),
        interval_hr(wave, FPS, filtered=True, aggregate=np.mean),
    ]
    assert reported_hr(wave, FPS, filtered=True) == pytest.approx(
        float(np.median(members)), abs=1e-9
    )


@pytest.mark.parametrize("bpm", [50.0, 72.0, 96.0, 145.0])
def test_the_reported_readout_reads_a_clean_tone_back(bpm: float) -> None:
    assert reported_hr(_tone(bpm, 20.0), FPS) == pytest.approx(bpm, abs=2.0)


def test_the_reported_readout_discards_the_member_that_is_furthest_out() -> None:
    """Why the median of three and not the mean of three. A stronger second harmonic
    pulls the spectral peak off the fundamental; beat timing does not follow it, so
    the middle value is a good one and the mean of all three would not be."""
    wave = bandpass(_tone_with_harmonic(), FPS)
    assert reported_hr(wave, FPS, filtered=True) == pytest.approx(72.0, abs=2.0)


def test_the_reported_readout_survives_a_member_returning_nothing() -> None:
    """A combination that failed whenever one member failed would be scored on an
    easier set of windows than its members."""
    wave = bandpass(_tone(72.0, 20.0), FPS)
    with_gap = np.concatenate([wave[:300], np.zeros(60), wave[360:]])
    assert np.isfinite(reported_hr(with_gap, FPS, filtered=True))


def test_variability_needs_a_minute_of_recording() -> None:
    """RMSSD over a dozen intervals is dominated by whichever one was miscounted.
    30 s of a clean tone is withheld; 90 s is not."""
    assert math.isnan(hrv(_tone(72.0, 30.0), FPS)["rmssd_ms"])
    assert np.isfinite(hrv(_tone(72.0, 90.0), FPS)["rmssd_ms"])
    assert HRV_MIN_SECONDS == 60.0


def test_beats_are_found_by_msptd() -> None:
    """`beats` is a thin wrapper: it picks the band and delegates. If the two ever
    diverge, a rate and a beat plot would disagree about where the beats were."""
    trace = detection_bandpass(_pulse_with_dicrotic_notch(bpm=72.0, seconds=20.0), FPS)
    assert np.array_equal(beats(trace, FPS, filtered=True), msptd(trace, FPS)[0])


def test_the_detection_band_is_wider_than_the_reporting_band() -> None:
    """Beats are found in 0.75-4 Hz, rates read in 0.75-2.5. The gap is the point:
    harmonics are what make a pulse a shape, and 4 Hz is 240 bpm where the reporting
    band stops answering at 150.

    The edges are swept values, recorded beside the constants, not the 0.5-8 Hz
    Charlton et al. 2022 use. This pins them so a change has to be a decision.
    """
    assert (DETECTION_LOW_HZ, DETECTION_HIGH_HZ, DETECTION_ORDER) == (0.75, 4.0, 4)
    assert DETECTION_HIGH_HZ > HIGH_HZ
    assert DETECTION_LOW_HZ == LOW_HZ
    assert DETECTION_HIGH_HZ * 60.0 == 240.0


def test_the_detection_band_is_linearly_detrended() -> None:
    """Not with the toolbox's smoothness prior, which `bandpass` uses.

    The prior is a high-pass whose corner moves with the signal length, and at 45 bpm
    it lands on the pulse: MSPTD reads 22 beats from a 15-cycle 45 bpm trace detrended
    that way, and 14 from the same trace detrended linearly. ppg-beats removes a
    straight line before its own scalogram, and so does this.
    """
    slow = _pulse_with_dicrotic_notch(bpm=45.0, seconds=20.0)
    assert len(beats(slow, FPS)) == pytest.approx(15, abs=1.5)


@pytest.mark.parametrize("bpm", [45.0, 66.0, 95.0, 140.0])
def test_the_rate_survives_a_notch_across_the_whole_band(bpm: float) -> None:
    """The regression the swap exists for, at the level callers see it.

    The old detector was `find_peaks` with a 12-frame spacing floor and a conditional
    prominence guard. On this signal it read 93 bpm for a 45 bpm pulse, because 0.75 Hz
    is 45 bpm and its band-pass corner sat on the trace.
    """
    assert interval_hr(_pulse_with_dicrotic_notch(bpm=bpm, seconds=20.0), FPS) == \
        pytest.approx(bpm, abs=3.0)


def test_the_notch_guard_is_gone() -> None:
    """MSPTD separates a notch from a beat by scale, so the prominence guard that
    patched `find_peaks` has no job left. Leaving a dead threshold in the module
    would invite someone to tune it."""
    import src.model.postprocess as module
    assert not hasattr(module, "DOUBLE_BEAT_RATIO")
    assert not hasattr(module, "NOTCH_PROMINENCE_FRAC")


def test_the_reported_readout_reads_each_member_in_its_own_band() -> None:
    """`reported_hr` builds two filtered copies rather than one. Handing the interval
    members a 0.75-2.5 Hz trace would strip the harmonics MSPTD needs; handing the
    spectral member a 0.5-8 Hz trace would put a second harmonic in its search band.
    """
    wave = _pulse_with_dicrotic_notch(bpm=72.0, seconds=20.0)
    assert reported_hr(wave, FPS) == pytest.approx(72.0, abs=2.0)
    # Pre-filtering to one band is what `filtered=True` means, and it is measurably
    # worse on this signal -- the point of the flag is legacy call sites, not parity.
    assert reported_hr(wave, FPS) != reported_hr(bandpass(wave, FPS), FPS, filtered=True)


def test_detrend_matches_the_vendored_toolbox_exactly() -> None:
    """`detrend` is solved rather than inverted, and must give the same answer.

    The vendored `_detrend` forms a dense N x N matrix and inverts it. The matrix is
    pentadiagonal and symmetric positive definite, so the same result comes from a
    banded solve in O(N) instead of O(N^3) -- 341 ms per 300-sample window here
    against 0.2 ms, because this machine's LAPACK inverts a 300 x 300 in 93 ms.

    Every SNR and MACC number this project reports against the rPPG-Toolbox tables
    passes through it, so "the same answer" has to be checked rather than assumed.
    """
    from src.model.postprocess import _POST, detrend
    rng = np.random.default_rng(0)
    for length in (64, 181, 300, 901):
        wave = np.cumsum(rng.standard_normal(length)) + _tone(72.0, length / FPS)[:length]
        assert np.allclose(
            detrend(wave, 100), _POST._detrend(wave, 100), rtol=0, atol=1e-8
        )


def test_detrend_returns_the_shape_it_was_given() -> None:
    """`baselines` injects this into the vendored POS implementation as
    `utils.detrend`. POS hands it an N x 1 matrix and transposes the result, so a
    flattened return collapses to a scalar and `filtfilt` raises on it."""
    from src.model.postprocess import detrend
    column = _tone(72.0, 10.0).reshape(-1, 1)
    assert detrend(column, 100).shape == column.shape
    assert detrend(column.ravel(), 100).shape == (column.size,)


def test_detrend_propagates_non_finite_input_rather_than_raising() -> None:
    """What the dense inverse did, kept. POS divides by the standard deviation of the
    signal, which is zero on a still clip, so a NaN reaches this function on the exact
    input `tests/test_baselines.py` uses to check that a still clip reports no rate."""
    from src.model.postprocess import detrend
    wave = _tone(72.0, 10.0)
    wave[17] = np.nan
    assert np.isnan(detrend(wave, 100)).all()


def test_align_to_peaks_moves_a_marker_onto_the_maximum() -> None:
    """The whole job: an index a few samples off lands on the peak."""
    wave = bandpass(_tone(72.0, 20.0), FPS)
    true_peaks = beats(wave, FPS, filtered=True)
    nudged = true_peaks + 2
    assert np.array_equal(align_to_peaks(wave, nudged, FPS), true_peaks)


def test_align_to_peaks_returns_indices_that_are_local_maxima() -> None:
    """A caller indexes the trace with these, so every one has to be a real sample
    and a real maximum -- not an interpolated position, which is what `refine` gives."""
    wave = bandpass(_tone(84.0, 20.0), FPS)
    aligned = align_to_peaks(wave, beats(wave, FPS, filtered=True) - 1, FPS)
    inner = aligned[(aligned > 0) & (aligned < wave.size - 1)]
    assert aligned.dtype == np.dtype(int)
    assert np.all(wave[inner] >= wave[inner - 1])
    assert np.all(wave[inner] >= wave[inner + 1])


def test_align_to_peaks_will_not_move_a_marker_onto_the_next_beat() -> None:
    """The bound, and the reason for it. The radius is half the shortest beat interval
    the detection band admits, so a marker cannot be walked to its neighbour however
    wrong it is. Here every marker is placed midway between two beats: each has a
    maximum roughly half an interval away on both sides, further than the radius, so
    all of them are left where they are."""
    wave = bandpass(_tone(72.0, 20.0), FPS)
    found = beats(wave, FPS, filtered=True)
    midpoints = ((found[:-1] + found[1:]) // 2).astype(int)
    assert np.array_equal(align_to_peaks(wave, midpoints, FPS), midpoints)


def test_align_to_peaks_handles_nothing_to_align() -> None:
    flat = np.zeros(300)
    assert align_to_peaks(flat, np.array([10, 20]), FPS).tolist() == [10, 20]
    assert align_to_peaks(bandpass(_tone(72.0, 10.0), FPS), np.empty(0, int), FPS).size == 0


def test_peak_interpolation_does_not_divide_by_zero() -> None:
    """`_vertex_shift` guards its result with `np.where`, which does not stop numpy
    evaluating the division for every element first. Three equal samples make the
    denominator zero, and a flat stretch of a band-passed trace produces them: two of
    the HRV tests in this file emitted `invalid value encountered in divide` from a
    real call path before this was masked."""
    from src.model.postprocess import _vertex_shift
    flat = np.zeros(5)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.all(_vertex_shift(flat, flat, flat) == 0.0)
        assert np.all(np.isfinite(_vertex_shift(flat, flat, flat)))


def test_refine_leaves_a_flat_peak_where_it_is() -> None:
    """The behaviour the mask has to preserve: no curvature means no vertex to move
    to, so the index stands."""
    trace = np.zeros(300)
    trace[100:103] = 1.0
    assert refine(trace, np.array([101]))[0] == pytest.approx(101.0)
