"""MSPTD, ported from ppg-beats: does it find one beat per cycle where find_peaks does not?

The port is checked two ways. Against known signals, where the beat positions are
arithmetic rather than estimated; and against `postprocess.beats`, on the shape that
readout exists to survive -- a pulse carrying a dicrotic notch.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy import signal as sps

from src.model.msptd import (
    MSPTD_PLAUSIBLE_BPM,
    MSPTD_WINDOW_S,
    beat_table,
    msptd,
)

FPS = 30.0


def _tone(bpm: float, seconds: float = 10.0, fps: float = FPS) -> np.ndarray:
    t = np.arange(int(seconds * fps)) / fps
    return np.sin(2 * math.pi * (bpm / 60.0) * t)


def _pulse_with_dicrotic_notch(
    bpm: float = 66.0, seconds: float = 10.0, notch: float = 0.35,
    at: float = 0.4, fps: float = FPS,
) -> np.ndarray:
    """The shape `postprocess.beats` needs a prominence guard to survive.

    Same generator as `tests/test_postprocess.py`, with the notch at 0.4 of the cycle
    rather than that file's 0.5. 0.5 is a degenerate position and is tested for
    separately below: a notch exactly halfway between two beats makes a perfectly
    regular rhythm at twice the pulse rate, which no scale-based method can tell from
    a pulse. Real notches sit earlier, on the diastolic decay.
    """
    period = 60.0 / bpm * fps
    t = np.arange(int(seconds * fps))
    phase = np.mod(t, period) / period
    systolic = np.exp(-((phase - 0.12) ** 2) / (2 * 0.055**2))
    dicrotic = notch * np.exp(-((phase - at) ** 2) / (2 * 0.055**2))
    return systolic + dicrotic


def _wide_bandpass(x: np.ndarray, fps: float = FPS) -> np.ndarray:
    """0.5-8 Hz, 4th order, zero phase. The band Charlton et al. detect beats in."""
    b, a = sps.butter(4, [0.5 / fps * 2, 8.0 / fps * 2], btype="bandpass")
    return sps.filtfilt(b, a, np.double(sps.detrend(x)))


@pytest.mark.parametrize("bpm", [50.0, 72.0, 96.0, 145.0])
def test_one_peak_per_cycle_on_a_clean_tone(bpm: float) -> None:
    peaks, _ = msptd(_tone(bpm, 20.0), FPS)
    expected = 20.0 * bpm / 60.0
    assert len(peaks) == pytest.approx(expected, abs=2)


def test_peaks_land_on_the_maxima() -> None:
    """A peak index that is not a local maximum of the signal is a wrong index, not
    an approximate one."""
    wave = _tone(72.0, 20.0)
    peaks, _ = msptd(wave, FPS)
    inner = peaks[(peaks > 0) & (peaks < wave.size - 1)]
    assert np.all(wave[inner] >= wave[inner - 1])
    assert np.all(wave[inner] >= wave[inner + 1])


def test_onsets_land_on_the_minima() -> None:
    wave = _tone(72.0, 20.0)
    _, onsets = msptd(wave, FPS)
    inner = onsets[(onsets > 0) & (onsets < wave.size - 1)]
    assert np.all(wave[inner] <= wave[inner - 1])
    assert np.all(wave[inner] <= wave[inner + 1])


def test_beats_are_sorted_unique_and_inside_the_signal() -> None:
    """The windows overlap by 20%, so the same beat is detected more than once and
    the duplicates have to go. An index outside the signal would crash every caller
    that indexes with it."""
    wave = _tone(72.0, 40.0)
    for found in msptd(wave, FPS):
        assert found.size == np.unique(found).size
        assert np.all(np.diff(found) > 0)
        assert found.min() >= 0
        assert found.max() < wave.size


def test_the_notch_is_not_counted_as_a_beat() -> None:
    """The reason to port this at all.

    11 cycles in 10 s at 66 bpm. `find_peaks` with a spacing floor returns 17 on this
    shape and needs a conditional prominence guard bolted on to recover; MSPTD is
    scale-based and the notch is never a maximum at the scale that wins.
    """
    trace = _wide_bandpass(_pulse_with_dicrotic_notch(bpm=66.0))
    peaks, _ = msptd(trace, FPS)
    assert len(peaks) == pytest.approx(11, abs=1)


@pytest.mark.parametrize("bpm", [45.0, 55.0, 75.0, 95.0, 125.0, 140.0])
def test_the_notch_is_rejected_across_the_whole_plausible_band(bpm: float) -> None:
    """Both directions at once: a slow pulse is not doubled and a fast one is not
    halved, anywhere in 45-140 bpm.

    This is what the port buys. Over the same sweep `postprocess.beats` returns 31
    beats for the 15 cycles of a 45 bpm pulse -- 0.75 Hz is 45 bpm, so its band-pass
    corner sits on the signal -- and 27 for the 53 cycles at 160. `find_peaks` with a
    spacing floor on the wide band returns 27 for 15 cycles at 45 bpm and 48 for 25
    at 75. MSPTD is within one beat of the cycle count at every rate here.
    """
    seconds = 20.0
    trace = _wide_bandpass(_pulse_with_dicrotic_notch(bpm=bpm, seconds=seconds))
    peaks, _ = msptd(trace, FPS)
    assert len(peaks) == pytest.approx(seconds * bpm / 60.0, abs=1.5)


def test_a_notch_exactly_halfway_between_beats_cannot_be_rejected() -> None:
    """The limit of the method, stated rather than hidden.

    MSPTD reads scale and nothing else -- it never compares two maxima by height. A
    notch at 0.5 of the cycle is a second peak train at exactly twice the pulse rate,
    evenly spaced, and there is no scale at which the real beats are maxima and the
    notches are not. The algorithm then locks onto the doubled rhythm and returns
    about twice the true count.

    It is a degenerate case rather than a common one: the notch is the aortic valve
    closing, it falls on the diastolic decay at roughly 0.35-0.45 of the cycle, and
    at 0.4 the same signal reads correctly at every rate in the test above. Recorded
    because a real trace drifting toward 0.5 would fail this way and quietly.
    """
    trace = _wide_bandpass(
        _pulse_with_dicrotic_notch(bpm=85.0, seconds=20.0, at=0.5)
    )
    peaks, _ = msptd(trace, FPS)
    assert len(peaks) > 1.5 * (20.0 * 85.0 / 60.0)


def test_a_signal_shorter_than_one_window_is_still_processed() -> None:
    """ppg-beats makes the whole signal one window below 6 s rather than returning
    nothing."""
    peaks, _ = msptd(_tone(72.0, 3.0), FPS)
    assert len(peaks) == pytest.approx(3, abs=1)


def test_a_signal_longer_than_one_window_covers_its_whole_duration() -> None:
    """The last window is pinned to the end of the signal, so beats in the final
    partial window are not dropped."""
    wave = _tone(72.0, 43.0)
    peaks, _ = msptd(wave, FPS)
    assert peaks.max() > wave.size - MSPTD_WINDOW_S * FPS


def test_degenerate_input_returns_nothing_rather_than_raising() -> None:
    for bad in (np.zeros(300), np.full(300, np.nan), np.array([1.0, 2.0])):
        peaks, onsets = msptd(bad, FPS)
        assert peaks.size == 0
        assert onsets.size == 0


def test_the_scale_range_stops_at_the_slowest_plausible_rate() -> None:
    """Charlton's v2 refinement. Scale k resolves a rhythm of fs/(2k) Hz, so scales
    past k = fs only look for rhythms below 30 bpm and cost time to find nothing."""
    assert MSPTD_PLAUSIBLE_BPM == (30.0, 200.0)


def test_the_beat_table_pairs_each_peak_with_the_onset_before_it() -> None:
    table = beat_table(_tone(72.0, 20.0), FPS)
    assert isinstance(table, pl.DataFrame)
    assert table["onset"].to_numpy().max() < table["peak"].to_numpy().max()
    assert np.all(table["onset"].to_numpy() < table["peak"].to_numpy())


def test_the_beat_table_reports_the_interval_series() -> None:
    """One row per beat, and the interval column is the gap to the previous beat, so
    the first row has none."""
    table = beat_table(_tone(72.0, 20.0), FPS)
    assert table["ibi_ms"][0] is None
    intervals = table["ibi_ms"].drop_nulls().to_numpy()
    assert np.median(intervals) == pytest.approx(60_000.0 / 72.0, abs=35.0)


def test_the_beat_table_is_empty_rather_than_absent_when_there_are_no_beats() -> None:
    """A caller aggregating over several clips should not have to special-case one
    that produced nothing."""
    table = beat_table(np.zeros(300), FPS)
    assert table.height == 0
    assert set(table.columns) == {
        "beat", "peak", "onset", "t_s", "ibi_ms", "amplitude"
    }
