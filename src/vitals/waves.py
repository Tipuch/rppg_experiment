"""The RR network's inputs: each 30 s segment's PPG as 900 samples, and the two
pulse-interval series derived from it.

    10 s windows at 30 Hz (CFMamba's output, or contact PPG resampled to match)
      -> `segment_waves`   3 windows per segment, each z-scored, joined end to end
      -> `wave`            Array(Float32, 900)
      -> `pp_table`        + `tach` and `ls`, Array(Float32, 900) each

**Each window is z-scored on its own.** CFMamba is trained with a Pearson loss, so
each 10 s window comes back at an arbitrary scale and offset. Joined raw, a segment
steps at 10 s and 20 s, and the step is larger than any breath. After per-window
z-scoring each window has mean 0 and std 1. What survives is what breathing does
inside a window: the beat-to-beat change in pulse height, the slow baseline wander
and the pulse timing. A window flatter than 1e-8 becomes zeros (`zscore`), not NaN.
Contact rows go through the same rule, so a row means the same thing on both
domains.

**Pulse-interval channels.** After Zuern et al. 2026 (Sci Rep 16:22597), who
estimate RR from the pulse-to-pulse (PP) interval series of the PPG:

    wave -> Butterworth band-pass 0.6-4 Hz, order 4, zero-phase, z-scored
         -> `find_peaks` (systolic maxima) -> parabolic sub-sample peak time
         -> PP intervals: range 230-2400 ms, <= 5 robust SD, <= 20% off a centred
            median of 5
         -> `tach`  PCHIP of the kept intervals on the 30 Hz grid, z-scored
         -> `ls`    Lomb-Scargle power of the intervals, 0.05-1.0 Hz, z-scored

Both come from `wave` itself, not from the raw source PPG, so the same function
(`pp_signals`) runs at inference on whatever wave the model sees. Breathing speeds
the heart on inhalation and slows it on exhalation (respiratory sinus arrhythmia),
so `tach` shows the breath as a slow wave in beat timing, and `ls` shows its rate as
a peak. A row with fewer than `MIN_INTERVALS` kept intervals gets zeros in both, the
same "no information" value a flat window gets.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import ndimage
from scipy import signal as sps
from scipy.interpolate import PchipInterpolator

# One row per 30 s segment: its clip and start.
KEYS = ["clip_id", "segment_start_s"]
# CFMamba's output rate (`src.model.dataset.TARGET_FPS`), not imported so this module
# loads without torch. 3 windows of 10 s.
WAVE_FPS = 30.0
WAVE_SAMPLES = 900
WAVE_FILE = "waves.parquet"
PP_FILE = "waves_pp.parquet"
PP_SCHEMA = {
    "clip_id": pl.String,
    "segment_start_s": pl.Float64,
    **{c: pl.Array(pl.Float32, WAVE_SAMPLES) for c in ("wave", "tach", "ls")},
}


def zscore(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Zero mean, unit spread along `axis`; a window flatter than 1e-8 becomes 0, so
    a near-flat window is not blown up to full amplitude by its noise level."""
    x = np.asarray(x, dtype=np.float64)
    spread = x.std(axis=axis, keepdims=True)
    centred = x - x.mean(axis=axis, keepdims=True)
    return np.where(spread > 1e-8, centred / np.where(spread > 1e-8, spread, 1.0), 0.0)


def segment_waves(tagged: pl.DataFrame, column: str) -> pl.DataFrame:
    """One row per segment of `tagged` (KEYS, `window_start_s`, `column` holding a
    10 s window): `KEYS` and `wave`, the segment's windows in time order, each
    z-scored, joined end to end. Every segment must have the same window count."""
    ordered = tagged.sort(*KEYS, "window_start_s")
    keys = ordered.select(KEYS).unique(maintain_order=True)
    if keys.height == 0:
        return pl.DataFrame(schema={k: PP_SCHEMA[k] for k in (*KEYS, "wave")})
    x = zscore(
        ordered[column].to_numpy().reshape(keys.height, -1, ordered[column].dtype.size)
    )
    return keys.with_columns(
        wave=pl.Series(x.reshape(keys.height, -1).astype(np.float32))
    )


# Zuern et al.'s band: 0.6 Hz (36 bpm) keeps the slowest heart and removes the
# breathing baseline wander, which would otherwise lift or sink whole beats past the
# peak threshold in step with the breath; 4 Hz (240 bpm) keeps the pulse but not
# the dicrotic detail.
PP_BAND_HZ = (0.6, 4.0)
PP_ORDER = 4
_SOS = sps.butter(PP_ORDER, PP_BAND_HZ, btype="band", fs=WAVE_FPS, output="sos")
# Band-passed, then z-scored, so the peak thresholds below are in SDs of the
# filtered row. The zero-phase filter has an edge transient: on a pure tone the
# intervals in the first and last ~4 s are off by up to tens of ms, the interior by
# well under 1 ms.
_bandpass = lambda x: zscore(sps.sosfiltfilt(_SOS, x, axis=-1))
# The paper's find_peaks settings (height 1, prominence 1.5) miss beats on this
# z-scored band-passed wave: on 2,000 real rows they found a median 0.89x the beats
# the spectral heart rate predicts. Height 0.3 and prominence 0.5 find 1.00x
# (10th-90th percentile 0.94-1.11x). Spurious and missed beats that pass both still
# make outlying intervals, which the 3 interval rules below drop.
PEAK_HEIGHT = 0.3
PEAK_PROMINENCE = 0.5
# Smart Fusion's plausible pulse periods (Karlen 2013), 230-2400 ms (the user chose
# them over the paper's 600-1500 ms, so slow and fast hearts keep their beats). The
# shortest, in whole samples, is find_peaks' minimum spacing: 7 samples = 233 ms.
PP_RANGE_S = (0.23, 2.4)
PEAK_DISTANCE = int(np.ceil(PP_RANGE_S[0] * WAVE_FPS))
# Intervals more than 5 robust SD (1.4826 x MAD) from the row's median are dropped.
# The robust SD is floored at 1 sample (33 ms), so a very regular row doesn't lose
# intervals to timing jitter.
PP_MAX_SD = 5.0
PP_MIN_SD_S = 1 / WAVE_FPS
# Unverified: the paper's non-causal outlier filter cites Vila et al. 1997 with no
# detail. The stand-in drops an interval more than 20% off a centred median of 5
# kept intervals (Malik's 20% rule). The median is a reference, not a smoother: 5
# beats at 70 bpm span about 1 breath, so replacing the series by it would erase
# the breathing.
PP_LOCAL_BEATS = 5
PP_LOCAL_TOL = 0.2
# Fewer than 8 beats in 30 s can't resolve a breath cycle.
MIN_INTERVALS = 8
# 900 linear bins over 0.05-1.0 Hz, 3-60 breaths/min, so `ls` has the wave's length.
# A 30 s row resolves only ~1/30 Hz, so neighbouring bins are smooth.
LS_FREQS = np.linspace(0.05, 1.0, WAVE_SAMPLES)
_GRID = np.arange(WAVE_SAMPLES) / WAVE_FPS


def beat_intervals(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Beat times (s) and the kept PP intervals (s) of 1 band-passed row (`_bandpass`).

    Each interval is placed at the time of the beat that ends it. Peaks are refined
    to sub-sample time with a parabola through the peak sample and its 2 neighbours:
    at 30 Hz a whole-sample peak is off by up to 17 ms, comparable to the breathing
    modulation of the interval itself. Unverified: the paper refines with a
    2nd-order Chebyshev fit; whether it fits the same 3 points is not confirmed.
    """
    peaks = sps.find_peaks(
        y, height=PEAK_HEIGHT, prominence=PEAK_PROMINENCE, distance=PEAK_DISTANCE
    )[0]
    # find_peaks never returns the first or last sample, so both neighbours exist.
    a, b, c = y[peaks - 1], y[peaks], y[peaks + 1]
    curve = a - 2 * b + c
    shift = np.where(curve < 0, 0.5 * (a - c) / np.where(curve < 0, curve, -1.0), 0.0)
    beat = (peaks + np.clip(shift, -0.5, 0.5)) / WAVE_FPS
    t, pp = beat[1:], np.diff(beat)
    keep = (pp >= PP_RANGE_S[0]) & (pp <= PP_RANGE_S[1])
    t, pp = t[keep], pp[keep]
    if pp.size == 0:
        return t, pp
    centre = np.median(pp)
    sd = max(1.4826 * np.median(np.abs(pp - centre)), PP_MIN_SD_S)
    keep = np.abs(pp - centre) <= PP_MAX_SD * sd
    t, pp = t[keep], pp[keep]
    local = ndimage.median_filter(pp, size=PP_LOCAL_BEATS, mode="nearest")
    keep = np.abs(pp - local) <= PP_LOCAL_TOL * local
    return t[keep], pp[keep]


def pp_series(t: np.ndarray, pp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`tach` and `ls`, float32 x `WAVE_SAMPLES` each, from kept intervals `pp` at
    beat times `t` (`beat_intervals`). Zeros for both below `MIN_INTERVALS`.

    `tach` is PCHIP, not a cubic spline: a dropped beat leaves a gap between knots,
    and a C2 spline overshoots across a gap into a swing no beat measured. Before the
    first beat and after the last it holds the end value flat. `ls` takes the uneven
    beat times as they are, so a dropped beat costs a sample, not a resample.
    """
    if pp.size < MIN_INTERVALS:
        return np.zeros(WAVE_SAMPLES, np.float32), np.zeros(WAVE_SAMPLES, np.float32)
    tach = PchipInterpolator(t, pp)(np.clip(_GRID, t[0], t[-1]))
    ls = sps.lombscargle(t, pp - pp.mean(), 2 * np.pi * LS_FREQS)
    return zscore(tach).astype(np.float32), zscore(ls).astype(np.float32)


def pp_signals(wave: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(tach, ls)` of 1 upright `WAVE_SAMPLES` wave at `WAVE_FPS`. The function to
    call at inference, on the same wave the model reads."""
    return pp_series(*beat_intervals(_bandpass(np.asarray(wave, np.float64))))


def pp_table(waves: pl.DataFrame) -> pl.DataFrame:
    """`KEYS`, `wave`, `tach` and `ls` for each row of `waves` (KEYS, `wave`), in its
    order. The filter runs on all rows at once; peaks and intervals per row."""
    if waves.height == 0:
        return pl.DataFrame(schema=PP_SCHEMA)
    y = _bandpass(waves["wave"].to_numpy().astype(np.float64))
    tach, ls = zip(*(pp_series(*beat_intervals(r)) for r in y), strict=True)
    return waves.select(*KEYS, "wave").with_columns(
        tach=pl.Series(np.stack(tach)), ls=pl.Series(np.stack(ls))
    )
