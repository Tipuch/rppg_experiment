"""MSPTD: multi-scale peak and trough detection, ported from ppg-beats.

A port of `msptdfastv2_beat_detector.m` and the `msptdpcref` core it calls, from
Peter H. Charlton's ppg-beats toolbox (github.com/peterhcharlton/ppg-beats, MIT).
The algorithm is Bishop and Ercole, "Multi-scale peak and trough detection optimised
for periodic and quasi-periodic neuroscience data", Acta Neurochirurgica Supplement
126 (2018) 189-195; the v2 refinements are Charlton et al., "MSPTDfast: An Efficient
Photoplethysmography Beat Detection Algorithm", Computing in Cardiology 2024.

**Why this exists here.** `postprocess.beats` is `scipy.signal.find_peaks` with a
minimum spacing, and a spacing floor cannot tell a dicrotic notch from a beat -- the
notch lands 15 frames after the systolic peak at 66 bpm, past the 12-frame floor. The
repair in `postprocess` is a conditional prominence threshold, which is a patch on a
detector that has no notion of scale. MSPTD has one: a sample is a peak only if it is
the largest in its neighbourhood at *every* scale up to the scale that best explains
the signal, and the notch fails that at the scale the pulse rate wins.

**How it works.** For each half-width k, mark every sample larger than the samples k
before and k after it. That is the local maxima scalogram, one row per k. The row
with the most marks is the scale that best matches the signal's own periodicity --
call it lambda. Keep rows 1..lambda and return the columns marked in all of them.
Troughs are the same with the comparison reversed.

**Numpy, not polars, for the scalogram.** It is a dense boolean matrix reduced along
both axes; there are no rows, keys or groups in it. `beat_table` is where the result
becomes tabular and where polars earns its place.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import signal as sps

# ppg-beats splits into overlapping windows and detects within each, so one noisy
# stretch cannot set the winning scale for the whole recording. 6 s and 20% are the
# MSPTDfast v1.1 settings (`msptdfastv2_beat_detector.m`), not the 8 s default.
MSPTD_WINDOW_S = 6.0
MSPTD_OVERLAP = 0.2
# Scale k resolves a rhythm at fs/(2k) Hz, so scales past the slowest plausible pulse
# only look for rhythms nobody has and cost time to find nothing. This is v2's
# `use_reduced_lms_scales`, and at 30 fps it caps the scalogram at about 30 rows
# instead of the 90 a 6 s window would otherwise give.
MSPTD_PLAUSIBLE_BPM = (30.0, 200.0)
# Downsampling target. `floor(fs/20)` is the factor, so 30 fps gives 1 and nothing is
# resampled; the branch is kept because the port should behave like the original on
# the contact-PPG sampling rates it was tuned at.
MSPTD_DS_HZ = 20.0


def _windows(n: int, fs: float) -> list[tuple[int, int]]:
    """Half-open [start, end) windows covering the whole signal.

    The original computes an inclusive end and pins a final window to the last
    sample, so the tail is never dropped when the length is not a whole number of
    strides. That final window overlaps its predecessor by however much is left over,
    which is why the detections have to be de-duplicated afterwards.
    """
    width = round(MSPTD_WINDOW_S * fs)
    if n <= width:
        return [(0, n)]
    stride = max(1, round(width * (1.0 - MSPTD_OVERLAP)))
    starts = list(range(0, n - width, stride))
    spans = [(s, s + width + 1) for s in starts]
    if not spans or spans[-1][1] < n:
        spans.append((n - width - 1, n))
    return spans


def _max_scale(n: int, fs: float) -> int:
    """Number of scalogram rows, capped at the slowest plausible pulse.

    Follows the original's arithmetic rather than the equivalent closed form: scale
    index k is included while `(L / k) / duration` is at or above the low end of the
    plausible band, where L is the largest scale a signal of this length admits.
    """
    limit = (n + 1) // 2 - 1
    if limit < 1:
        return 0
    duration = n / fs
    scales = np.arange(1, limit + 1, dtype=np.float64)
    resolvable_hz = (limit / scales) / duration
    inside = np.flatnonzero(resolvable_hz >= MSPTD_PLAUSIBLE_BPM[0] / 60.0)
    return int(inside[-1]) + 1 if inside.size else 1


def _scalogram(x: np.ndarray, max_scale: int, maxima: bool) -> np.ndarray:
    """The local maxima (or minima) scalogram: one boolean row per scale.

    Row k-1 marks every sample strictly beyond both the sample k before it and the
    sample k after it. The original loops over both scale and sample; here each row
    is three array slices, which is the same comparison for every sample at once.
    """
    n = x.size
    marks = np.zeros((max_scale, n), dtype=bool)
    for k in range(1, max_scale + 1):
        if n - 2 * k < 1:
            continue
        middle, before, after = x[k : n - k], x[: n - 2 * k], x[2 * k :]
        marks[k - 1, k : n - k] = (
            (middle > before) & (middle > after) if maxima
            else (middle < before) & (middle < after)
        )
    return marks


def _extrema(x: np.ndarray, max_scale: int, maxima: bool) -> np.ndarray:
    """Indices marked at every scale up to the one with the most marks."""
    marks = _scalogram(x, max_scale, maxima)
    if marks.size == 0:
        return np.empty(0, dtype=int)
    winner = int(np.argmax(marks.sum(axis=1)))
    return np.flatnonzero(marks[: winner + 1].all(axis=0))


def _snap(x: np.ndarray, found: np.ndarray, tolerance: int, maxima: bool) -> np.ndarray:
    """Move each index to the largest sample within `tolerance` of it.

    The scalogram is computed on a detrended and possibly downsampled copy, so an
    index can sit a sample or two off the extremum of the signal the caller passed.
    The ppg-beats original slices without bounds-checking and relies on the tolerance
    being smaller than the margins; clipping here makes a beat at the very edge safe.

    This returns the largest sample in the window, which is not necessarily a local
    maximum -- if the true peak is further away than `tolerance`, the result is a point
    on the flank. That is the original's behaviour and is right here, where the window
    is small and the index is nearly correct already. `postprocess.align_to_peaks` is
    the stricter operation, for moving an index between two differently filtered
    copies of the same signal.
    """
    if found.size == 0:
        return found
    starts = np.maximum(found - tolerance, 0)
    ends = np.minimum(found + tolerance + 1, x.size)
    pick = np.argmax if maxima else np.argmin
    return np.array(
        [start + int(pick(x[start:end])) for start, end in zip(starts, ends, strict=True)],
        dtype=int,
    )


def _tolerance(fs: float) -> int:
    """Snapping radius in samples. The original widens it at low sampling rates."""
    seconds = 0.05 if fs >= 20.0 else (0.1 if fs >= 10.0 else 0.2)
    return int(np.ceil(fs * seconds))


def msptd(sig: np.ndarray, fs: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """Pulse peaks and onsets, as sorted arrays of indices into `sig`.

    `sig` should already be band-passed for beat detection. Charlton et al. use a
    4th-order zero-phase Butterworth at 0.5-8 Hz; `postprocess.detection_bandpass` is
    the same filter over the swept 0.75-4 Hz, and either way it is deliberately wider
    than the 0.75-2.5 Hz band rates are read in. The extra harmonics are what make a
    pulse wave a shape rather than a sinusoid, and this detector reads shape.

    Returns two empty arrays for a signal too short, flat, or not finite, so a dead
    prediction early in training cannot crash a pass.
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    if x.size < 8 or not np.isfinite(x).all() or np.ptp(x) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)

    factor = max(1, int(np.floor(fs / MSPTD_DS_HZ)))
    tolerance = _tolerance(fs / factor)
    peaks: list[np.ndarray] = []
    onsets: list[np.ndarray] = []

    for start, end in _windows(x.size, fs):
        window = x[start:end]
        # Linear detrend, per the original's `detrend(x)`. Not the smoothness-prior
        # detrend `postprocess.bandpass` uses: the scalogram compares a sample with
        # its neighbours k apart, so only a straight-line tilt across the window can
        # bias it, and removing more would remove pulses.
        scan = sps.detrend(window[::factor]) if window[::factor].size >= 8 else None
        if scan is None:
            continue
        max_scale = _max_scale(scan.size, fs / factor)
        if max_scale < 1:
            continue
        found_peaks = _extrema(scan, max_scale, maxima=True) * factor
        found_onsets = _extrema(scan, max_scale, maxima=False) * factor
        peaks.append(_snap(window, found_peaks, tolerance, maxima=True) + start)
        onsets.append(_snap(window, found_onsets, tolerance, maxima=False) + start)

    # The windows overlap, so the same beat is found more than once. `tidy_beats` in
    # the original is exactly this: sort and de-duplicate.
    return (
        np.unique(np.concatenate(peaks)) if peaks else np.empty(0, dtype=int),
        np.unique(np.concatenate(onsets)) if onsets else np.empty(0, dtype=int),
    )


BEAT_SCHEMA = {
    "beat": pl.Int64, "peak": pl.Int64, "onset": pl.Int64,
    "t_s": pl.Float64, "ibi_ms": pl.Float64, "amplitude": pl.Float64,
}


def beat_table(sig: np.ndarray, fs: float = 30.0) -> pl.DataFrame:
    """One row per detected beat: its peak, its onset, its time, interval and height.

    A peak is paired with the last onset before it, which is the pairing every
    downstream measurement assumes -- the AC amplitude of a beat is the rise from its
    own onset, and the systolic upslope runs between the two. Peaks with no onset
    before them are dropped rather than paired with a later one.

    `ibi_ms` is the gap to the previous beat, so the first row is null: there is no
    interval before the first beat, and a zero there would be counted as one.

    Returns an empty frame with these columns when nothing is detected, so a caller
    concatenating over clips does not have to special-case the silent ones.
    """
    x = np.asarray(sig, dtype=np.float64).ravel()
    peaks, onsets = msptd(x, fs)
    if peaks.size == 0 or onsets.size == 0:
        return pl.DataFrame(schema=BEAT_SCHEMA)

    # The onset for each peak is the last one strictly before it. `searchsorted` on a
    # sorted array gives how many onsets precede the peak, so one less is its index,
    # and -1 marks a peak that has none.
    preceding = np.searchsorted(onsets, peaks, side="left") - 1
    keep = preceding >= 0
    peaks, preceding = peaks[keep], preceding[keep]
    if peaks.size == 0:
        return pl.DataFrame(schema=BEAT_SCHEMA)
    paired = onsets[preceding]

    return pl.DataFrame(
        {
            "beat": np.arange(peaks.size, dtype=np.int64),
            "peak": peaks.astype(np.int64),
            "onset": paired.astype(np.int64),
            "t_s": peaks / fs,
            "amplitude": x[peaks] - x[paired],
        },
        schema_overrides=BEAT_SCHEMA,
    ).with_columns(
        (pl.col("t_s").diff() * 1000.0).alias("ibi_ms")
    ).select(list(BEAT_SCHEMA))
