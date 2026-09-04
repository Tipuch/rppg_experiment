"""Heart rate, SNR and MACC from a predicted waveform.

Both source papers state the same post-processing: band-pass the predicted BVP
with a second-order Butterworth at 0.75-2.5 Hz, then take the dominant spectral
peak. Both also state they used the rPPG-Toolbox, which is vendored under
tools/rPPG-Toolbox -- so the primitives here are imported from it rather than
rewritten, and the numbers stay comparable with every method in its tables.

The toolbox's own wrapper is not reused. Its `calculate_metric_per_video`
hardcodes `butter(1, [0.6, 3.3])` -- a first-order filter over a wider range than
either paper specifies. Its comments say to use 0.75 and 2.5 "to more closely match
results in the NeurIPS 2023 toolbox paper", but the constants are not parameters,
so matching the papers means calling the primitives directly. Inheriting 0.6-3.3 Hz
admits a 36 bpm drift and a 198 bpm harmonic into every heart-rate estimate.

The range is the same 0.75-2.5 Hz the DF-FFN's Gaussian mask is constrained to.

Beats are not detected in that range. `beats` runs MSPTD over 0.75-4 Hz, following
Charlton et al., "Detecting beats in the photoplethysmogram: benchmarking open-source
algorithms", Physiological Measurement 43(8) 085007 (2022),
<https://doi.org/10.1088/1361-6579/ac826d>, which benchmarked fifteen open-source
detectors and found MSPTD and qppg best, MSPTD with the higher positive predictive
value. The port is `src.model.msptd`. The idea of detecting in a wider band than the
one rates are reported in is from the same group's "Determinants of
photoplethysmography signal quality at the wrist", PLOS Digital Health 4(6) e0000585
(2025), which uses 0.5-8 Hz; the edges here were swept and are recorded beside
`DETECTION_LOW_HZ`.

Two consequences worth stating plainly. Detection reaches 240 bpm where the reporting
band stops at 150. And on the one labelled split this has been measured over, the
detector this replaced scores better -- 6.24 RMSE against 7.28 -- on a sample whose
contact-PPG rate has a p99 of 118 bpm. The trade is stated at `DETECTION_LOW_HZ`.
"""

from __future__ import annotations

import importlib.util
import itertools
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import linalg as sps_linalg
from scipy import signal as sps

from .msptd import msptd as msptd_peaks

# Both papers, Section 4.1 / 4.2. 45-150 bpm.
LOW_HZ, HIGH_HZ = 0.75, 2.5
BUTTER_ORDER = 2
# Detrending strength, from the toolbox's own call site.
DETREND_LAMBDA = 100

_TOOLBOX = Path(__file__).resolve().parents[2] / "tools" / "rPPG-Toolbox"


def _vendored() -> ModuleType:
    """Load the toolbox's post_process module by path.

    By path rather than by `sys.path` insertion: the toolbox has top-level modules
    named `evaluation`, `dataset` and `tools`, and putting its root on the import
    path would let any of them shadow a module of this project. The module imports
    only numpy and scipy, so it loads in isolation.
    """
    path = _TOOLBOX / "evaluation" / "post_process.py"
    if not path.exists():
        raise FileNotFoundError(
            f"vendored rPPG-Toolbox not found at {path}. It provides the SNR and "
            "MACC definitions both papers report against."
        )
    spec = importlib.util.spec_from_file_location("_rppg_toolbox_post_process", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_POST = _vendored()
macc = _POST._compute_macc


def detrend(wave: np.ndarray, lambda_value: float = DETREND_LAMBDA) -> np.ndarray:
    """Smoothness-prior detrend. The toolbox's `_detrend`, solved instead of inverted.

    Both compute `x - inv(I + lambda^2 D'D) x`, where D is the second-difference
    operator. The vendored version forms that N x N matrix densely and inverts it.
    The matrix is pentadiagonal and symmetric positive definite, so a banded Cholesky
    solve gives the same answer in O(N) rather than O(N^3): 0.2 ms per 300-sample
    window here against 341 ms, on a machine whose LAPACK inverts a 300 x 300 in 93 ms.

    It is not a fork of the toolbox's definition -- the arithmetic is identical and
    `tests/test_postprocess.py` asserts agreement to 1e-8 against the vendored
    function, which is still imported and still the reference. It is a fork of its
    algorithm, taken because the readout sweep calls this tens of thousands of times
    and three hours of dense inverses is the difference between a measurement that
    gets run and one that does not.
    """
    # Shape is preserved, not flattened. `baselines` injects this into the vendored
    # POS implementation, which hands it an N x 1 matrix and transposes what comes
    # back; a 1-D return there collapses to a scalar and `filtfilt` fails on it.
    x = np.asarray(wave, dtype=np.float64)
    flat = x.ravel()
    n = flat.size
    if n < 3:
        return x.astype(np.float64, copy=True)
    # The dense inverse propagated a non-finite sample into every output sample and
    # returned quietly; the banded solve validates its input and raises instead. A
    # still clip does reach here -- POS divides by the standard deviation of a
    # constant signal -- so the degenerate case has to keep behaving as it did.
    if not np.isfinite(flat).all():
        return np.full(x.shape, np.nan)
    # Upper-banded form of I + lambda^2 D'D, two superdiagonals. Row i of D'D is the
    # correlation of the [1, -2, 1] stencil with itself, [1, -4, 6, -4, 1], truncated
    # at the ends where fewer stencils overlap -- which is what building D'D from D
    # does, so it is built rather than written out.
    stencil = np.array([1.0, -2.0, 1.0])
    d_t_d = np.zeros((3, n))
    for row in range(n - 2):
        for offset_a in range(3):
            for offset_b in range(offset_a, 3):
                d_t_d[offset_b - offset_a, row + offset_a] += (
                    stencil[offset_a] * stencil[offset_b]
                )
    banded = (lambda_value**2) * d_t_d
    banded[0] += 1.0
    # solveh_banded wants ab[u + i - j, j]; our row d holds the d-th diagonal, which
    # is ab[u - d].
    ab = np.zeros((3, n))
    ab[2] = banded[0]
    ab[1, 1:] = banded[1, : n - 1]
    ab[0, 2:] = banded[2, : n - 2]
    return (flat - sps_linalg.solveh_banded(ab, flat, lower=False)).reshape(x.shape)


def bandpass(wave: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Detrend, then second-order Butterworth over the cardiac band."""
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 3 * BUTTER_ORDER + 1 or not np.isfinite(wave).all():
        return wave
    detrended = detrend(wave, DETREND_LAMBDA)
    b, a = sps.butter(
        BUTTER_ORDER, [LOW_HZ / fps * 2, HIGH_HZ / fps * 2], btype="bandpass"
    )
    return sps.filtfilt(b, a, np.double(detrended))


def heart_rate(wave: np.ndarray, fps: float = 30.0, filtered: bool = False) -> float:
    """Dominant cardiac-band frequency of a waveform, in bpm.

    The periodogram is zero-padded to the next power of two, which interpolates the
    spectrum so the peak is not forced onto a coarse bin grid: at 160 frames and
    30 fps the raw bins are 11.25 bpm apart, wider than the accuracy reported.
    Padding adds no information; it stops the grid from discarding what is there.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    if not filtered:
        wave = bandpass(wave, fps)
    return float(_POST._calculate_fft_hr(wave, fs=fps, low_pass=LOW_HZ, high_pass=HIGH_HZ))


def _vertex_shift(
    left: np.ndarray | float, mid: np.ndarray | float, right: np.ndarray | float
) -> np.ndarray:
    """Offset of a parabola's vertex from its middle sample, in samples.

    Three equally spaced points define one parabola, and its vertex is where the
    peak sits. Two readouts need this for the same reason: both quantise a
    continuous position onto a grid coarser than the accuracy reported -- a spectral
    peak onto 3.5 bpm bins, a beat onto 33 ms frames.

    Clipped to half a sample. A larger offset means the three points do not
    bracket a peak, and the middle one was not a maximum.
    """
    denominator = np.asarray(left - 2.0 * mid + right, dtype=np.float64)
    usable = np.abs(denominator) > 1e-12
    # Divided only where the denominator is non-zero, rather than divided everywhere
    # and selected afterwards. `np.where` picks between two arrays that have both
    # already been evaluated, so the guarded form still computed 0/0 for every flat
    # triple and warned about it. A flat triple is ordinary: a band-passed trace has
    # stretches where three samples are equal to floating-point precision.
    shift = np.zeros(denominator.shape, dtype=np.float64)
    np.divide(
        0.5 * np.asarray(left - right, dtype=np.float64),
        denominator,
        out=shift,
        where=usable,
    )
    return np.clip(shift, -0.5, 0.5)


def _band_spectrum(
    wave: np.ndarray, fps: float, pad: int = 8, window: str = "hann"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodogram of `wave`, plus the indices of the cardiac band within it.

    `pad` multiplies the next power of two. Padding adds no information; it
    interpolates the grid so a peak is not forced onto bins 3.5 bpm apart.

    The band is returned as indices rather than as a sliced spectrum, because a
    caller reading a peak's neighbours must read them from the full spectrum: at
    the band edge the neighbour it needs lies outside the band, and slicing first
    would score an edge bin against a truncated window.
    """
    nfft = pad * (1 << int(np.ceil(np.log2(wave.size))))
    freqs, power = sps.periodogram(
        wave, fs=fps, nfft=nfft, detrend=False, window=window
    )
    return freqs, power, np.flatnonzero((freqs >= LOW_HZ) & (freqs <= HIGH_HZ))


def spectral_hr(
    wave: np.ndarray, fps: float = 30.0, *, filtered: bool = False,
    pad: int = 8, window: str = "hann", interpolate: bool = True,
) -> float:
    """Dominant cardiac-band rate, with the refinements `heart_rate` omits.

    `heart_rate` is the vendored toolbox's readout, so numbers computed through it
    stay comparable with the toolbox's own tables. This is the same measurement plus
    the steps production PPG readouts add: a window function, heavier zero-padding,
    and interpolation of the peak between bins.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    if not filtered:
        wave = bandpass(wave, fps)
    freqs, power, inside = _band_spectrum(wave, fps, pad=pad, window=window)
    if inside.size == 0:
        return float("nan")
    peak = int(inside[np.argmax(power[inside])])
    if not interpolate or peak == 0 or peak == power.size - 1:
        return float(freqs[peak] * 60.0)
    left, mid, right = np.log(np.maximum(power[peak - 1 : peak + 2], 1e-30))
    shift = float(_vertex_shift(left, mid, right))
    return float((freqs[peak] + shift * (freqs[1] - freqs[0])) * 60.0)


def _rate_from_peaks(
    trace: np.ndarray, peaks: np.ndarray, fps: float,
    aggregate: Callable[[np.ndarray], float] = np.median,
) -> float:
    """bpm from the aggregated gap between `peaks`. NaN if there is no gap to take.

    `aggregate` defaults to the median, which is the readout this project reports.
    It is a parameter so `readout.VARIANTS` can score another one over the same
    peaks; see `interval_hr` for why the median is the default.
    """
    if len(peaks) < 2:
        return float("nan")
    gaps = np.diff(refine(trace, peaks))
    if not len(gaps):
        return float("nan")
    return float(60.0 * fps / aggregate(gaps))


def beats(
    trace: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> np.ndarray:
    """Peak indices, one per cardiac cycle, by MSPTD.

    Charlton et al. 2022 benchmarked fifteen open-source PPG beat detectors and found
    MSPTD among the two best, with the higher positive predictive value of the pair --
    which is the property that matters here, because a false beat is what breaks an
    interval readout. `src.model.msptd` is the port.

    `filtered` means the trace is already in the detection band. Otherwise it is
    filtered here, to 0.75-4 Hz rather than the 0.75-2.5 Hz rates are read in: MSPTD
    separates a beat from a dicrotic notch by scale, and the harmonics that make a
    notch a distinguishable feature live above 2.5 Hz.

    This replaced `find_peaks` with a spacing floor and a conditional prominence
    guard. The floor could not tell a notch from a beat -- the notch lands about 15
    frames after the systolic peak at 66 bpm, past a 12-frame floor -- and the guard
    was a patch on a detector with no notion of scale. On a 20 s notched pulse the old
    pair read 31 beats for the 15 cycles of a 45 bpm trace and 27 for the 53 cycles of
    a 160 bpm one; MSPTD is within one beat of the cycle count at every rate from 45
    to 160 bpm.

    One case defeats it, and no scale-based method can do better: a notch at exactly
    half the cycle is an evenly spaced second peak train at twice the pulse rate, and
    no scale marks the beats without also marking the notches. Real notches fall at
    0.35-0.45 of the cycle. `tests/test_msptd.py` pins both the working range and that
    limit.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    if trace.size < 8 or not np.isfinite(trace).all():
        return np.empty(0, dtype=int)
    scan = trace if filtered else detection_bandpass(trace, fps)
    return msptd_peaks(scan, fps)[0]


def align_to_peaks(
    trace: np.ndarray, peaks: np.ndarray, fps: float = 30.0
) -> np.ndarray:
    """Move each index onto the nearest true local maximum of `trace`.

    For reading an index found on one filtered copy of a signal against another. Beats
    are detected in 0.75-4 Hz and plotted on the 0.75-2.5 Hz trace; over 300 windows of
    `build/readout_test_s900.npz` those two filters disagree about which sample is the
    peak for 3152 of 3785 beats, always by a frame or a few. A marker left where it was
    sits on the flank, and `trace[peaks]` read as a beat amplitude reads the flank too.

    The search radius is half the shortest beat interval the detection band admits --
    `fps / DETECTION_HIGH_HZ` is 7.5 frames at 30 fps, so 4 -- which is what stops a
    marker being moved onto its neighbour. An index with no local maximum inside that
    radius is left alone: at that distance the two bands disagree about where the beat
    is, not about which sample of it is highest, and moving the marker further would
    hide the disagreement rather than resolve it. That leaves 2 of 3785 in place.

    Unlike `refine`, this returns integer indices. `refine` interpolates a position
    between samples for timing; this picks a sample, because a caller indexing the
    trace needs one that exists.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    peaks = np.asarray(peaks, dtype=int).ravel()
    if peaks.size == 0 or trace.size < 3:
        return peaks
    radius = max(1, round(fps / DETECTION_HIGH_HZ / 2))
    candidates, _ = sps.find_peaks(trace)
    if candidates.size == 0:
        return peaks
    nearest = candidates[np.argmin(np.abs(candidates[None, :] - peaks[:, None]), axis=1)]
    return np.where(np.abs(nearest - peaks) <= radius, nearest, peaks)


def refine(trace: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions, by the vertex of a parabola through 3 points.

    An integer index quantises the interval by one frame, which at 30 fps is 3.4 bpm
    at 110 and 6 bpm at 150 -- coarser than the differences being reported. The
    band-pass leaves a smooth trace, so the three samples around a peak locate its
    vertex.
    """
    if len(peaks) == 0:
        return peaks.astype(np.float64)
    inner = peaks[(peaks > 0) & (peaks < len(trace) - 1)]
    if len(inner) != len(peaks):
        return peaks.astype(np.float64)
    return peaks + _vertex_shift(trace[peaks - 1], trace[peaks], trace[peaks + 1])


def interval_hr(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False,
    aggregate: Callable[[np.ndarray], float] = np.median,
) -> float:
    """Heart rate from the median inter-beat interval. One member of `reported_hr`.

    Detect one peak per cardiac cycle, difference their positions, take the median.
    This is what pulse oximeters and wrist wearables display. Median rather than
    mean because one missed or doubled beat moves a mean of a dozen intervals by
    several bpm; `aggregate` swaps that choice, which is what the `interval_mean`
    variant in `readout.VARIANTS` is.

    Over 1569 labelled test windows it makes fewer large misses than the toolbox's
    spectral argmax -- RMSE 7.73 against 8.18 and rho 0.821 against 0.793 -- for
    0.51 bpm more MAE. It is not what this project reports on its own: `reported_hr`
    votes it against the spectral peak and the mean interval and beats both.
    `src.cli readout` reruns that sweep.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    # The detection band, not the reporting band. `filtered` says the caller already
    # applied it; a caller that applied `bandpass` instead is handing over a trace
    # whose harmonics are gone, and MSPTD will still find the beats but without the
    # shape information it separates a notch by.
    scan = wave if filtered else detection_bandpass(wave, fps)
    return _rate_from_peaks(scan, beats(scan, fps, filtered=True), fps,
                            aggregate=aggregate)


def snr(wave: np.ndarray, reference_bpm: float, fps: float = 30.0,
        filtered: bool = False) -> float:
    """Signal-to-noise ratio in dB, CFMamba Eqs. 26-27.

    Signal is the power within 6 bpm of the reference rate and of its second
    harmonic; noise is the rest of the range. The harmonic is included because a
    pulse has one: a prediction carrying only the fundamental is a sinusoid rather
    than a pulse waveform.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or not np.isfinite(reference_bpm):
        return float("nan")
    if not filtered:
        wave = bandpass(wave, fps)
    return float(
        _POST._calculate_SNR(wave, reference_bpm, fs=fps, low_pass=LOW_HZ, high_pass=HIGH_HZ)
    )


def compare(predicted: np.ndarray, truth: np.ndarray, fps: float = 30.0) -> dict[str, float]:
    """Everything reportable for one window, from a prediction and its target.

    The ground-truth rate is taken from the contact PPG over the same window, not
    from the manifest's label column. DATASETS.md records subject24 labelled 96 bpm
    against a PPG reading of 127.2, plus four other UBFC subjects whose HR readout
    drops out while the waveform stays intact. Reading the rate off the signal
    avoids that, and is what the toolbox does.
    """
    # SNR and MACC are defined on the reporting band and read the filtered traces.
    # The rates take the raw waves, because `reported_hr` needs both bands and can
    # only build the detection band from a signal that still has its harmonics.
    pred_filtered = bandpass(predicted, fps)
    truth_filtered = bandpass(truth, fps)
    hr_truth = reported_hr(truth, fps)
    return {
        "hr_pred": reported_hr(predicted, fps),
        "hr_true": hr_truth,
        "snr": snr(pred_filtered, hr_truth, fps, filtered=True),
        "macc": float(macc(pred_filtered, truth_filtered)),
    }


# The respiratory band, 6-30 breaths a minute. Wider than an adult at rest, which is
# 12-20, because the readouts below are also the ones a subject holding their breath
# or recovering from exertion has to fall inside.
RESP_LOW_HZ, RESP_HIGH_HZ = 0.1, 0.5
# Shortest window a respiratory rate is returned from. The slowest breath in the band
# has a 10 s period, and a window holding one cycle of it cannot place a peak: the
# periodogram returns the bottom of the band whatever the subject is doing. Three
# cycles is the usual floor and is what this is. `predict` runs 10 s windows by
# default, so most single-window calls return NaN here, which is the intent.
RESP_MIN_SECONDS = 30.0
# The interval series is sampled unevenly -- once per beat -- so it is resampled onto
# a uniform grid before any spectrum is taken. 4 Hz is the HRV frequency-domain
# convention and is eight times the top of the respiratory band.
IBI_RESAMPLE_HZ = 4.0
# A successive difference this large counts toward pNN50, in milliseconds. At 30 fps
# one frame is 33.3 ms, so this threshold is 1.5 frames: it is only meaningful because
# `refine` locates each peak between samples, and it is close enough to the sampling
# floor that pNN50 is the least trustworthy of the three.
NN50_MS = 50.0


def intervals_ms(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """The inter-beat interval series and the time of each interval, in ms and s.

    Every variability and respiratory number below is read off this pair, and it
    comes from the same `beats` and `refine` the heart rate does, so a rate and a
    variability reported together cannot disagree about which peaks were beats.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return np.empty(0), np.empty(0)
    scan = wave if filtered else detection_bandpass(wave, fps)
    positions = refine(scan, beats(scan, fps, filtered=True))
    if len(positions) < 2:
        return np.empty(0), np.empty(0)
    return np.diff(positions) / fps * 1000.0, positions[1:] / fps


def reported_hr(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> float:
    """Heart rate. The readout this project reports, and the median of three others.

    The members are the spectral peak over a rectangular 8x-padded periodogram, the
    median inter-beat interval, and the mean inter-beat interval. The middle of the
    three is returned.

    Each member has one failure the other two do not share. The spectrum locks onto a
    harmonic or a subharmonic and misses by tens of bpm. Beat timing counts a dicrotic
    notch as a beat. The mean interval moves by a whole 1/(N-1) of the rate when a
    single peak is spurious. Taking the middle value discards whichever member is
    furthest out, which is the one that failed, without needing to know which it was.

    Over the 1569 test windows in `build/readout_test_s900.npz`, with beats detected
    in 0.75-4 Hz:

    | readout | MAE | RMSE | rho |
    |---|---|---|---|
    | reported_hr, median of three | 3.41 | 7.28 | 0.834 |
    | mean of three | 3.43 | 7.14 | 0.839 |
    | spectral peak, rectangular, 8x pad | 3.25 | 8.01 | 0.800 |
    | interval, median IBI | 3.86 | 7.73 | 0.821 |
    | interval, mean IBI | 4.03 | 8.00 | 0.795 |

    The vote beats each of its members on RMSE and rho, which is what it is for. It
    does not beat the mean of the same three: under this detection band the mean is
    ahead by 0.15 RMSE and 0.005 rho and behind by 0.02 MAE. Under the previous
    detector the median led on all three, so that ordering is a property of the
    current configuration rather than a settled result. `src.cli readout` reruns it.

    Non-finite members are dropped rather than propagated. A combination that returned
    nothing whenever one member returned nothing would be scored on a smaller and
    easier set of windows than its members.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    # Two bands, deliberately. The spectral member reads the reporting band, where a
    # fundamental is isolated; the interval members read the detection band, where a
    # beat has a shape. Filtering once for both would cost one of them.
    narrow = wave if filtered else bandpass(wave, fps)
    wide = wave if filtered else detection_bandpass(wave, fps)
    members = np.array(
        [
            spectral_hr(narrow, fps, filtered=True, pad=8, window="boxcar"),
            interval_hr(wide, fps, filtered=True),
            interval_hr(wide, fps, filtered=True, aggregate=np.mean),
        ],
        dtype=np.float64,
    )
    usable = members[np.isfinite(members)]
    return float(np.median(usable)) if usable.size else float("nan")


# There is no TMCC floor, and there was one until it was measured. Over the 1569
# windows of `build/readout_test_s900.npz`, template-matching correlation through the
# detection band distributes as:
#
#                   p05     p25     med     p75     p95
#   predicted      0.678   0.783   0.853   0.916   0.971
#   contact PPG    0.640   0.845   0.949   0.979   0.991
#   white noise    0.716   0.783   0.829   0.869   0.919
#
# The predicted windows and white noise overlap across the whole range -- their
# medians differ by 0.024 -- so no threshold separates a prediction worth reading from
# one that is noise. A gate at 0.8 would have passed most of the noise while reading
# as though it had excluded something. TMCC is reported by `hrv` and gates nothing.
#
# It does separate contact PPG from noise, which is the signal Charlton et al. 2025
# measured it on. Whether it can be made to work on a predicted waveform is open; the
# distributions above are the evidence that it does not work as it stands.

# Shortest recording a variability is reported from. RMSSD is a root mean square over
# N successive differences, so its own sampling error falls as 1/sqrt(2N): ten percent
# precision needs about fifty differences, which at 72 bpm is fifty beats and 42 s.
# 60 s rounds that up and is the floor the ultra-short-term HRV literature puts RMSSD
# at. SDNN and pNN50 need minutes rather than seconds and are not separately gated --
# `seconds` is returned so a caller can see what a number was computed from.
#
# This is not a standards threshold. The Task Force short-term recording is 5 minutes,
# and nothing this pipeline produces from a 300-frame window is comparable to one.
HRV_MIN_SECONDS = 60.0
# Beats are found and shapes are judged in this band, deliberately not the 0.75-2.5 Hz
# band every rate is read in. Charlton et al. 2022 detect beats with a 4th-order
# zero-phase Butterworth over 0.5-8.0 Hz; a band chosen to isolate a fundamental is
# the wrong band to find a beat or judge a wave shape in, because it removes the
# harmonics the shape is made of.
#
# The edges are narrower than that paper's, and both were swept rather than argued.
# `reported_hr` over the 1569 windows of `build/readout_test_s900.npz`:
#
#   detector     band        MAE    RMSE    rho
#   find_peaks   0.75-2.5   3.289   6.235   0.870     <- what this replaced
#   msptd        0.75-2.5   3.234   6.777   0.850
#   find_peaks   0.75-4.0   3.838   6.944   0.828
#   msptd        0.75-3.0   3.361   7.072   0.838
#   msptd        0.75-4.0   3.409   7.285   0.834     <- here
#   msptd        0.75-5.0   3.556   8.304   0.805
#   msptd        0.5-8.0    3.709   8.708   0.789     <- the paper's band
#
# On that split the old pair scores better, and this is a deliberate trade rather than
# an improvement the numbers show. The split does not sample what the ceiling buys:
# its contact-PPG rate has a p99 of 118 bpm and only 7 of 1569 windows sit above
# 150 bpm, which is where 2.5 Hz stops being able to answer at all. 4 Hz is 240 bpm,
# past the top of any pulse a subject survives, and a readout that cannot report a
# tachycardia is wrong in a way an RMSE over resting subjects does not price.
#
# 0.75 rather than 0.5 at the bottom, and 4 not 8 at the top, because both extensions
# measured worse and neither holds much: the predicted windows carry 1.2% of their
# power below 0.75 Hz and 0.9% above 5 Hz.
#
# The band also decides whether the quality metric works at all. Scored through
# 0.75-2.5 Hz, band-passed white noise returns TMCC 0.849 and a medium-quality trace
# 0.964 -- that filter shapes noise into a clean sinusoid and every beat then matches
# every other. Through this band the same two return 0.565 and 0.803.
DETECTION_LOW_HZ, DETECTION_HIGH_HZ = 0.75, 4.0
DETECTION_ORDER = 4


def detection_bandpass(wave: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Detrend, then 4th-order Butterworth over 0.75-4 Hz. For beats and shape.

    Linearly detrended, unlike `bandpass`, which uses the toolbox's smoothness prior.
    The prior is a high-pass with a corner that moves with the signal length, and at
    45 bpm it lands on the pulse: MSPTD reads 22 beats from a 15-cycle 45 bpm trace
    detrended that way and 14 from the same trace detrended linearly. A straight line
    is also what ppg-beats removes before its own scalogram.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 3 * DETECTION_ORDER + 1 or not np.isfinite(wave).all():
        return wave
    b, a = sps.butter(
        DETECTION_ORDER,
        [DETECTION_LOW_HZ / fps * 2, DETECTION_HIGH_HZ / fps * 2],
        btype="bandpass",
    )
    return sps.filtfilt(b, a, np.double(sps.detrend(wave)))


def _pulse_onsets(trace: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Index of the systolic upslope mid-point before each peak but the first.

    Charlton et al. 2025 centre each pulse wave on this point rather than on the
    peak. The peak of a corrupted beat is the part of it that moved, so centring
    there would align the noise and score it as agreement; the upslope mid-point is
    the steepest part of the wave and the most stable landmark on it.
    """
    midpoints = []
    for start, peak in itertools.pairwise(peaks):
        segment = trace[start:peak]
        if segment.size < 3:
            continue
        onset = start + int(np.argmin(segment))
        upslope = trace[onset:peak]
        if upslope.size < 2:
            continue
        crossings = np.flatnonzero(upslope >= 0.5 * (trace[onset] + trace[peak]))
        if crossings.size:
            midpoints.append(onset + int(crossings[0]))
    return np.array(midpoints, dtype=int)


def template_match(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> float:
    """Template-matching correlation coefficient: how alike this signal's beats are.

    Charlton et al. 2025, "Determinants of photoplethysmography signal quality at the
    wrist", PLOS Digital Health 4(6) e0000585, section "PPG signal processing".
    Segment the trace into one window per beat, each the median inter-beat interval
    long and centred on a systolic upslope mid-point; average them into a template;
    correlate every beat against it; return the mean correlation.

    It needs no reference signal, which is what makes it usable on a video with no
    contact PPG beside it. `macc` measures a prediction against a target and answers
    a different question -- whether the model was right. This answers whether the
    trace is self-consistent enough for a beat-to-beat number to mean anything.

    The paper's Fig 4 is the argument for preferring it to SNR: a clean trace with an
    irregular rhythm scores 0.0 dB and a noisy one scores 8.2 dB, because SNR reads
    the spectrum and quality lives in the pulse wave.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    # `filtered` here means the caller has already narrowed the trace to the cardiac
    # band, and the harmonics this metric reads are gone. The number is then inflated
    # -- everything lands between 0.85 and 1.00 -- and the gate barely gates. Pass the
    # raw wave to get the measurement the paper describes.
    if not filtered:
        wave = detection_bandpass(wave, fps)
    peaks = beats(wave, fps, filtered=True)
    positions = refine(wave, peaks)
    if len(positions) < 4:
        return float("nan")
    width = round(float(np.median(np.diff(positions))))
    half = width // 2
    if half < 2:
        return float("nan")
    centres = _pulse_onsets(wave, peaks)
    windows = np.array(
        [
            wave[c - half : c - half + 2 * half]
            for c in centres
            if c - half >= 0 and c - half + 2 * half <= wave.size
        ]
    )
    # Three beats is the fewest that makes the metric mean anything: with two, each
    # beat is half of the template it is scored against.
    if windows.ndim != 2 or len(windows) < 3:
        return float("nan")
    template = windows.mean(axis=0)
    if np.ptp(template) == 0:
        return float("nan")
    scores = [
        float(np.corrcoef(w, template)[0, 1]) for w in windows if np.ptp(w) > 0
    ]
    finite = [s for s in scores if np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def hrv(wave: np.ndarray, fps: float = 30.0, filtered: bool = False) -> dict[str, float]:
    """Time-domain heart-rate variability from the predicted BVP.

    SDNN is the spread of the intervals and RMSSD the spread of their successive
    differences; RMSSD is the beat-to-beat measure and the one short windows support.
    pNN50 is the fraction of successive differences over 50 ms.

    Three cautions travel with these numbers, and none of them is a bug to fix:

    - **The sampling floor is 33.3 ms.** One frame at 30 fps. `refine` interpolates
      each peak between samples, which is what makes a 50 ms threshold meaningful at
      all, but a real RMSSD of 20 ms is below what this can separate from its own
      jitter. Contact devices sample at 250-1000 Hz for this reason.
    - **A window this short is not a clinical SDNN.** The Task Force standard is a
      5-minute recording. `seconds` is returned so a caller can see what it got.
    - **One miscounted beat dominates.** A doubled beat puts one half-interval and
      one interval-and-a-half into the series, and RMSSD is a root mean square, so
      that single pair can exceed every genuine difference in the window. The notch
      guard in `beats` removes the periodic case; it does not remove the isolated one.
    - **The band-pass takes about a third of it.** Variability is a modulation of the
      beat, and an alternation between two intervals sits near 0.6 Hz, below the
      0.75 Hz corner every reported waveform passes through. A 100 ms alternation
      measures 92 ms on the raw wave and 64 ms once band-passed. The bias is toward a
      calmer subject than the recording holds, and nothing here corrects it: the
      correction depends on where the subject's own variability sits in the band,
      which is not known per window. `tests/test_postprocess.py` pins the size.

    `n_beats` and `seconds` are returned unconditionally so these can be judged.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    nn, _ = intervals_ms(wave, fps, filtered)
    quality = template_match(wave, fps, filtered)
    empty = {
        "sdnn_ms": float("nan"), "rmssd_ms": float("nan"), "pnn50": float("nan"),
        "mean_nn_ms": float("nan"), "n_beats": float(len(nn) + 1 if len(nn) else 0),
        "seconds": float(wave.size / fps), "tmcc": quality,
    }
    # SDNN needs two intervals to have a spread and RMSSD two successive differences
    # to be a root mean square of more than one number. Below that the series is a
    # heart rate, not a variability, and reporting 0.0 would read as a metronomic
    # subject rather than as an unmeasured one.
    if len(nn) < 3:
        return empty
    # `tmcc` is reported and gates nothing; see the distributions above the constants
    # for why. The duration gate is arithmetic rather than empirical and does gate.
    if empty["seconds"] < HRV_MIN_SECONDS:
        return empty
    successive = np.diff(nn)
    return empty | {
        "sdnn_ms": float(np.std(nn, ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(successive**2))),
        "pnn50": float(np.mean(np.abs(successive) > NN50_MS)),
        "mean_nn_ms": float(np.mean(nn)),
    }


def _band_rate(series: np.ndarray, times: np.ndarray) -> float:
    """Dominant respiratory-band frequency of an unevenly sampled series, in brpm.

    The series is resampled onto a uniform grid before the spectrum, because a
    periodogram of samples taken once per beat would read the beat rate's own
    irregularity as a frequency.
    """
    span = times[-1] - times[0]
    if span < RESP_MIN_SECONDS or series.size < 4:
        return float("nan")
    grid = np.arange(times[0], times[-1], 1.0 / IBI_RESAMPLE_HZ)
    if grid.size < 8:
        return float("nan")
    uniform = np.interp(grid, times, series)
    uniform = uniform - uniform.mean()
    if np.ptp(uniform) == 0:
        return float("nan")
    b, a = sps.butter(
        BUTTER_ORDER,
        [RESP_LOW_HZ / IBI_RESAMPLE_HZ * 2, RESP_HIGH_HZ / IBI_RESAMPLE_HZ * 2],
        btype="bandpass",
    )
    uniform = sps.filtfilt(b, a, uniform)
    nfft = 8 * (1 << int(np.ceil(np.log2(uniform.size))))
    freqs, power = sps.periodogram(
        uniform, fs=IBI_RESAMPLE_HZ, nfft=nfft, detrend=False
    )
    inside = np.flatnonzero((freqs >= RESP_LOW_HZ) & (freqs <= RESP_HIGH_HZ))
    if inside.size == 0:
        return float("nan")
    return float(freqs[inside[np.argmax(power[inside])]] * 60.0)


def respiratory_rate(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> dict[str, float]:
    """Breaths per minute, read off the beat series rather than off the waveform.

    Breathing reaches a PPG three ways: it modulates the rate (respiratory sinus
    arrhythmia), the pulse amplitude, and the baseline. Only the first two are
    available here. The baseline route needs 0.1-0.5 Hz content in the signal itself,
    and this project band-passes at 0.75-2.5 Hz before anything is read -- and the
    model cannot supply it either, because the DF-FFN's Gaussian mask is constrained
    to the same cardiac band, so a breath has no path to the output as a slow trend.
    RSA and amplitude modulation survive because both are carried by the beats.

    `brpm` is the median of whichever routes returned a number.

    **These are not validated against ground truth.** DATASETS.md lists respiration
    labels only in MCD-rPPG, whose video is the corpus this project rejected at a
    4.4% pass rate, so there is no paired video and breathing label to score against.
    Every heart-rate number in README.md came from a sweep against contact PPG;
    nothing of the kind exists for these, and they are output, not measurement.

    Amplitude modulation carries a further caveat: `predict.stitch` z-scores each
    window separately, so on a multi-window trace the amplitude series has a step at
    every seam that is an artefact of the stitch.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    seconds = float(wave.size / fps)
    if not filtered and wave.size >= 8 and np.isfinite(wave).all():
        wave = detection_bandpass(wave, fps)
    nn, times = intervals_ms(wave, fps, filtered=True)
    empty = {
        "brpm": float("nan"), "brpm_rsa": float("nan"), "brpm_riav": float("nan"),
        "seconds": seconds, "tmcc": float("nan"),
    }
    quality = template_match(wave, fps, filtered=True)
    empty["tmcc"] = quality
    if len(nn) < 4:
        return empty
    rsa = _band_rate(nn, times)
    # The amplitude route reads the height of each beat. Peak positions are taken
    # again at integer resolution here because a sample value is wanted, not a
    # position: interpolating between two samples would invent an amplitude.
    peaks = beats(wave, fps, filtered=True)
    riav = (
        _band_rate(wave[peaks[1:]], peaks[1:] / fps)
        if len(peaks) == len(nn) + 1 else float("nan")
    )
    both = np.array([rsa, riav], dtype=np.float64)
    usable = both[np.isfinite(both)]
    return empty | {
        "brpm_rsa": rsa,
        "brpm_riav": riav,
        "brpm": float(np.median(usable)) if usable.size else float("nan"),
    }
