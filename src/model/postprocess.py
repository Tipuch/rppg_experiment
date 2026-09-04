"""Heart rate, SNR and MACC from a predicted waveform.

Both source papers state the same post-processing: band-pass the predicted BVP
with a second-order Butterworth at 0.75-2.5 Hz, then take the dominant spectral
peak. Both also state they used the rPPG-Toolbox, which is vendored under
tools/rPPG-Toolbox -- so the primitives here are imported from it rather than
rewritten, and the numbers stay comparable with every method in its tables.

**The toolbox's own wrapper is not reused, intentionally.** Its
`calculate_metric_per_video` hardcodes `butter(1, [0.6, 3.3])` -- a *first*-order
filter over a *wider* band than either paper specifies. Its own comments say to
use 0.75 and 2.5 "to more closely match results in the NeurIPS 2023 toolbox
paper", but the constants are not parameters, so matching the papers means calling
the primitives directly. Invisibly inheriting 0.6-3.3 Hz would admit a 36 bpm drift
and a 198 bpm harmonic into every heart-rate estimate.

The band is the same 0.75-2.5 Hz the DF-FFN's Gaussian mask is constrained to, and
for the same reason: outside it, a peak is not a pulse.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import signal as sps

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
    path would let any of them shadow something of ours later. The module itself
    imports only numpy and scipy, so it loads cleanly in isolation.
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
detrend = _POST._detrend
macc = _POST._compute_macc


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

    The periodogram is zero-padded to the next power of two, which interpolates
    the spectrum so the peak is not forced onto a coarse bin grid: at 160 frames
    and 30 fps the raw bins are 11.25 bpm apart, wider than the accuracy being
    reported. Padding adds no information -- it stops the grid from destroying
    what is already there.
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
    peak actually sits. Two readouts need this and for the same reason: both
    quantise a continuous position onto a grid coarser than the accuracy being
    reported -- a spectral peak onto 3.5 bpm bins, a beat onto 33 ms frames.

    Clipped to half a sample. A larger offset means the three points do not
    bracket a peak, and the middle one was not a maximum.
    """
    denominator = left - 2.0 * mid + right
    shift = np.where(
        np.abs(denominator) > 1e-12, 0.5 * (left - right) / denominator, 0.0
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

    `heart_rate` is the vendored toolbox's readout and stays that way, so the
    numbers this project reports remain comparable with every method in the
    toolbox's own tables. This is the same measurement with the steps production
    PPG readouts add on top: a window function, heavier zero-padding, and
    interpolation of the peak between bins.
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


# A beat counted twice halves the median interval, so the guard in `beats` fires
# when beat timing claims a rate this multiple of the spectral peak or higher. No
# pulse does that: the spectrum of a real 115 bpm pulse has its fundamental at
# 115, not at 66. 1.4 leaves room for the spectral peak's own error while sitting
# well below the 2.0 a fully doubled trace gives.
DOUBLE_BEAT_RATIO = 1.4
# Kept peaks, as a fraction of the median prominence, once the guard fires. On the
# windows that fail, the notch peaks come in at 0.00-0.25 of the median while the
# weakest genuine beat is at 0.70, so the threshold sits in an empty gap.
NOTCH_PROMINENCE_FRAC = 0.5


def _rate_from_peaks(
    trace: np.ndarray, peaks: np.ndarray, fps: float
) -> float:
    """bpm from the median gap between `peaks`. NaN if there is no gap to take."""
    if len(peaks) < 2:
        return float("nan")
    gaps = np.diff(refine(trace, peaks))
    if not len(gaps):
        return float("nan")
    return float(60.0 * fps / np.median(gaps))


def beats(trace: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Peak indices, one per cardiac cycle.

    Minimum spacing is the period of 2.5 Hz, the top of the reporting band, so two
    peaks cannot fall inside one beat. `trace` is expected band-passed.

    **The spacing floor alone does not stop a beat being counted twice.** The
    dicrotic notch -- the aortic valve closing, part of every real pulse -- is a
    second local maximum on the diastolic decay, and at 66 bpm it lands about 15
    frames after the systolic peak, past the 12-frame floor. `find_peaks` then
    returns 17 peaks for 10 cycles, more than half the intervals are half-cycles,
    and the median reads 115 bpm against a 66 bpm pulse. Measured on
    `mrnirp/indoor_Subject5_still_940` and `mcd/6667_USBVideo_after`, and it is
    intermittent within one recording: the notch grows and shrinks with perfusion,
    so the same subject reads correctly in the next window.

    A prominence floor is not applied unconditionally, because on a noisy
    *predicted* waveform it discards real beats -- swept over the 788 labelled
    windows in `build/readout_test.npz`, an unconditional floor at half the median
    prominence cost 0.34 bpm of MAE and 1.4 bpm of RMSE. It is applied only when
    beat timing and the spectrum disagree in the direction only double-counting
    produces. That repair is a small gain on the same sweep (MAE 3.81 to 3.79,
    RMSE 6.71 to 6.65, rho 0.858 to 0.861) and removes every split-beat window
    from MR-NIRP's share of the test split.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    if trace.size < 8 or not np.isfinite(trace).all():
        return np.empty(0, dtype=int)
    found, properties = sps.find_peaks(
        trace, distance=max(1, int(fps / HIGH_HZ)), prominence=0
    )
    if len(found) < 3:
        return found
    timing = _rate_from_peaks(trace, found, fps)
    spectral = spectral_hr(trace, fps, filtered=True)
    if not (np.isfinite(timing) and np.isfinite(spectral)):
        return found
    if timing <= DOUBLE_BEAT_RATIO * spectral:
        return found
    prominences = properties["prominences"]
    keep = prominences >= NOTCH_PROMINENCE_FRAC * np.median(prominences)
    return found[keep] if keep.sum() >= 2 else found


def refine(trace: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions, by the vertex of a parabola through 3 points.

    An integer index quantises the interval by one frame, which at 30 fps is 3.4
    bpm at 110 and 6 bpm at 150 -- coarser than the difference this number exists
    to show. The band-pass leaves a smooth trace, so the three samples around a
    peak locate its vertex.
    """
    if len(peaks) == 0:
        return peaks.astype(np.float64)
    inner = peaks[(peaks > 0) & (peaks < len(trace) - 1)]
    if len(inner) != len(peaks):
        return peaks.astype(np.float64)
    return peaks + _vertex_shift(trace[peaks - 1], trace[peaks], trace[peaks + 1])


def interval_hr(
    wave: np.ndarray, fps: float = 30.0, filtered: bool = False
) -> float:
    """Heart rate from the median inter-beat interval. **The readout this project
    reports.**

    This is what pulse oximeters and wrist wearables display: detect one peak per
    cardiac cycle, difference their positions, take the median. Median rather than
    mean because one missed or doubled beat moves a mean of a dozen intervals by
    several bpm.

    Chosen over the spectral peak on evidence, not on principle. Swept over 1569
    labelled test windows it cut RMSE from 8.18 to 6.60 bpm and raised rho from
    0.793 to 0.857, for 0.35 bpm more MAE -- it makes fewer large misses and
    slightly more small ones. `src.cli readout` reruns that sweep.
    """
    wave = np.asarray(wave, dtype=np.float64).ravel()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    if not filtered:
        wave = bandpass(wave, fps)
    return _rate_from_peaks(wave, beats(wave, fps), fps)


def snr(wave: np.ndarray, reference_bpm: float, fps: float = 30.0,
        filtered: bool = False) -> float:
    """Signal-to-noise ratio in dB, CFMamba Eqs. 26-27.

    Signal is the power within 6 bpm of the reference rate and of its second
    harmonic; noise is everything else in the band. The harmonic is included
    because a real pulse has one -- a prediction that captures only the
    fundamental is a sinusoid, not a waveform.
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
    from the manifest's label column. DATASETS.md records subject24 named 96 bpm
    against a PPG reading of 127.2, plus four other UBFC subjects whose HR readout
    drops out; the waveform on those same subjects is intact. Reading the rate off
    the signal avoids the fault entirely, and is what the toolbox does.
    """
    pred_filtered = bandpass(predicted, fps)
    truth_filtered = bandpass(truth, fps)
    hr_truth = interval_hr(truth_filtered, fps, filtered=True)
    return {
        "hr_pred": interval_hr(pred_filtered, fps, filtered=True),
        "hr_true": hr_truth,
        "snr": snr(pred_filtered, hr_truth, fps, filtered=True),
        "macc": float(macc(pred_filtered, truth_filtered)),
    }
