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
    hr_truth = heart_rate(truth_filtered, fps, filtered=True)
    return {
        "hr_pred": heart_rate(pred_filtered, fps, filtered=True),
        "hr_true": hr_truth,
        "snr": snr(pred_filtered, hr_truth, fps, filtered=True),
        "macc": float(macc(pred_filtered, truth_filtered)),
    }
