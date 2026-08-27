"""Derive HR, BR and HRV from raw physiological waveforms.

Every function returns values in the units the schema declares: bpm for HR,
breaths/min for BR, ms for HRV. None is returned when the window is too short or
the signal carries no usable peak, rather than a fabricated number.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sps

# Passbands. HR 0.7-3.5 Hz is 42-210 bpm; BR 0.1-0.5 Hz is 6-30 breaths/min.
HR_BAND = (0.7, 3.5)
BR_BAND = (0.1, 0.5)

# SDNN over a short record is dominated by how many beats happened to land in it.
# Below this we return None instead of a number that looks precise and is not.
MIN_HRV_SECONDS = 20.0
MIN_HRV_BEATS = 8


def _bandpass(x: np.ndarray, fs: float, band: tuple[float, float]) -> np.ndarray:
    lo, hi = band
    nyq = fs / 2.0
    hi = min(hi, nyq * 0.99)
    if lo >= hi:
        return x - x.mean()
    sos = sps.butter(3, [lo / nyq, hi / nyq], btype="band", output="sos")
    return sps.sosfiltfilt(sos, x)


def _dominant_freq(x: np.ndarray, fs: float, band: tuple[float, float]) -> float | None:
    """Dominant frequency in Hz inside band, via Welch PSD.

    The peak bin is refined by fitting a parabola to it and its two neighbours.
    Without that, a 10 s window at 30 Hz resolves to 0.125 Hz bins -- 7.5 bpm --
    so every HR lands on a multiple of 7.5 and the target is quantised rubbish.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if x.size < 8 or not np.isfinite(x).all() or np.ptp(x) == 0:
        return None
    nperseg = min(x.size, max(64, int(fs * 8)))
    nfft = int(2 ** np.ceil(np.log2(nperseg * 4)))
    freqs, psd = sps.welch(x - x.mean(), fs=fs, nperseg=nperseg, nfft=nfft)
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not mask.any() or psd[mask].max() <= 0:
        return None

    idx = int(np.flatnonzero(mask)[np.argmax(psd[mask])])
    if 0 < idx < psd.size - 1:
        a, b, c = psd[idx - 1], psd[idx], psd[idx + 1]
        denom = a - 2.0 * b + c
        if denom != 0:
            shift = 0.5 * (a - c) / denom
            if abs(shift) <= 1.0:
                df = freqs[1] - freqs[0]
                return float(freqs[idx] + shift * df)
    return float(freqs[idx])


def hr_from_wave(wave: np.ndarray, fs: float) -> float | None:
    """Heart rate in bpm from a PPG or ECG waveform."""
    f = _dominant_freq(wave, fs, HR_BAND)
    return None if f is None else f * 60.0


def br_from_wave(wave: np.ndarray, fs: float) -> float | None:
    """Breathing rate in breaths/min from a respiration waveform."""
    f = _dominant_freq(wave, fs, BR_BAND)
    return None if f is None else f * 60.0


def detect_beats(wave: np.ndarray, fs: float) -> np.ndarray:
    """Beat sample indices from a PPG/ECG waveform."""
    x = np.asarray(wave, dtype=np.float64).ravel()
    if x.size < int(fs * 2):
        return np.empty(0, dtype=int)
    filt = _bandpass(x, fs, HR_BAND)
    std = filt.std()
    if std == 0:
        return np.empty(0, dtype=int)
    # Refractory period of 0.3 s caps detections at 200 bpm.
    peaks, _ = sps.find_peaks(filt, distance=max(1, int(fs * 0.3)), height=0.2 * std)
    return peaks


def hrv_from_wave(wave: np.ndarray, fs: float) -> tuple[float | None, float | None]:
    """(SDNN, RMSSD) in ms. Returns (None, None) when the record is too short.

    Computed per clip, not per window: SDNN over a 10 s window is unstable.
    """
    x = np.asarray(wave, dtype=np.float64).ravel()
    if x.size / fs < MIN_HRV_SECONDS:
        return None, None
    peaks = detect_beats(x, fs)
    if peaks.size < MIN_HRV_BEATS:
        return None, None
    ibi_ms = np.diff(peaks) / fs * 1000.0
    # Keep only 30-200 bpm equivalents; anything outside is a detection error.
    good = (ibi_ms >= 300.0) & (ibi_ms <= 2000.0)
    if good.sum() < MIN_HRV_BEATS - 1:
        return None, None

    sdnn = float(np.std(ibi_ms[good], ddof=1))
    # RMSSD differences successive intervals, so a pair spanning a discarded beat
    # is not a real succession. Difference first, then keep only adjacent pairs.
    adjacent = good[:-1] & good[1:]
    diffs = np.diff(ibi_ms)[adjacent]
    rmssd = float(np.sqrt(np.mean(diffs**2))) if diffs.size else None
    return sdnn, rmssd
