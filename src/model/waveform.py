"""Pulse-waveform supervision: targets, loss and heart rate readout.

Regressing a single heart rate per clip gives the model one number to fit per
recording -- about 30 of them in a training fold -- and asks it to resolve a
frequency from a 5.33 s window whose DFT bins are 11.25 bpm apart. Measured, a
spectral estimator on that window has a median error of 15.8 bpm, which is the
scale of the errors the trained models produce.

Predicting the pulse waveform instead gives one target per frame, and heart rate
is then read off the predicted waveform by FFT rather than learned. The label also
stops untruthful: a clip-level median departs from the true heart rate of any given
5.33 s window by 4.02 bpm on average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HR_BAND = (0.7, 3.5)          # 42-210 bpm


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None,
    source: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contact PPG as (times in seconds from clip start, values), or None.

    One line of dispatch over `src.datasets`, where each corpus owns its own
    label format. It used to be three inline branches here, which meant a new
    corpus had to be remembered in two files -- and forgetting this one is
    invisible: a None return makes `WindowDataset._waveform` substitute zeros,
    and the model trains against a flat target without anything raising.

    Imported inside the function because `src.datasets.mrnirp` imports
    `hr_from_waveform` from this module.
    """
    from ..datasets import load_ppg as dispatch

    return dispatch(clip_dir, video_path=video_path, fps=fps, source=source)


def sample_ppg(
    times: np.ndarray, values: np.ndarray, frame_times: np.ndarray
) -> np.ndarray:
    """PPG resampled onto frame times and standardised to zero mean, unit sd.

    Standardised because absolute PPG amplitude is a property of the sensor and
    how hard it was strapped on, and carries nothing about the pulse. The loss is
    scale-invariant too, but standardising keeps the target range stable.
    """
    sampled = np.interp(frame_times, times, values).astype(np.float32)
    sampled -= sampled.mean()
    spread = float(sampled.std())
    return sampled / spread if spread > 1e-8 else sampled


def neg_pearson(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """1 - Pearson correlation, averaged over the batch. Range [0, 2], 0 is exact.

    The standard rPPG waveform loss. Correlation ignores scale and offset, which
    is required here: the network sees skin brightness and cannot know the
    sensor's gain, so demanding absolute agreement would penalise a perfectly
    shaped prediction for being the wrong size.
    """
    predicted = predicted - predicted.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (predicted * target).sum(dim=1)
    denominator = predicted.norm(dim=1) * target.norm(dim=1)
    return (1.0 - numerator / denominator.clamp_min(1e-8)).mean()


def hr_from_waveform(
    wave: np.ndarray, fps: float = 30.0, band: tuple[float, float] = HR_BAND
) -> float:
    """Heart rate in bpm from the dominant cardiac-band frequency of a waveform.

    Zero-padded to 8x length before the FFT. Padding does not add information, but
    it interpolates the spectrum so the peak is not forced onto a coarse bin grid:
    at 160 frames and 30 fps the raw bins are 11.25 bpm apart, which is wider than
    the accuracy being requested for.
    """
    wave = np.asarray(wave, dtype=np.float64)
    wave = wave - wave.mean()
    if wave.size < 8 or not np.isfinite(wave).all() or np.ptp(wave) == 0:
        return float("nan")
    window = np.hanning(wave.size)
    spectrum = np.abs(np.fft.rfft(wave * window, n=wave.size * 8))
    freqs = np.fft.rfftfreq(wave.size * 8, d=1.0 / fps)
    inside = (freqs >= band[0]) & (freqs <= band[1])
    if not inside.any():
        return float("nan")
    return float(freqs[inside][int(np.argmax(spectrum[inside]))] * 60.0)
