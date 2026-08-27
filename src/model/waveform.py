"""Pulse-waveform supervision: targets, loss and heart rate readout.

Regressing a single heart rate per clip gives the model one number to fit per
recording -- about 30 of them in a training fold -- and asks it to resolve a
frequency from a 5.33 s window whose DFT bins are 11.25 bpm apart. Measured, a
spectral estimator on that window has a median error of 15.8 bpm, which is the
scale of the errors the trained models produce.

Predicting the pulse waveform instead gives one target per frame, and heart rate
is then read off the predicted waveform by FFT rather than learned. The label also
stops lying: a clip-level median differs from the true heart rate of any given
5.33 s window by 4.02 bpm on average.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

HR_BAND = (0.7, 3.5)          # 42-210 bpm


# MCD-rPPG keeps one PPG sample per video frame in ppg_sync/<name>.txt, alongside
# video/<name>.avi. Column 0 is the amplitude, column 1 a per-frame capture
# interval that is not needed once the alignment is known.
MCD_PPG_DIR = "ppg_sync"
MCD_VIDEO_DIR = "video"


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contact PPG as (times in seconds from clip start, values), or None.

    Three formats, one per corpus that has a waveform at all:

      UBFC DATASET_2  <clip>/ground_truth.txt   3 rows: PPG, HR, timestamps (s)
      UBFC DATASET_1  <clip>/gtdump.xmp         4 cols: time ms, HR, SpO2, PPG
      MCD-rPPG        ppg_sync/<name>.txt       one sample per video frame

    MCD carries no timestamps because it does not need them: the file is already
    frame-synchronised, one sample per frame, so the time axis is reconstructed
    from the video's own frame rate. Verified on the manifest -- 5383 samples
    against 5391 frames on the first clip checked, and within a handful across a
    random sample of twelve.

    CLBP-300 is absent deliberately: its five clips ship as bare .mov files with
    the labels encoded in the filename and no waveform at all, so they cannot
    support per-frame supervision.
    """
    mcd = _load_mcd_ppg(video_path, fps)
    if mcd is not None:
        return mcd

    ground_truth = clip_dir / "ground_truth.txt"
    if ground_truth.exists():
        arr = np.loadtxt(ground_truth)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return None
        times, values = arr[2].astype(np.float64), arr[0].astype(np.float64)
    else:
        dump = clip_dir / "gtdump.xmp"
        if not dump.exists():
            return None
        arr = np.loadtxt(dump, delimiter=",")
        if arr.ndim != 2 or arr.shape[1] < 4:
            return None
        times, values = arr[:, 0] / 1000.0, arr[:, 3]
    if times.size < 2:
        return None
    return times - times[0], values


def _load_mcd_ppg(
    video_path: Path | None, fps: float | None
) -> tuple[np.ndarray, np.ndarray] | None:
    """MCD-rPPG's frame-synchronised PPG, timed from the video's frame rate."""
    if video_path is None or fps is None or fps <= 0:
        return None
    if video_path.parent.name != MCD_VIDEO_DIR:
        return None
    path = video_path.parent.parent / MCD_PPG_DIR / f"{video_path.stem}.txt"
    if not path.exists():
        return None
    arr = np.loadtxt(path)
    values = arr[:, 0] if arr.ndim == 2 else arr
    if values.size < 2:
        return None
    return np.arange(values.size, dtype=np.float64) / float(fps), values.astype(np.float64)


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
    the accuracy being asked for.
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
