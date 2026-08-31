"""UBFC-rPPG: the only corpus here with a confirmed pulse in the pixels.

50 subjects in two subsets, uncompressed rawvideo AVI, heart rate and contact PPG
but no blood pressure.

Two label layouts, and they are not interchangeable:

    DATASET_2  <clip>/ground_truth.txt   3 rows: PPG, HR bpm, timestamps (s)
    DATASET_1  <clip>/gtdump.xmp         4 cols: time ms, HR, SpO2, PPG

DATASET_2's labels align 1:1 with frames -- all 42 subjects have exactly as many
samples as the video has frames. DATASET_1's do not: its oximeter runs at 62 Hz
against 28.67 fps, so column 0 (ms) is the only correct time axis.

The frame rate is not 30 and not constant. DATASET_1 runs at 28.67; DATASET_2
spans 28.77-29.98 except subjects 25, 26 and 27, which run at 23.2-23.4. Nothing
here assumes a rate -- `clips.build_clip` probes each file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .base import median_of_valid, sessions_from

NAME = "ubfc"
ROOT = Path("datasets/ubfc-rppg")
SUBSETS = ("DATASET_1", "DATASET_2")


def _clip_hr(clip_dir: Path) -> float | None:
    """Clip-level heart rate from whichever label layout this subset uses.

    The traces contain dropouts as low as 1 bpm, so this is a median over
    physiological samples rather than a mean. See `base.median_of_valid`.
    """
    ground_truth = clip_dir / "ground_truth.txt"
    if ground_truth.exists():
        arr = np.loadtxt(ground_truth)
        if arr.ndim != 2 or arr.shape[0] < 3:
            return None
        return median_of_valid(arr[1])

    dump = clip_dir / "gtdump.xmp"
    if not dump.exists():
        return None
    arr = np.loadtxt(dump, delimiter=",")
    if arr.ndim != 2 or arr.shape[1] < 4:
        return None
    return median_of_valid(arr[:, 1])


def discover() -> pl.DataFrame:
    rows: list[dict] = []
    for subset in SUBSETS:
        base = ROOT / subset
        if not base.is_dir():
            continue
        for clip_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            video = clip_dir / "vid.avi"
            if not video.exists():
                continue          # label present, video not downloaded yet
            hr = _clip_hr(clip_dir)
            if hr is None:
                continue
            rows.append({
                "clip_id": f"{NAME}/{subset}_{clip_dir.name}",
                "source": NAME,
                "subject_id": f"{NAME}_{clip_dir.name}",
                "video_path": str(video),
                "hr_bpm": hr,
            })
    return sessions_from(rows)


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contact PPG as (seconds from clip start, values), or None."""
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
