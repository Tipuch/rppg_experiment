"""MCD-rPPG: 3600 recordings with complete labels and, measured, almost no pulse.

Kept wired up because the corpus is on disk and its contact PPG and 12-lead ECG
are real. The *video* is not usable for rPPG -- 159 of 3600 clips measured above
the pulse threshold, which is below the ~12% chance rate, and three cameras
filming one subject simultaneously return three different heart rates. See
DATASETS.md section B for the evidence and the suspected causes.

The waveform lives beside the video rather than inside a per-clip directory, and
carries no timestamps because it does not need them: `ppg_sync/<name>.txt` holds
one sample per video frame, so the time axis is reconstructed from the video's
own frame rate. Verified on the manifest -- 5383 samples against 5391 frames on
the first clip checked, and within a handful across a random sample of twelve.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from .base import sessions_from

NAME = "mcd"
ROOT = Path("datasets/mcd_rppg")
DB = ROOT / "db.csv"
PPG_DIR = "ppg_sync"
VIDEO_DIR = "video"

# Below this a git-lfs pointer file is still standing in for the video.
MIN_VIDEO_BYTES = 100_000


def discover() -> pl.DataFrame:
    if not DB.exists():
        return sessions_from([])

    # The size check has to happen per file, so the frame is built first and the
    # filesystem is consulted once per surviving row rather than once per row of
    # a 3600-row CSV read three times.
    catalogue = pl.read_csv(DB).with_columns(
        (pl.lit(str(ROOT) + "/") + pl.col("video").cast(pl.String)).alias("video_path")
    )
    present = catalogue.filter(
        pl.col("video_path").map_elements(
            lambda p: Path(p).exists() and Path(p).stat().st_size >= MIN_VIDEO_BYTES,
            return_dtype=pl.Boolean,
        )
    )
    return sessions_from(
        present.select(
            clip_id=pl.lit(f"{NAME}/")
            + pl.col("video_path").str.split("/").list.last().str.replace(r"\.\w+$", ""),
            source=pl.lit(NAME),
            subject_id=pl.col("patient_id").cast(pl.String),
            video_path=pl.col("video_path"),
            hr_bpm=pl.col("pulse").cast(pl.Float64),
            sbp_mmhg=pl.col("upper_ap").cast(pl.Float64),
            dbp_mmhg=pl.col("lower_ap").cast(pl.Float64),
        ).to_dicts()
    )


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Frame-synchronised PPG, timed from the video's frame rate.

    Returns None unless the video sits in a `video/` directory, which is what
    identifies a path as this corpus's. `remux.rewrite_manifest` exists because
    rewriting `video_path` alone would sever that link silently.
    """
    if video_path is None or fps is None or fps <= 0:
        return None
    if video_path.parent.name != VIDEO_DIR:
        return None
    path = video_path.parent.parent / PPG_DIR / f"{video_path.stem}.txt"
    if not path.exists():
        return None
    arr = np.loadtxt(path)
    values = arr[:, 0] if arr.ndim == 2 else arr
    if values.size < 2:
        return None
    return np.arange(values.size, dtype=np.float64) / float(fps), values.astype(np.float64)
