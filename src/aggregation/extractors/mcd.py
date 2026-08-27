"""MCD-rPPG extractor.

The only source we hold that carries every target at once.

db.csv is one row per recording (3600 rows, 600 patients, 3 cameras x 2 exercise
steps) with the clinical measurements:

    upper_ap -> SBP mmHg    lower_ap -> DBP mmHg    pulse -> HR bpm

Those are clinical readings taken once per recording, so SBP and DBP broadcast at
"clip" granularity. The frame-aligned PPG gives better-resolved per-window HR:

    ppg_sync/*.txt  frame-aligned PPG, one row per video frame, (value, dt)
    meta/*.txt      per-frame timestamps, which is where true fps comes from
                    (24 fps for IriunWebcam, 30 fps for the others)

Videos are git-lfs. Recordings whose .avi is still a 133-byte pointer are skipped
and picked up on a later run once `git lfs pull` has fetched them.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import polars as pl

from ..face import apply_box, median_face_box
from ..schema import TARGET_FPS, WINDOW_FRAMES
from ..signals import hr_from_wave
from ..video import read_frames
from .base import (
    ClipSignals,
    existing_frame_store,
    gather_windows,
    resize_frames,
    select_windows,
    window_rows,
    write_frame_store,
)
from .base import (
    ClipSignals as _CS,  # noqa: F401
)

SOURCE = "mcd"
LFS_POINTER_MAX = 1024  # a real .avi is tens of MB; a pointer is ~133 bytes


def _is_lfs_pointer(path: Path) -> bool:
    return path.exists() and path.stat().st_size <= LFS_POINTER_MAX


def meta_fps(meta_path: Path) -> float | None:
    """True frame rate from per-frame timestamps; cameras differ."""
    try:
        stamps = []
        for line in meta_path.read_text().splitlines():
            parts = line.split("  ")
            if len(parts) > 1:
                stamps.append(dt.datetime.fromisoformat(parts[1].strip()))
    except (OSError, ValueError):
        return None
    if len(stamps) < 2:
        return None
    span = (stamps[-1] - stamps[0]).total_seconds()
    return (len(stamps) - 1) / span if span > 0 else None


def read_ppg_sync(path: Path) -> np.ndarray | None:
    try:
        arr = np.loadtxt(path)
    except (OSError, ValueError):
        return None
    if arr.ndim != 2 or arr.shape[0] < 2:
        return None
    return arr[:, 0].astype(np.float64)


def _source_frame_count(ppg: np.ndarray | None, fps: float | None) -> int:
    """Frames the original recording had, at TARGET_FPS."""
    if ppg is None or not fps:
        return 0
    return int(len(ppg) / fps * TARGET_FPS)


def extract_row(rec: dict, root: Path, store_dir: Path) -> list[dict]:
    patient = str(rec["patient_id"])
    clip_id = f"{SOURCE}/{Path(str(rec['video'])).stem}"

    ppg_path = root / str(rec["ppg_sync"])
    meta_path = root / str(rec["meta"])
    ppg = read_ppg_sync(ppg_path) if ppg_path.exists() else None
    fps = meta_fps(meta_path) if meta_path.exists() else None

    signals = ClipSignals(
        hr_bpm=float(rec["pulse"]),
        sbp_mmhg=float(rec["upper_ap"]),
        dbp_mmhg=float(rec["lower_ap"]),
        hr_granularity="clip",
        bp_granularity="clip",
    )

    cached = existing_frame_store(store_dir, clip_id)
    if cached is not None:
        store, picks = cached
        n_frames = _source_frame_count(ppg, fps)
    else:
        video_path = root / str(rec["video"])
        if not video_path.exists() or _is_lfs_pointer(video_path):
            return []      # still an LFS pointer; a later run will pick it up
        got = read_frames(video_path, TARGET_FPS)
        if got is None:
            return []
        raw, _, _ = got
        frames = resize_frames(apply_box(raw, median_face_box(list(raw))))
        picks = select_windows(frames.shape[0])
        if not picks:
            return []
        n_frames = frames.shape[0]
        store = write_frame_store(
            gather_windows(frames, picks), store_dir, clip_id, picks
        )

    # Frame-aligned PPG resolves HR per window; db.csv only has one clinical
    # pulse reading for the whole recording.
    per_hr: dict[int, float | None] | None = None
    if ppg is not None and n_frames > 0:
        # ppg_sync is one sample per source frame, but its length and the decoded
        # frame count disagree by a handful of samples, so map by proportion
        # rather than by an fps-derived index.
        per_sample = len(ppg) / n_frames
        ppg_fs = TARGET_FPS * per_sample
        per_hr = {}
        for window_idx, _ in picks:
            lo = int(window_idx * WINDOW_FRAMES * per_sample)
            hi = int((window_idx + 1) * WINDOW_FRAMES * per_sample)
            segment = ppg[lo:hi]
            per_hr[window_idx] = (
                hr_from_wave(segment, ppg_fs) if len(segment) > 8 else None
            )

    return window_rows(
        clip_id=clip_id,
        source=SOURCE,
        subject_id=patient,
        picks=picks,
        frames_path=store,
        modality="rgb",
        signals=signals,
        compression="h264",
        per_window_hr=per_hr,
    )


def extract_all(root: Path, store_dir: Path, limit: int | None = None) -> list[dict]:
    db = pl.read_csv(root / "db.csv")
    if limit:
        db = db.head(limit)
    rows: list[dict] = []
    failures: list[str] = []
    for rec in db.iter_rows(named=True):
        try:
            rows.extend(extract_row(rec, root, store_dir))
        except Exception as exc:  # noqa: BLE001 - one unreadable clip must not end the pass
            # One bad recording must not sink 3600 others, but a systemic failure
            # (missing ffmpeg, absent detector) would otherwise be silent.
            if len(failures) < 5:
                print(f"    mcd skip {rec.get('video')}: {type(exc).__name__}: {exc}")
            failures.append(rec.get("video"))
            continue
    if failures:
        print(f"    mcd: {len(failures)} recordings failed")
    return rows
