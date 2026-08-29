"""Shared pieces for per-source extractors.

Each extractor turns one clip into: a .npy frame store of shape
(T, FRAME_H, FRAME_W, C) uint8, plus a list of window rows matching SCHEMA.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

from ..schema import (
    FRAME_H,
    FRAME_W,
    MAX_WINDOWS_PER_CLIP,
    TARGET_FPS,
    WINDOW_FRAMES,
    WINDOW_S,
)


@dataclasses.dataclass
class ClipSignals:
    """Per-clip labels. Scope is the strict intersection: HR, SBP, DBP."""

    hr_bpm: float | None = None
    sbp_mmhg: float | None = None
    dbp_mmhg: float | None = None
    hr_granularity: str = "absent"
    bp_granularity: str = "absent"


def resize_frames(frames: np.ndarray) -> np.ndarray:
    """Resize (T, H, W, C) to (T, FRAME_H, FRAME_W, C) uint8."""
    if frames.dtype != np.uint8:
        lo, hi = float(frames.min()), float(frames.max())
        scale = 255.0 / (hi - lo) if hi > lo else 0.0
        frames = ((frames - lo) * scale).astype(np.uint8)
    t, h, w, c = frames.shape
    if (h, w) == (FRAME_H, FRAME_W):
        return np.ascontiguousarray(frames)
    out = np.empty((t, FRAME_H, FRAME_W, c), dtype=np.uint8)
    for i in range(t):
        # INTER_AREA is the correct choice for downscaling; it averages rather
        # than samples, which preserves the low-amplitude pulse signal.
        interp = cv2.INTER_AREA if (h > FRAME_H or w > FRAME_W) else cv2.INTER_LINEAR
        r = cv2.resize(frames[i], (FRAME_W, FRAME_H), interpolation=interp)
        out[i] = r if r.ndim == 3 else r[:, :, None]
    return out


def square_crop(frames: np.ndarray) -> np.ndarray:
    """Centre-crop (T, H, W, C) to a square. Fallback when no face box exists."""
    _, h, w, _ = frames.shape
    if h == w:
        return frames
    side = min(h, w)
    top, left = (h - side) // 2, (w - side) // 2
    return frames[:, top : top + side, left : left + side, :]


def frame_store_path(out_dir: Path, clip_id: str) -> Path:
    return out_dir / f"{clip_id.replace('/', '__')}.npy"


def _picks_path(out_dir: Path, clip_id: str) -> Path:
    return out_dir / f"{clip_id.replace('/', '__')}.picks.json"


def existing_frame_store(
    out_dir: Path, clip_id: str
) -> tuple[Path, list[tuple[int, int]]] | None:
    """(path, picks) if this clip was already extracted, else None.

    The picks are read back from a sidecar rather than recalculated. Recomputing
    them cannot work: which windows a store keeps depends on the decoded frame
    count, which a later run has no cheap way to reproduce, so a re-run would
    label the same pixels with different timestamps.

    Decoding a 180 s clip and running face detection is the expensive part of a
    build; the labels behind it are cheap text. Reusing a finished store makes
    repeats incremental, which matters while git-lfs is still delivering videos.
    """
    path = frame_store_path(out_dir, clip_id)
    sidecar = _picks_path(out_dir, clip_id)
    if not path.exists() or not sidecar.exists():
        return None
    try:
        arr = np.load(path, mmap_mode="r")
        picks = [(int(w), int(o)) for w, o in json.loads(sidecar.read_text())]
    except (OSError, ValueError, TypeError):
        return None
    # Frame size is part of the contract; a changed FRAME_H/W invalidates the store.
    if arr.ndim != 4 or arr.shape[1:3] != (FRAME_H, FRAME_W):
        return None
    if arr.shape[0] != len(picks) * WINDOW_FRAMES:
        return None
    return path, picks


def write_frame_store(
    frames: np.ndarray, out_dir: Path, clip_id: str, picks: list[tuple[int, int]]
) -> Path:
    """Persist the sampled frames plus the picks needed to interpret them."""
    path = frame_store_path(out_dir, clip_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, frames)
    _picks_path(out_dir, clip_id).write_text(json.dumps(picks))
    return path


def select_windows(
    n_total_frames: int, max_windows: int = MAX_WINDOWS_PER_CLIP
) -> list[tuple[int, int]]:
    """Pick up to max_windows evenly-spread window starts, as source frame indices.

    A 180 s left recording is highly redundant, and keeping all 18 windows of
    every MCD clip needs ~894 GB of frame stores against the disk available.
    Three well-spread windows carry nearly the same information for ~159 GB, and
    stop MCD outweighing CLBP-300 by 18:1 on row count.

    Returns [(window_idx, src_offset)] where window_idx is the index on the
    original clip timeline, so t_start_s stays truthful about when it happened.
    """
    available = n_total_frames // WINDOW_FRAMES
    if available <= 0:
        return []
    if available <= max_windows:
        chosen = list(range(available))
    else:
        chosen = sorted({int(i) for i in np.linspace(0, available - 1, max_windows)})
    return [(w, w * WINDOW_FRAMES) for w in chosen]


def gather_windows(frames: np.ndarray, picks: list[tuple[int, int]]) -> np.ndarray:
    """Concatenate only the selected windows, so the store keeps nothing else."""
    return np.concatenate(
        [frames[off : off + WINDOW_FRAMES] for _, off in picks], axis=0
    )


def window_rows(
    *,
    clip_id: str,
    source: str,
    subject_id: str,
    picks: list[tuple[int, int]],
    frames_path: Path,
    modality: str,
    signals: ClipSignals,
    compression: str,
    per_window_hr: dict[int, float | None] | None = None,
) -> list[dict]:
    """One row per selected window.

    Windows are non-overlapping: overlapping windows that share a broadcast
    per-clip label are near-duplicates and inflate any score computed over them.

    frames_offset indexes the compacted store (0, 300, 600, ...), while
    window_idx and t_start_s refer to the original clip timeline.
    """
    rows: list[dict] = []
    for store_pos, (window_idx, _src_offset) in enumerate(picks):
        hr, hr_gran = signals.hr_bpm, signals.hr_granularity
        if per_window_hr and per_window_hr.get(window_idx) is not None:
            hr, hr_gran = per_window_hr[window_idx], "window"
        rows.append(
            {
                "clip_id": clip_id,
                "source": source,
                "subject_id": subject_id,
                "window_idx": window_idx,
                "t_start_s": window_idx * WINDOW_S,
                "t_end_s": (window_idx + 1) * WINDOW_S,
                "fps": TARGET_FPS,
                "n_frames": WINDOW_FRAMES,
                # Absolute: a loader run from elsewhere must still find the store.
                "frames_path": str(Path(frames_path).resolve()),
                "frames_offset": store_pos * WINDOW_FRAMES,
                "modality": modality,
                "frame_h": FRAME_H,
                "frame_w": FRAME_W,
                "hr_bpm": hr,
                "sbp_mmhg": signals.sbp_mmhg,
                "dbp_mmhg": signals.dbp_mmhg,
                "hr_granularity": hr_gran,
                "bp_granularity": signals.bp_granularity,
                "compression": compression,
            }
        )
    return rows


def resample_indices(n_src: int, fps_src: float) -> np.ndarray:
    """Frame indices that resample a clip from fps_src to TARGET_FPS."""
    duration = n_src / fps_src
    n_dst = int(duration * TARGET_FPS)
    return np.clip((np.arange(n_dst) / TARGET_FPS * fps_src).astype(int), 0, n_src - 1)
