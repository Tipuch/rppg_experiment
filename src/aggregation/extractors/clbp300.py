"""CLBP-300 extractor.

The only source we hold that carries blood pressure. Labels are encoded in the
filename and confirmed by the dataset spec sheet:

    Subject001_M44_138_99_66_448.mov
    subject___ sex+age  SBP DBP HR  lux

There is no waveform, so SBP/DBP/HR are one scalar per video. They are broadcast
across that video's windows with granularity "clip" -- every window of a clip
then shares an identical label, which is why splits must group by subject_id.
No respiration or beat-to-beat data, so BR and HRV stay null.

Frames are 4K at 60 fps. Holding a decoded clip would need ~48 GB, so this
decodes streaming: one pass to find the face box, one to crop and resize.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..face import apply_box, median_face_box
from ..schema import TARGET_FPS
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

SOURCE = "clbp300"
NAME_RE = re.compile(
    r"^(?P<subject>Subject\d+)_(?P<sex>[MF])(?P<age>\d+)_"
    r"(?P<sbp>\d+)_(?P<dbp>\d+)_(?P<hr>\d+)_(?P<lux>\d+)$"
)


def parse_labels(stem: str) -> dict | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    g = m.groupdict()
    return {
        "subject": g["subject"],
        "sbp": float(g["sbp"]),
        "dbp": float(g["dbp"]),
        "hr": float(g["hr"]),
    }


def extract_clip(video_path: Path, store_dir: Path) -> list[dict]:
    labels = parse_labels(video_path.stem)
    if labels is None:
        raise ValueError(f"malformed CLBP-300 filename: {video_path.name}")

    # ffmpeg resamples 60 -> 30 fps and downscales in one pass, so the 4K source
    # never materialises in full (a decoded clip would be ~48 GB).
    clip_id = f"{SOURCE}/{labels['subject']}"
    cached = existing_frame_store(store_dir, clip_id)
    if cached is not None:
        store, picks = cached
    else:
        got = read_frames(video_path, TARGET_FPS)
        if got is None:
            return []
        raw, _, _ = got
        frames = resize_frames(apply_box(raw, median_face_box(list(raw))))
        picks = select_windows(frames.shape[0])
        if not picks:
            return []
        store = write_frame_store(
            gather_windows(frames, picks), store_dir, clip_id, picks
        )

    signals = ClipSignals(
        hr_bpm=labels["hr"],
        sbp_mmhg=labels["sbp"],
        dbp_mmhg=labels["dbp"],
        hr_granularity="clip",
        bp_granularity="clip",
    )
    return window_rows(
        clip_id=clip_id,
        source=SOURCE,
        subject_id=labels["subject"],
        picks=picks,
        frames_path=store,
        modality="rgb",
        signals=signals,
        compression="h264",
    )


def extract_all(root: Path, store_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(root.glob("*.mov")):
        try:
            rows.extend(extract_clip(p, store_dir))
        except Exception as exc:  # noqa: BLE001 - one illegible clip must not end the pass
            # One malformed filename must not discard the whole source.
            print(f"    clbp300 skip {p.name}: {type(exc).__name__}: {exc}")
    return rows
