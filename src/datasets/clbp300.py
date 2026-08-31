"""CLBP-300: five sample clips, labels encoded in the filename.

    Subject001_M44_138_99_66_448.mov
    subject___ sex+age  SBP DBP HR  lux

Only the free 5-subject sample is on disk; the full 300 subjects sit behind a data
use agreement. There is no waveform at all -- SBP, DBP and HR are one scalar per
video -- so this corpus cannot support per-frame supervision, which is why
`load_ppg` always returns None and `train.py` lists it in NO_WAVEFORM_SOURCES.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl

from .base import sessions_from

NAME = "clbp300"
ROOT = Path("datasets/clbp-300-sample/ClBP-300_samples")
NAME_RE = re.compile(
    r"^(?P<subject>Subject\d+)_(?P<sex>[MF])(?P<age>\d+)_"
    r"(?P<sbp>\d+)_(?P<dbp>\d+)_(?P<hr>\d+)_(?P<lux>\d+)$"
)


def parse_labels(stem: str) -> dict | None:
    found = NAME_RE.match(stem)
    if not found:
        return None
    fields = found.groupdict()
    return {
        "subject": fields["subject"],
        "sbp": float(fields["sbp"]),
        "dbp": float(fields["dbp"]),
        "hr": float(fields["hr"]),
    }


def discover() -> pl.DataFrame:
    rows: list[dict] = []
    if ROOT.is_dir():
        for video in sorted(ROOT.glob("*.mov")):
            labels = parse_labels(video.stem)
            if labels is None:
                continue
            rows.append({
                "clip_id": f"{NAME}/{labels['subject']}",
                "source": NAME,
                "subject_id": labels["subject"],
                "video_path": str(video),
                "hr_bpm": labels["hr"],
                "sbp_mmhg": labels["sbp"],
                "dbp_mmhg": labels["dbp"],
            })
    return sessions_from(rows)


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Always None: this corpus ships no waveform. Present to satisfy the protocol."""
    return None
