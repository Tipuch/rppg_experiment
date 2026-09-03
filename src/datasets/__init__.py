"""Per-corpus readers, one module each, behind a common interface.

Before this package the same knowledge lived in three places: inline blocks in
`clips.iter_sources`, format branches in `waveform.load_ppg`, and an unwired set
of extractors under `aggregation/`. Adding a corpus meant editing all three and
finding out at training time which one had been missed -- a missing `load_ppg`
branch returns None silently, and the loader then substitutes zeros.

    REGISTRY[name].discover()  -> pl.DataFrame of recordings
    REGISTRY[name].load_ppg()  -> (times, values) | None

Order is deliberate: `iter_sessions` and `load_ppg` both walk REGISTRY in order,
and it reproduces what `clips.iter_sources` did before the move, so an existing
`build/clips.parquet` rebuilds row for row.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from . import mcd, mrnirp, ubfc
from .base import SESSION_COLUMNS, Reader, empty_sessions

REGISTRY: dict[str, Reader] = {
    ubfc.NAME: ubfc,
    mcd.NAME: mcd,
    mrnirp.NAME: mrnirp,
}

# Corpora whose frames this repo cannot decode with ffmpeg, so they build their
# own cache in `discover`-time terms and are not walked by `clips.build_clip`.
SELF_PREPARED = {mrnirp.NAME}

__all__ = ["REGISTRY", "SELF_PREPARED", "iter_sessions", "load_ppg", "sessions"]


def sessions(
    names: list[str] | None = None, limit_per_source: int | None = None
) -> pl.DataFrame:
    """Every discoverable recording across the requested corpora, as one table.

    MR-NIRP is excluded unless asked for by name: `discover` there reads ~245 GiB
    of zip indexes out of ~/Downloads, which is not something an unrelated
    `src.cli clips` run should pay for.
    """
    wanted = names if names is not None else [
        n for n in REGISTRY if n not in SELF_PREPARED
    ]
    frames = []
    for name in wanted:
        found = REGISTRY[name].discover()
        if limit_per_source:
            found = found.head(limit_per_source)
        if found.height:
            frames.append(found.select(list(SESSION_COLUMNS)))
    return pl.concat(frames, how="vertical") if frames else empty_sessions()


def iter_sessions(limit_per_source: int | None = None):
    """The tuple stream `clips.main` consumes: (clip_id, source, subject, video, targets)."""
    for row in sessions(limit_per_source=limit_per_source).iter_rows(named=True):
        yield (
            row["clip_id"], row["source"], row["subject_id"], Path(row["video_path"]),
            {k: row[k] for k in ("hr_bpm", "sbp_mmhg", "dbp_mmhg")},
        )


def load_ppg(
    clip_dir: Path, video_path: Path | None = None, fps: float | None = None,
    source: str | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Contact PPG as (seconds from clip start, values), or None if there is none.

    With `source` given the dispatch is exact. Without it every reader is tried in
    REGISTRY order, which is what the callers in `dataset.py` and `framecache.py`
    do -- they hold a path, not a corpus name. Each reader identifies its own
    layout and returns None otherwise, so the probe cannot cross-match: MCD
    requires a `video/` parent, UBFC requires one of its two label files, MR-NIRP
    requires a `pulseOx.mat`.
    """
    if source is not None:
        reader = REGISTRY.get(source)
        return None if reader is None else reader.load_ppg(clip_dir, video_path, fps)
    for reader in REGISTRY.values():
        found = reader.load_ppg(clip_dir, video_path, fps)
        if found is not None:
            return found
    return None
