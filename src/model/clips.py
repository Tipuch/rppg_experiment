"""Build the clip-level manifest that the on-the-fly loader reads.

One row per recording, not per window. Windows are chosen at load time, so every
frame of every recording stays reachable instead of being fixed into a fixed
sample at build time.

YuNet finds one face box per clip and SegFace produces one median skin mask.
SegFace runs at ~20-50 ms per frame, so a 160-frame clip would cost longer than the
training step itself; both outputs are small enough to store. A mask is 256x256 bits,
so all 3692 clips together are a few MB, against the ~3 TB that materialising frames
would need.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import polars as pl

from ..aggregation.face import BOX_PAD, apply_box, median_face_box
from ..aggregation.skin import median_skin_mask
from ..aggregation.video import probe, read_keyframes
from ..paths import BUILD_ROOT

CROP = 256
DETECT_FRAMES = 24
OUT_PARQUET = BUILD_ROOT / "clips.parquet"
MASK_DIR = BUILD_ROOT / "masks"

CLIP_SCHEMA: dict[str, pl.DataType] = {
    "clip_id": pl.String,
    "source": pl.Enum(["mcd", "mrnirp", "ubfc"]),
    "subject_id": pl.String,
    "video_path": pl.String,
    "fps": pl.Float32,
    "n_frames": pl.UInt32,
    "duration_s": pl.Float32,
    "box_x": pl.UInt16,
    "box_y": pl.UInt16,
    "box_side": pl.UInt16,
    "mask_path": pl.String,
    "skin_frac": pl.Float32,
    "hr_bpm": pl.Float32,
    "sbp_mmhg": pl.Float32,
    "dbp_mmhg": pl.Float32,
}


def crop_and_resize(frames: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """Square-crop to the face box then resize to CROP, in BGR."""
    cropped = apply_box(frames, box)
    return np.stack(
        [cv2.resize(f, (CROP, CROP), interpolation=cv2.INTER_AREA) for f in cropped]
    )


def build_clip(
    clip_id: str, source: str, subject_id: str, video: Path, targets: dict[str, float]
) -> dict | None:
    info = probe(video)
    if info["fps"] <= 0 or info["duration_s"] <= 0:
        return None

    # Keyframes only: detection and segmentation need a spread across the clip, and
    # decoding every frame to sample 24 costs 10x more for no extra information.
    sample = read_keyframes(video, DETECT_FRAMES)
    if sample is None:
        return None

    box = median_face_box(list(sample), pad=BOX_PAD)
    rgb = crop_and_resize(sample, box)[:, :, :, ::-1].copy()
    mask = median_skin_mask(rgb)
    if not mask.any():
        return None

    mask_path = MASK_DIR / f"{clip_id.replace('/', '__')}.npy"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(mask_path, np.packbits(mask))

    x, y, side = (box[0], box[1], box[2]) if box else (0, 0, 0)
    return {
        "clip_id": clip_id,
        "source": source,
        "subject_id": subject_id,
        "video_path": str(video.resolve()),
        "fps": float(info["fps"]),
        "n_frames": int(info["n_frames"]),
        "duration_s": float(info["duration_s"]),
        "box_x": x,
        "box_y": y,
        "box_side": side,
        "mask_path": str(mask_path.resolve()),
        "skin_frac": float(mask.mean()),
        **targets,
    }


def _write(rows: list[dict]) -> pl.DataFrame:
    df = pl.DataFrame(rows).select(
        [pl.col(k).cast(v).alias(k) for k, v in CLIP_SCHEMA.items()]
    )
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PARQUET)
    return df


def main(limit_per_source: int | None = None, resume: bool = True) -> int:
    """Build the manifest, skipping clips already present when resume is set.

    The full pass is thousands of clips and takes hours, so it survives being
    interrupted: completed rows are reloaded from the existing parquet rather than
    recalculated. `resume=False` rebuilds from scratch.
    """
    from ..datasets import iter_sessions

    rows: list[dict] = []
    done: set[str] = set()
    if resume and OUT_PARQUET.exists():
        previous = pl.read_parquet(OUT_PARQUET)
        rows = previous.to_dicts()
        done = set(previous["clip_id"])
        print(f"resuming: {len(done)} clips already built")

    failures = 0
    processed = 0
    for clip_id, source, subject, video, targets in iter_sessions(limit_per_source):
        if clip_id in done:
            continue
        try:
            row = build_clip(clip_id, source, subject, video, targets)
        except Exception as exc:  # noqa: BLE001 - one bad clip must not end the build
            print(f"  skip {clip_id}: {type(exc).__name__}: {exc}")
            failures += 1
            continue
        if row is None:
            failures += 1
            continue
        rows.append(row)
        processed += 1
        if processed % 25 == 0 or processed == 1:
            print(f"  [{len(rows):5d}] {clip_id:40} {row['duration_s']:6.1f}s "
                  f"@{row['fps']:5.2f}fps  skin {100 * row['skin_frac']:4.1f}%", flush=True)
        # Checkpoint periodically so an interruption costs minutes, not hours.
        if processed % 200 == 0:
            _write(rows)

    if not rows:
        print("no clips built")
        return 1

    df = _write(rows)
    print(f"\n{df.height} clips, {failures} skipped -> {OUT_PARQUET}")
    print(df.group_by("source").agg(pl.len().alias("clips"),
                                    pl.col("subject_id").n_unique().alias("subjects"),
                                    pl.col("duration_s").sum().alias("total_s")))
    return 0
