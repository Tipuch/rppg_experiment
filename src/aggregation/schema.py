"""Schema for the unified rPPG table.

Scope is the strict intersection of the blood-pressure datasets that also filmed
faces: facial images, HR, SBP, DBP. Nothing else.

Out of scope, and intentionally without extractors: SCAMPS, UBFC and MR-NIRP have
facial video but no blood pressure, and BIDMC and BUT PPG have blood pressure but
no camera. Neither group can satisfy the gate below, which requires all three
targets on every row. AGGREGATION_PLAN.md records what each of them keeps.

Polars keeps the manifest and labels only. Frames live in .npy stores that rows
point at via frames_path/frames_offset -- one 128x128x300x3 window is ~14.7 MB,
so keeping pixels in the table would make it unusable.
"""

from __future__ import annotations

import polars as pl

# Every extractor must emit frames at these settings, or windows stop being
# comparable across sources.
FRAME_H = 128
FRAME_W = 128
TARGET_FPS = 30.0
WINDOW_S = 10.0
WINDOW_FRAMES = int(WINDOW_S * TARGET_FPS)
# Windows kept per recording. See select_windows() for why this is not all
# of them: full extraction of MCD needs ~894 GB of frame stores.
MAX_WINDOWS_PER_CLIP = 3

# In scope: blood pressure AND facial video.
SOURCES = ["clbp300", "mcd"]

MODALITIES = ["rgb"]
COMPRESSIONS = ["raw", "h264"]

# How a label was obtained. "clip" means it was broadcast from a per-recording
# value across every window of that recording: overlapping windows then share an
# identical label, which is why splits must group by subject.
GRANULARITIES = ["window", "clip", "absent"]

SourceEnum = pl.Enum(SOURCES)
ModalityEnum = pl.Enum(MODALITIES)
CompressionEnum = pl.Enum(COMPRESSIONS)
GranularityEnum = pl.Enum(GRANULARITIES)

SCHEMA: dict[str, pl.DataType] = {
    "clip_id": pl.String,
    "source": SourceEnum,
    "subject_id": pl.String,
    "window_idx": pl.UInt16,
    "t_start_s": pl.Float32,
    "t_end_s": pl.Float32,
    "fps": pl.Float32,
    "n_frames": pl.UInt16,
    "frames_path": pl.String,
    "frames_offset": pl.UInt32,
    "modality": ModalityEnum,
    "frame_h": pl.UInt16,
    "frame_w": pl.UInt16,
    # Targets, most important first. Every row must carry all three.
    "hr_bpm": pl.Float32,
    "sbp_mmhg": pl.Float32,
    "dbp_mmhg": pl.Float32,
    "hr_granularity": GranularityEnum,
    "bp_granularity": GranularityEnum,
    "compression": CompressionEnum,
}

TARGETS = ["hr_bpm", "sbp_mmhg", "dbp_mmhg"]

# Credible physiological ranges, checked by the validation gate.
UNIT_RANGES: dict[str, tuple[float, float]] = {
    "hr_bpm": (30.0, 220.0),
    "sbp_mmhg": (70.0, 220.0),
    "dbp_mmhg": (40.0, 130.0),
}


def empty_frame() -> pl.DataFrame:
    """An empty table with the canonical dtypes."""
    return pl.DataFrame(schema=SCHEMA)
