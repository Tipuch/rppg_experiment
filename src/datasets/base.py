"""What every corpus reader has to provide, and the helpers they share.

A reader is a module, not a class -- the rest of this codebase is function-first
and a class here would buy nothing but ceremony. Each one exposes:

    NAME                     the `source` value it writes into the manifest
    discover() -> pl.DataFrame   one row per recording, SESSION_COLUMNS at minimum
    load_ppg(clip_dir, video_path, fps) -> (times, values) | None

and, when the corpus does not ship a container ffmpeg can open, an additional
`prepare()` that writes the frame cache itself. MR-NIRP is the only such corpus:
it ships directories of PGM stills inside nested zips.

`discover` returns a DataFrame rather than yielding tuples because everything
downstream of it -- pairing streams, joining labels, counting coverage, assigning
splits -- is a table operation, and doing those in Python dicts is how the
per-corpus blocks in the manifest builder grew to look like four different
programs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import polars as pl

# The columns every reader must produce. Extra columns are allowed and preserved
# -- MR-NIRP carries the zip coordinates of its two streams through here -- but
# these are the ones the manifest builder reads.
SESSION_COLUMNS: dict[str, pl.DataType] = {
    "clip_id": pl.String,
    "source": pl.String,
    "subject_id": pl.String,
    "video_path": pl.String,
    "hr_bpm": pl.Float64,
    "sbp_mmhg": pl.Float64,
    "dbp_mmhg": pl.Float64,
}

# Credible heart rates. A label outside this is a sensor dropout, not a person.
HR_VALID = (30.0, 220.0)


class Reader(Protocol):
    NAME: str

    def discover(self) -> pl.DataFrame: ...

    def load_ppg(
        self, clip_dir: Path, video_path: Path | None, fps: float | None
    ) -> tuple[np.ndarray, np.ndarray] | None: ...


def empty_sessions() -> pl.DataFrame:
    """An empty session table with the canonical dtypes."""
    return pl.DataFrame(schema=SESSION_COLUMNS)


def sessions_from(rows: list[dict]) -> pl.DataFrame:
    """Rows to a session table, filling absent targets with null.

    Absent rather than zero: UBFC and MR-NIRP have no blood pressure, and a
    0 mmHg reading is a number a loss function trains on without complaint.
    """
    if not rows:
        return empty_sessions()
    frame = pl.DataFrame(rows)
    for column, dtype in SESSION_COLUMNS.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return frame.cast({k: v for k, v in SESSION_COLUMNS.items()})


def median_of_valid(values: np.ndarray, band: tuple[float, float] = HR_VALID) -> float | None:
    """Median over physiologically credible samples, or None if there are none.

    Median, not mean, and filtered first. UBFC's released HR trace drops to 1 bpm
    on five subjects; the raw mean was 12.2 bpm from a PPG-derived estimate and
    the median-of-valid is 3.07.
    """
    inside = values[(values >= band[0]) & (values <= band[1])]
    return float(np.median(inside)) if inside.size else None


# --------------------------------------------------------------------------- #
# Still-image sequences
#
# Only MR-NIRP needs these today. They live here rather than in mrnirp.py because
# a PGM sequence is a container format, not a property of that corpus, and the
# next dataset that ships stills should not have to reimplement the parser.
# --------------------------------------------------------------------------- #
def read_pgm16(blob: bytes) -> np.ndarray:
    """One binary PGM (P5) as an (H, W) big-endian uint16 array.

    Hand-rolled because `cv2.imdecode` returns 8-bit for these, which would throw
    away the low four bits of a 12-bit sensor reading before anything downstream
    saw them -- and a pulse is 0.1-0.5 LSB.
    """
    if blob[:2] != b"P5":
        raise ValueError("not a binary PGM")
    fields: list[int] = []
    at = 2
    while len(fields) < 3:
        while blob[at : at + 1].isspace():
            at += 1
        if blob[at : at + 1] == b"#":
            while blob[at : at + 1] not in (b"\n", b""):
                at += 1
            continue
        end = at
        while not blob[end : end + 1].isspace():
            end += 1
        fields.append(int(blob[at:end]))
        at = end
    width, height, maxval = fields
    if maxval < 256:
        raise ValueError(f"expected a 16-bit PGM, got maxval {maxval}")
    return np.frombuffer(
        blob, dtype=">u2", count=width * height, offset=at + 1
    ).reshape(height, width)


def detect_shift(frames: list[np.ndarray]) -> int:
    """Bits to drop to reach 8-bit: 8 if 12-bit data is left-shifted, else 4.

    Checked, not assumed. A right-aligned 12-bit frame reduced by 8 comes out
    almost black, which trains quietly to a flat prediction instead of raising.
    """
    sampled = np.concatenate([f.ravel() for f in frames])
    if (sampled & 0xF).any():
        return 8 if sampled.max() > 4095 else 4
    return 8


BAYER_CANDIDATES = {
    "BayerBG": cv2.COLOR_BayerBG2BGR,
    "BayerGB": cv2.COLOR_BayerGB2BGR,
    "BayerRG": cv2.COLOR_BayerRG2BGR,
    "BayerGR": cv2.COLOR_BayerGR2BGR,
}


def to_bgr(raw: np.ndarray, shift: int, code: int) -> np.ndarray:
    """One raw Bayer frame reduced to 8-bit and demosaiced to BGR."""
    return cv2.cvtColor((raw >> shift).astype(np.uint8), code)


def bayer_modulation(raw: np.ndarray) -> float:
    """How strongly a frame is mosaiced, as a fraction of its own mean level.

    A colour-filter array makes the four positions of each 2x2 cell sample
    different spectra, so their means separate. A monochrome sensor's four
    positions are the same detector and their means agree to within noise.
    Measured on MR-NIRP: genuine Bayer frames modulate 20-30% of the mean,
    monochrome ones 0.05%. Two orders of magnitude, so the test needs no
    calibration.
    """
    means = [float(raw[i::2, j::2].mean()) for i in (0, 1) for j in (0, 1)]
    level = float(np.mean(means))
    return (max(means) - min(means)) / max(level, 1.0)


# Below this a stream is monochrome, not colour. The gap between the two
# populations is ~500x, so the threshold's exact value cannot matter.
BAYER_MODULATION_MIN = 0.01


def choose_bayer_code(
    raw: np.ndarray, shift: int, box: tuple[int, int, int, int] | None
) -> tuple[str, dict[str, float]]:
    """Recover the colour-filter pattern from the pixels. Returns name and margins.

    A raw Bayer frame carries no record of its own tile, and guessing wrong swaps
    red with blue -- which for rPPG is not cosmetic, since the pulse is largest in
    green and the two chroma channels carry different noise.

    Two tests, run in order:

      1. *Green parity.* Two of the four positions in each 2x2 cell are green and
         sit on a diagonal, so their means nearly match. Whichever diagonal that
         is rules out half the candidates.
      2. *Red above blue on the face.* Of the two survivors, the correct one puts
         more red than blue on skin.

    Test 2 is **R > B, not R > G > B**. MR-NIRP Car is lit by 940/975 nm
    illuminators through a garage, where green outruns red on skin; demanding the
    full ordering rejects the correct pattern on every Car session.
    """
    parity = {p: float(raw[p[0] :: 2, p[1] :: 2].mean())
              for p in [(0, 0), (0, 1), (1, 0), (1, 1)]}
    greens_antidiagonal = (
        abs(parity[(0, 1)] - parity[(1, 0)]) < abs(parity[(0, 0)] - parity[(1, 1)])
    )
    survivors = ["BayerBG", "BayerRG"] if greens_antidiagonal else ["BayerGB", "BayerGR"]

    x, y, side = (box[0], box[1], box[2]) if box else (0, 0, raw.shape[0])
    margins = {}
    for name in survivors:
        face = to_bgr(raw, shift, BAYER_CANDIDATES[name])[y : y + side, x : x + side]
        margins[name] = float(face[:, :, 2].mean()) - float(face[:, :, 0].mean())
    return max(margins, key=lambda n: margins[n]), margins
