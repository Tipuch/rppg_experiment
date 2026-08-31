"""One manifest over every corpus that can supervise a waveform.

    uv run python -m src.cli combine

Each corpus is built by its own command and lands in its own file -- `clips` for
UBFC, `mrnirp` for MR-NIRP, `remux` for MCD's rewritten containers. Training on
all three needs them in one table with one split, and the join is not a plain
concatenation:

  * **Columns and dtypes both differ.** MR-NIRP carries `corpus`,
    `ppg_zero_frac`, `ppg_max_gap_s` and `fps_source`; the MCD manifest carries
    `ppg_video_path`; and `fps` is Float32 in the manifests `clips` writes
    against MR-NIRP's Float64. `diagonal_relaxed` fills absent columns with typed
    nulls and widens to a common supertype, which a plain vertical concat refuses
    and a hand-rolled alignment gets wrong in the dtype direction.
  * **A corpus is taken from exactly one file.** `clips_remux.parquet` holds UBFC
    rows too, and CLBP-300 rows whose videos are no longer on disk. Selecting per
    source by name is unambiguous where a concatenate-and-dedup is not.
  * **MCD comes from the remuxed manifest**, so `video_path` points at the
    indexed container and `ppg_video_path` still points at the original -- the
    labels live beside the original and are found from its path.

The split is assigned here, over the pooled table, and **stratified by source**.
Without that a greedy fill is free to hand a whole corpus to one side: MR-NIRP's
own split did exactly that, giving dev only Car clips and test only Indoor ones.

**MCD dominates by volume and nothing here changes that.** It is 180 h against
UBFC's 0.9 and MR-NIRP's 2.0, so it is ~98% of the segments on every side.
Stratifying guarantees the other two are *present* in dev and test, not that they
matter to an aggregate over them -- which is why `evaluate` reports per source,
and why `--stride` exists to subsample the large corpus.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..paths import BUILD_ROOT
from .splits import RATIOS_MRNIRP, assign, segment_counts

# Where each corpus is looked for, best source first. Candidates rather than one
# path because the manifests are written by different commands and a user may
# have run only some: MCD is taken from the remuxed manifest when it exists,
# since seeking there is 9.5x faster, and from the plain one when it does not.
DEFAULT_PARTS: dict[str, tuple[Path, ...]] = {
    "ubfc": (BUILD_ROOT / "clips_ubfc.parquet",
             BUILD_ROOT / "clips_remux.parquet",
             BUILD_ROOT / "clips.parquet"),
    "mrnirp": (BUILD_ROOT / "clips_mrnirp.parquet",),
    "mcd": (BUILD_ROOT / "clips_remux.parquet",
            BUILD_ROOT / "clips.parquet"),
}
OUT_PARQUET = BUILD_ROOT / "clips_all.parquet"

# Rebuilt here over the pooled table, so a per-corpus split cannot leak in.
DROP_COLUMNS = ("split", "n_segments")


def _first_with_rows(source: str, candidates: tuple[Path, ...]) -> pl.DataFrame | None:
    """The first candidate file that actually holds rows for `source`."""
    for path in candidates:
        if not path.exists():
            continue
        part = pl.read_parquet(path).filter(pl.col("source").cast(pl.String) == source)
        if part.height:
            part = part.drop([c for c in DROP_COLUMNS if c in part.columns])
            # `source` is an Enum in some manifests and a String in others; the
            # union needs one dtype, and String is the one that admits a new
            # corpus without rewriting the others.
            print(f"  {source:8s} {part.height:5d} clips  "
                  f"{part['duration_s'].sum() / 3600:6.2f} h  from {path}")
            return part.with_columns(pl.col("source").cast(pl.String))
    return None


def load_parts(parts: dict[str, tuple[Path, ...]] = DEFAULT_PARTS) -> pl.DataFrame:
    """Every named corpus, from one file each, as one table with a union schema.

    A corpus with no manifest is skipped with a note rather than raised on: the
    corpora are built by separate commands and having only some is normal.
    """
    frames: list[pl.DataFrame] = []
    for source, candidates in parts.items():
        part = _first_with_rows(source, candidates)
        if part is None:
            print(f"  {source:8s} SKIP  (none of "
                  f"{', '.join(str(c) for c in candidates)})")
            continue
        frames.append(part)

    if not frames:
        raise SystemExit("no corpus manifests found; build them first")
    return pl.concat(frames, how="diagonal_relaxed").sort("clip_id")


def build(
    parts: dict[str, tuple[Path, ...]] = DEFAULT_PARTS,
    ratios: dict[str, float] = RATIOS_MRNIRP,
    seed: int = 20260822,
    n_frames: int = 300,
) -> pl.DataFrame:
    """The pooled manifest with a `split` column, balanced in segments per source."""
    pooled = load_parts(parts)
    weighted = pooled.with_columns(segment_counts(pooled, n_frames).alias("n_segments"))
    return assign(
        weighted, ratios=ratios, seed=seed, weight="n_segments",
        order="size", stratify="source",
    )
