"""Randomised train/dev/test bucketing at 85 / 10 / 5.

Randomised over *subjects*, not rows. Two reasons this is a correctness
requirement rather than a preference:

  - Windows cut from one recording are near-duplicates of each other. Splitting
    at row level puts near-copies on both sides.
  - SBP and DBP are one clinical reading broadcast across every window of a
    recording (bp_granularity="clip"). A row-level shuffle therefore leaks the
    exact test labels into training.

Subjects are shuffled deterministically (seeded hash, no dependence on row
order or machine) and then filled greedily by row count, so the resulting split
tracks the 85/10/5 target in rows while keeping every subject whole.
"""

from __future__ import annotations

import hashlib

import polars as pl

RATIOS = {"train": 0.85, "dev": 0.10, "test": 0.05}
SPLIT_NAMES = ["train", "dev", "test"]
DEFAULT_SEED = 20260822


def _order_key(source: str, subject_id: str, seed: int) -> str:
    """Deterministic shuffle key. Stable across runs, machines and row order."""
    return hashlib.sha256(f"{seed}:{source}/{subject_id}".encode()).hexdigest()


def assign(
    df: pl.DataFrame,
    ratios: dict[str, float] = RATIOS,
    seed: int = DEFAULT_SEED,
) -> pl.DataFrame:
    """Add a `split` column of train/dev/test, assigned per (source, subject)."""
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("split"))

    groups = (
        df.group_by("source", "subject_id")
        .agg(pl.len().alias("rows"))
        .with_columns(
            pl.struct("source", "subject_id")
            .map_elements(
                lambda s: _order_key(str(s["source"]), s["subject_id"], seed),
                return_dtype=pl.String,
            )
            .alias("_key")
        )
        .sort("_key")
    )

    total = int(groups["rows"].sum())
    targets = {name: ratios[name] * total for name in SPLIT_NAMES}

    assignments: dict[tuple[str, str], str] = {}
    filled = {name: 0 for name in SPLIT_NAMES}
    for source, subject_id, n_rows, _ in groups.iter_rows():
        name = max(
            SPLIT_NAMES,
            key=lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9),
        )
        assignments[(str(source), subject_id)] = name
        filled[name] += n_rows

    return df.with_columns(
        pl.struct("source", "subject_id")
        .map_elements(
            lambda s: assignments[(str(s["source"]), s["subject_id"])],
            return_dtype=pl.String,
        )
        .alias("split")
    )


def summarise(df: pl.DataFrame) -> pl.DataFrame:
    total = df.height
    return (
        df.group_by("split")
        .agg(
            pl.len().alias("rows"),
            pl.col("subject_id").n_unique().alias("subjects"),
            pl.col("clip_id").n_unique().alias("clips"),
        )
        .with_columns((pl.col("rows") / total * 100).round(2).alias("pct_rows"))
        .sort(
            pl.col("split").replace_strict(
                {"train": 0, "dev": 1, "test": 2}, default=3, return_dtype=pl.Int8
            )
        )
    )
