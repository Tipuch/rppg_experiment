"""Assemble extractor rows into one Polars table and validate it."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from .schema import SCHEMA, TARGETS, UNIT_RANGES


def build(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=SCHEMA)
    df = pl.DataFrame(rows)
    # Enum casts fail loudly on unknown values, which is the point.
    return df.select(
        [pl.col(name).cast(dtype).alias(name) for name, dtype in SCHEMA.items()]
    )


def coverage(df: pl.DataFrame) -> pl.DataFrame:
    """Non-null count per target, per source. The sparsity map."""
    return df.group_by("source").agg(
        pl.len().alias("rows"),
        pl.col("subject_id").n_unique().alias("subjects"),
        *[pl.col(t).is_not_null().sum().alias(t) for t in TARGETS],
        pl.col("hr_granularity").eq("window").sum().alias("hr_per_window"),
    ).sort("source")


def validate(df: pl.DataFrame) -> list[str]:
    """Return a list of problems. Empty list means the table passed."""
    problems: list[str] = []

    for target, (lo, hi) in UNIT_RANGES.items():
        bad = df.filter(
            pl.col(target).is_not_null() & ~pl.col(target).is_between(lo, hi)
        )
        if bad.height:
            got = bad[target].to_list()[:3]
            problems.append(
                f"{target}: {bad.height} rows outside [{lo}, {hi}] e.g. {got}"
            )

    # A target that is present must declare how it was obtained, and one that is
    # absent must not claim a granularity. Either mismatch means a broken extractor.
    pairs = [("hr_bpm", "hr_granularity"), ("sbp_mmhg", "bp_granularity")]
    for value_col, gran_col in pairs:
        mismatched = df.filter(
            (pl.col(value_col).is_not_null() & (pl.col(gran_col) == "absent"))
            | (pl.col(value_col).is_null() & (pl.col(gran_col) != "absent"))
        )
        if mismatched.height:
            problems.append(
                f"{value_col}/{gran_col}: {mismatched.height} rows disagree "
                "about whether the label exists"
            )

    incomplete = df.filter(
        pl.any_horizontal(pl.col(t).is_null() for t in TARGETS)
    )
    if incomplete.height:
        problems.append(
            f"{incomplete.height} rows are missing one of {TARGETS}; "
            "every in-scope row must carry all three"
        )

    dupes = df.group_by(["clip_id", "window_idx"]).len().filter(pl.col("len") > 1)
    if dupes.height:
        problems.append(f"{dupes.height} duplicate (clip_id, window_idx) pairs")

    missing = [p for p in df["frames_path"].unique() if not Path(p).exists()]
    if missing:
        problems.append(f"{len(missing)} frame stores missing, e.g. {missing[0]}")

    return problems


def assert_disjoint_subjects(df: pl.DataFrame) -> list[str]:
    """Windows from one recording are near-duplicates and share a broadcast BP
    label, so a subject appearing in two splits leaks the test answers."""
    if "split" not in df.columns:
        return ["no split column to check"]
    offenders = (
        df.group_by("source", "subject_id")
        .agg(pl.col("split").n_unique().alias("n_splits"))
        .filter(pl.col("n_splits") > 1)
    )
    if offenders.height:
        sample = offenders.head(5).select("source", "subject_id").rows()
        return [f"{offenders.height} subjects span multiple splits, e.g. {sample}"]
    return []
