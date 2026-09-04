"""Train/dev/test bucketing over subjects.

`RATIOS` (85/10/5) is the default; the pooled manifest is built at
`RATIOS_POOLED` (90/3/7). See `src/aggregation/combine.py`.

Bucketing is over *subjects*, not rows:

  - Windows cut from one recording are near-duplicates of each other. Splitting
    at row level puts near-copies on both sides.
  - SBP and DBP are one clinical reading broadcast across every window of a
    recording. A row-level shuffle leaks the test labels into training.

Subjects are shuffled deterministically (seeded hash, no dependence on row order
or machine) and then filled greedily by row count, so the split tracks the target
ratios in rows while keeping each subject on one side.
"""

from __future__ import annotations

import hashlib

import polars as pl

RATIOS = {"train": 0.85, "dev": 0.10, "test": 0.05}
# Used by `combine` for the pooled manifest and by `mrnirp` for its own. 3% of a
# 15-subject corpus is under one subject, so `summarise` reports what was achieved;
# that is the number to quote, not this target.
RATIOS_POOLED = {"train": 0.90, "dev": 0.03, "test": 0.07}
SPLIT_NAMES = ["train", "dev", "test"]
DEFAULT_SEED = 20260822
# Window length the segment count is defined at. `src.model.dataset` imports its
# own DEFAULT_CLIP_FRAMES from here so the two cannot disagree.
DEFAULT_FRAMES = 300


def _order_key(source: str, subject_id: str, seed: int) -> str:
    """Deterministic shuffle key. Stable across runs, machines and row order."""
    return hashlib.sha256(f"{seed}:{source}/{subject_id}".encode()).hexdigest()


def segment_counts(
    n_frames: int = DEFAULT_FRAMES,
    fps: float = 30.0,
    stride_frames: int | None = None,
) -> pl.Expr:
    """Windows each clip yields, as an expression over `duration_s`.

    This is the definition `expand_to_segments` enumerates and the weight the
    splitter balances by, so the two cannot drift apart.

    `stride_frames` defaults to non-overlapping. A caller that subsamples a large
    corpus with a stride gets fewer windows, and a count that ignored the stride
    would weight the split by segments that are never enumerated.

    The epsilon: a clip whose duration is an exact multiple of the span lands on a
    floating-point boundary where the division evaluates to 28.999999, and the
    floor then drops a segment.
    """
    span = n_frames / fps
    stride = (stride_frames if stride_frames is not None else n_frames) / fps
    return (
        pl.when(pl.col("duration_s") > span)
        .then(((pl.col("duration_s") - span) / stride + 1e-9).floor() + 1)
        .otherwise(1)
        .cast(pl.UInt32)
    )


def assign(
    df: pl.DataFrame,
    ratios: dict[str, float] = RATIOS,
    seed: int = DEFAULT_SEED,
    weight: str | None = None,
    order: str = "hash",
    stratify: str | None = None,
) -> pl.DataFrame:
    """Add a `split` column of train/dev/test, assigned per (source, subject).

    `weight` names a column to balance by instead of the row count. Callers that
    pass segments already have one row per example, so they need nothing; a
    caller holding one row per *recording* passes `segment_counts` so the ratios
    still land in examples rather than in recordings, which differ by 4x here.

    `order` decides which subject is placed next, and on a small corpus it decides
    the whole result:

      "hash"  shuffled order. Every split recorded in this repo was produced this
              way, so it stays the default; changing it moves the assignment and
              breaks comparability with the runs already recorded.
      "size"  largest subject first, into whichever split has the most rows still
              owing in absolute terms. Use it when a target share is smaller than
              one subject.

    `stratify` names a column to split *within*, so each stratum contributes to
    each side. Without it a greedy fill can put one stratum entirely on one side:
    MR-NIRP's first split gave dev 2 Car clips and test 3 Indoor ones, so the test
    score measured Indoor and dev measured Car. Use it only where strata do not
    share subjects, which is checked below.

    "size" changes two things together. The default rule picks the split furthest
    below target *as a fraction of its own target*, which is scale-free, so an
    empty dev bin reads as equally starved whether it wants 3% or 90% and the
    first subjects placed go to the smallest bins -- the bins a single subject can
    overshoot. Absolute capacity sends the large subjects to train, the only bin
    with room for them, and leaves the small ones for dev and test. Measured on
    MR-NIRP's 15 subjects at 90/3/7: shuffled + relative gives 64/14/21,
    largest-first + relative gives 67/17/16, largest-first + absolute gives the
    ratios DATASETS.md records.
    """
    if df.height == 0:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias("split"))

    if stratify is not None:
        parts = df.partition_by(stratify, maintain_order=True)
        # A subject spanning two strata would be assigned twice, independently,
        # and could land on both sides of the split. Refuse rather than leak.
        seen: dict[str, str] = {}
        for part in parts:
            stratum = str(part[stratify][0])
            for subject in part["subject_id"].unique().to_list():
                if seen.setdefault(subject, stratum) != stratum:
                    raise ValueError(
                        f"subject {subject!r} appears in more than one {stratify}; "
                        "stratifying would assign it twice"
                    )
        return pl.concat(
            [assign(part, ratios, seed, weight, order) for part in parts],
            how="vertical",
        )

    groups = (
        df.group_by("source", "subject_id")
        .agg(pl.len().alias("rows") if weight is None
             else pl.col(weight).sum().cast(pl.Int64).alias("rows"))
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
    if order == "size":
        # Descending, with the hash as tie-break so equal-sized subjects still
        # order deterministically rather than by whatever group_by returned.
        groups = groups.sort(["rows", "_key"], descending=[True, False])
    elif order != "hash":
        raise ValueError(f"unknown order {order!r}, expected 'hash' or 'size'")

    total = int(groups["rows"].sum())
    targets = {name: ratios[name] * total for name in SPLIT_NAMES}

    assignments: dict[tuple[str, str], str] = {}
    filled = {name: 0 for name in SPLIT_NAMES}
    for source, subject_id, n_rows, _ in groups.iter_rows():
        name = max(
            SPLIT_NAMES,
            key=(
                (lambda s: targets[s] - filled[s]) if order == "size"
                else (lambda s: (targets[s] - filled[s]) / max(targets[s], 1e-9))
            ),
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
