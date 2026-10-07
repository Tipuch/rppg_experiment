"""The RR training table run AH was trained on: all sources pooled, one seeded
subject shuffle, 90/5/5 train/dev/test, shifted windows and balance copies in train
only (user spec, 2026-10-07).

    .venv/bin/python -m src.vitals.rr_table      # settings are the constants below

    contact  BIDMC, DaLiA, CapnoBase records (`contact.READERS`)
      -> a 30 s segment every `STRIDE_S` (5 s)                     `shifted`
      -> its label from the breaths inside it                      `span_labels`
    camera   MCD: CFMamba's cached 10 s windows (`PPG_DIR`), turned upright
      -> a 30 s segment at every cached window start, so every 10 s
      -> the subject's rest RR reading                             `mcd_clips`
    -> `wave` per segment, each 10 s window z-scored (`waves.segment_waves`)
    -> a segment at a multiple of 30 s is "natural", any other "shifted"
    -> one seeded shuffle of all subjects, 90/5/5                  `split_subjects`
    -> shifted rows kept in train only, and only with a label
    -> `skewness`, and `tach` and `ls` (`waves.pp_table`)
    -> train labels oversampled                                    `balance`
    -> `OUT_DIR`/  features.parquet, waves_pp.parquet, waves.parquet

**Why one shuffle over all sources.** Each source alone crowds its labels into a
narrow band (MCD 15-19 /min); pooling and shuffling subjects puts every source in
train, dev and test. Subjects are split whole, so no person is in two splits.
Subject ids are prefixed with the source, so ids that two datasets share stay apart.

**Why shifted windows.** A 30 s segment every 5 s gives about 6x the distinct train
waves of the 30 s grid. Neighbours share up to 25 s, so they are correlated, but
each subject sits in one split, so nothing leaks. Dev and test keep the 30 s grid
only, so they score the natural rows and nothing twice. MCD's camera windows come
from a cache cut every 10 s, so its shifted rows start every 10 s.

**Labels.** Contact: `contact.annotated_rate` of the breaths in the span (`span_labels`);
a CapnoBase span that overlaps a marked CO2 artifact gets none. MCD: one rest
reading per subject, `respiratory` in db.csv at step "before", shared by the 3
cameras and all segments. After exercise the reading came before the video, so
those clips have no label (`REST_STEP`).

**`balance`.** The train labels crowd 12-20 /min, and a model scores well by staying
near them. Train rows are binned by label (`BIN_WIDTH`) and each bin is topped up by
repeating its rows (random oversampling), to the size of the largest bin or to
`MAX_REPEAT` copies of each row, whichever is smaller. No row is removed. Repeats
cycle through a bin's rows in one random order over all sources, so each row is
used about equally and no dataset is favoured. `repeat` numbers each key's copies
from 0. The table also keeps the natural train mean (`NATURAL_MEAN`): the mean of
the train rows with `augment` "natural" and a label, the baseline the network's
scores are read against; the oversampled mean is far from where the data sit.

**Row order matters.** `balance` shuffles with a seed, so the same rows in another
order give other copies. The order here is the one the AH table was built in:
BIDMC, DaLiA, CapnoBase in reader order, then MCD.

**Inputs not built here.** The MCD windows are CFMamba's forward pass, cached in
`PPG_DIR` (part-*.parquet: `clip_id`, `window_start_s`, `ppg_pred`) by the camera
pipeline. The camera PPG is stored inverted, as MCD's reference is, so it is
multiplied by `CAMERA_POLARITY`. `seen_by_ppg_model` marks clips in the camera
model's train split (`MANIFEST`): their PPG is cleaner than an unseen face's.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

from ..paths import BUILD_ROOT, DATA_ROOT
from . import contact, waves
from .waves import KEYS

TARGET = "rr_bpm"
NATURAL_MEAN = f"{TARGET}_natural_mean"
OUT_DIR = BUILD_ROOT / "vitals" / "rr_shuffled" / TARGET
PPG_DIR = BUILD_ROOT / "vitals" / "ppg"
MANIFEST = BUILD_ROOT / "clips_all.parquet"
MCD_DB = DATA_ROOT / "mcd_rppg" / "db.csv"
REST_STEP = "before"
CAMERA_POLARITY = -1

SEGMENT_S = 30.0
STRIDE_S = 5.0
PER_SEGMENT = round(SEGMENT_S / contact.WINDOW_S)
OFFSETS = [k * contact.WINDOW_S for k in range(PER_SEGMENT)]
FRACTIONS = {"dev": 0.05, "test": 0.05}
SPLIT_SEED = 20261007
BIN_WIDTH = 2.0
MAX_REPEAT = 10
BALANCE_SEED = 20261003
COLUMNS = (
    "clip_id", "source", "subject_id", "split", "seen_by_ppg_model", "segment_start_s",
    TARGET, "augment", "repeat", "skewness", NATURAL_MEAN,
)  # fmt: skip
on_grid = (pl.col("segment_start_s") / SEGMENT_S).round() * SEGMENT_S == pl.col(
    "segment_start_s"
)


def skewness(wave: np.ndarray) -> np.ndarray:
    """Skewness of each row of `wave` (N, T); NaN for a flat or non-finite row.
    The one table column taken straight from the wave, so `rr_mamba.check_waves`
    recomputes it to catch stale waves."""
    x = np.atleast_2d(np.asarray(wave, np.float64))
    finite = np.isfinite(x).all(axis=1)
    x = np.where(finite[:, None], x, 0.0)
    with np.errstate(all="ignore"):
        return np.where(
            ~finite | (np.ptp(x, axis=1) == 0), np.nan, stats.skew(x, axis=1)
        )


def _segments(windows: pl.DataFrame) -> pl.DataFrame:
    """`windows` (`clip_id`, `window_start_s`, a 10 s window per row) tagged
    `segment_start_s`: a segment starts at every window and holds it and the next
    two, kept only when all 3 are there. 3 rows per segment; segments overlap."""
    return (
        windows.select("clip_id", segment_start_s="window_start_s")
        .with_columns(window_start_s=pl.lit(OFFSETS)).explode("window_start_s", empty_as_null=True)
        .with_columns(pl.col("window_start_s") + pl.col("segment_start_s"))
        .join(windows, on=["clip_id", "window_start_s"], how="inner")
        .filter(pl.len().over(KEYS) == PER_SEGMENT)
        .sort(*KEYS, "window_start_s")
    )  # fmt: skip


def shifted(record: dict, stride_s: float = STRIDE_S) -> pl.DataFrame:
    """The 10 s windows of every 30 s segment of `record` (`contact.RECORD_KEYS`) that
    starts at a multiple of `stride_s` and is filled, tagged `segment_start_s`.

    The windows are `contact.windows`' own, run on the recording from each offset
    that is a multiple of `stride_s` below 10 s; offset 0 gives the 30 s grid's.
    """
    fs, parts = record["fs"], []
    for k in range(round(contact.WINDOW_S / stride_s)):
        start = round(k * stride_s * fs)
        keep = None if record["keep"] is None else record["keep"][start:]
        parts.append(contact.windows(record["clip_id"], record["ppg"][start:], fs, keep)
                     .with_columns(pl.col("window_start_s") + k * stride_s))  # fmt: skip
    return _segments(pl.concat(parts))


def span_labels(record: dict, starts: np.ndarray) -> np.ndarray:
    """The label of each span [t, t + 30 s) of `record`: `contact.annotated_rate` of
    the breath times inside it; NaN where the rule gives none.

    One call labels every span: span i's breaths are laid at 60 i + (b - t), so
    `breath_rate`'s 30 s grid puts each span in its own segment, 60 i, and leaves the
    odd segments empty. A breath is kept 1 us under the slot's 30 s, so rounding
    never moves it into the next segment.
    """
    starts, slot, packed = np.asarray(starts, np.float64), 2 * SEGMENT_S, []
    for b in record["breaths"]:
        b = np.sort(np.asarray(b, np.float64))
        lo = np.searchsorted(b, starts)
        n = np.searchsorted(b, starts + SEGMENT_S) - lo
        span = np.repeat(np.arange(starts.size), n)
        index = np.arange(span.size) - np.repeat(np.cumsum(n) - n, n) + lo[span]
        packed.append(
            span * slot + np.minimum(b[index] - starts[span], SEGMENT_S - 1e-6)
        )
    rates = contact.annotated_rate(record["clip_id"], packed, SEGMENT_S).drop_nulls(
        TARGET
    )
    out = np.full(starts.size, np.nan)
    out[(rates["segment_start_s"].to_numpy() / slot).round().astype(int)] = rates[
        TARGET
    ]
    return out


def contact_rows(record: dict) -> pl.DataFrame:
    """Every shifted segment of `record`: KEYS, `wave`, ids and label. A span that
    overlaps a CO2 artifact (CapnoBase) gets no label."""
    rows = waves.segment_waves(shifted(record), "ppg_true")
    starts = rows["segment_start_s"].to_numpy()
    label = span_labels(record, starts)
    for a, b in record.get("co2_artifacts", ()):
        label[(starts <= b) & (starts + SEGMENT_S > a)] = np.nan
    return rows.with_columns(
        source=pl.lit(record["source"]), subject_id=pl.lit(record["subject_id"]),
        rr_bpm=pl.Series(label).fill_nan(None), seen_by_ppg_model=pl.lit(False),
    )  # fmt: skip


def mcd_clips(manifest: Path = MANIFEST, db: Path = MCD_DB) -> pl.DataFrame:
    """MCD clips with a rest RR reading: `clip_id`, `source`, `subject_id`,
    `seen_by_ppg_model` (in the camera model's train split) and `rr_bpm`."""
    labels = pl.read_csv(db).filter(pl.col("step") == REST_STEP).select(
        clip_id="mcd/" + pl.col("video").str.split("/").list.last().str.replace(r"\.\w+$", ""),
        rr_bpm=pl.col("respiratory").cast(pl.Float64),
    )  # fmt: skip
    return (
        pl.read_parquet(manifest).filter(pl.col("source") == "mcd")
        .select("clip_id", "source", "subject_id", seen_by_ppg_model=pl.col("split") == "train")
        .join(labels, on="clip_id", how="inner").drop_nulls(TARGET)
    )  # fmt: skip


def camera_rows(clips: pl.DataFrame, ppg_dir: Path = PPG_DIR) -> pl.DataFrame:
    """Every segment of `clips` (`mcd_clips`) from the cached forward pass: KEYS,
    `wave`, and `clips`' columns."""
    parts = sorted(ppg_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(
            f"no cached CFMamba windows (part-*.parquet) in {ppg_dir}"
        )
    found = (
        pl.scan_parquet(parts).select("clip_id", "window_start_s", "ppg_pred")
        .filter(pl.col("clip_id").is_in(clips["clip_id"].implode())).collect()
        .with_columns(pl.col("ppg_pred") * CAMERA_POLARITY)
    )  # fmt: skip
    if found.select("clip_id", "window_start_s").is_duplicated().any():
        raise ValueError(f"{ppg_dir} holds a window twice")
    tagged = waves.segment_waves(_segments(found), "ppg_pred")
    return tagged.join(clips, on="clip_id", how="inner", maintain_order="left")


def split_subjects(rows: pl.DataFrame, seed: int = SPLIT_SEED) -> pl.DataFrame:
    """`rows` with `split`: the sorted unique `subject_id`s shuffled once with `seed`,
    the first `FRACTIONS["dev"]` of them dev, the next `FRACTIONS["test"]` test, the
    rest train. Whole subjects, so no person is in two splits."""
    subjects = rows.select("subject_id").unique().sort("subject_id")
    n_dev, n_test = (round(subjects.height * FRACTIONS[s]) for s in ("dev", "test"))
    draw = pl.int_range(pl.len()).shuffle(seed)
    return rows.join(
        subjects.with_columns(
            split=pl.when(draw < n_dev).then(pl.lit("dev"))
            .when(draw < n_dev + n_test).then(pl.lit("test")).otherwise(pl.lit("train"))
        ),
        on="subject_id", maintain_order="left",
    )  # fmt: skip


def balance(table: pl.DataFrame, seed: int = BALANCE_SEED) -> pl.DataFrame:
    """`table` with the rare label bins of the labelled train rows oversampled (module
    docstring); all rows kept, the train rows first."""
    is_train = (pl.col("split") == "train") & pl.col(TARGET).is_not_null()
    size = pl.len().over("bin")
    # At least `size`, so the first copy of each row is always kept.
    want = pl.min_horizontal(size.max(), size * MAX_REPEAT)
    train = (
        table.filter(is_train)
        .sample(fraction=1.0, seed=seed, shuffle=True)
        .with_columns(bin=(pl.col(TARGET) / BIN_WIDTH).floor() * BIN_WIDTH,
                      order=pl.int_range(pl.len()))
        .with_columns(want=want, copies=(want / size).ceil().cast(pl.UInt32))
        # Copy k of each row comes after copy k-1 of all of them, so the first
        # `want` rows of a bin use each row about equally.
        .with_columns(copy=pl.int_ranges(0, "copies")).explode("copy", empty_as_null=True)
        .sort("bin", "copy", "order")
        .filter(pl.int_range(pl.len()).over("bin") < pl.col("want"))
        .sample(fraction=1.0, seed=seed + 1, shuffle=True)
        .drop("bin", "order", "want", "copies", "copy")
    )  # fmt: skip
    return pl.concat([train, table.filter(~is_train)])


def build(
    segments: pl.DataFrame, seed: int = SPLIT_SEED
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """(features, waves_pp) from every segment of every source (`contact_rows` and
    `camera_rows`, concatenated in that order): split, shifted rows outside train
    dropped, `skewness`, `tach` and `ls`, `balance`."""
    rows = split_subjects(
        segments.select(*KEYS, "source", "subject_id", "seen_by_ppg_model", TARGET, "wave")
        .with_columns(subject_id=pl.col("source") + ":" + pl.col("subject_id"),
                      augment=pl.when(on_grid).then(pl.lit("natural")).otherwise(pl.lit("shifted"))),
        seed,
    ).filter((pl.col("augment") == "natural")
             | ((pl.col("split") == "train") & pl.col(TARGET).is_not_null()))  # fmt: skip
    if rows.select(KEYS).is_duplicated().any():
        raise ValueError("a segment key appears twice")
    rows = rows.with_columns(
        skewness=pl.Series(skewness(rows["wave"].to_numpy())).fill_nan(None)
    )
    natural_mean = rows.filter(pl.col("split") == "train", pl.col("augment") == "natural",
                               pl.col(TARGET).is_not_null())[TARGET].mean()  # fmt: skip
    table = (
        balance(rows.drop("wave"))
        .with_columns(pl.lit(natural_mean, pl.Float64).alias(NATURAL_MEAN),
                      repeat=pl.int_range(pl.len()).over(KEYS).cast(pl.UInt32))
        .select(COLUMNS)
    )  # fmt: skip
    return table, waves.pp_table(rows.select(*KEYS, "wave"))


def main(out_dir: Path = OUT_DIR) -> pl.DataFrame:
    """Build the table from the raw contact recordings and the camera cache, write
    features.parquet, waves_pp.parquet and waves.parquet to `out_dir`, and print the
    rows per split and source. An existing output file is refused, never
    overwritten. Returns the table."""
    start = time.perf_counter()
    paths = [
        out_dir / name for name in ("features.parquet", waves.PP_FILE, waves.WAVE_FILE)
    ]
    if found := [str(p) for p in paths if p.exists()]:
        raise FileExistsError(f"refusing to overwrite {', '.join(found)}")
    records = (record for reader in contact.READERS for record in reader())
    segments = pl.concat(
        [*map(contact_rows, records), camera_rows(mcd_clips())], how="diagonal"
    )
    table, pp = build(segments)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(paths[0])
    pp.write_parquet(paths[1])
    pp.select(*KEYS, "wave").write_parquet(paths[2])
    with pl.Config(tbl_rows=-1):
        print(table.group_by("split", "source", "augment").agg(
            subjects=pl.col("subject_id").n_unique(), keys=(pl.col("repeat") == 0).sum(),
            rows=pl.len(), labelled=pl.col(TARGET).count(),
        ).sort("split", "source", "augment"))  # fmt: skip
    print(
        f"wrote {table.height} rows, {pp.height} waves -> {out_dir}; natural train mean "
        f"{table[NATURAL_MEAN][0]:.3f}; {time.perf_counter() - start:.0f} s"
    )
    return table


if __name__ == "__main__":
    main()
