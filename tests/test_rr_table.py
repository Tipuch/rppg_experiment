"""The RR table builder (src/vitals/rr_table.py): shifted windows and their labels,
the 90/5/5 subject split, and train-only balancing. All on synthetic records, so no
dataset is needed."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from src.vitals import contact, rr_table, waves

FS = 125.0


def _record(seconds: float = 100.0, breaths=None, keep=None) -> dict:
    t = np.arange(round(seconds * FS)) / FS
    return {
        "clip_id": "rec", "source": "bidmc", "subject_id": "p1", "fs": FS, "keep": keep,
        "ppg": np.sin(2 * np.pi * 1.2 * t), "breaths": breaths or [np.arange(0, seconds, 4.0)],
    }  # fmt: skip


def test_shifted_cuts_a_full_segment_every_5_s_from_the_readers_own_windows():
    record = _record()
    tagged = rr_table.shifted(record)
    starts = tagged["segment_start_s"].unique().sort().to_numpy()
    # 100 s of PPG: the last full 30 s segment starts at 70 s.
    assert np.array_equal(starts, np.arange(0.0, 75.0, 5.0))
    assert (tagged.group_by(waves.KEYS).len()["len"] == 3).all()
    at_5 = tagged.filter(pl.col("segment_start_s") == 5.0).sort("window_start_s")
    assert at_5["window_start_s"].to_list() == [5.0, 15.0, 25.0]
    own = contact.windows("rec", record["ppg"][round(5 * FS) :], FS)["ppg_true"][
        0
    ].to_numpy()
    np.testing.assert_array_equal(at_5["ppg_true"][0].to_numpy(), own)


def test_shifted_drops_every_segment_that_touches_an_unusable_sample():
    keep = np.ones(round(100 * FS), dtype=bool)
    keep[round(52 * FS)] = False
    starts = rr_table.shifted(_record(keep=keep))["segment_start_s"].unique().to_numpy()
    assert (
        not any(s <= 52 < s + 30 for s in starts) and 0.0 in starts and 55.0 in starts
    )


def test_span_labels_take_the_breaths_inside_each_span():
    # 15 /min for the first minute, 20 /min after; a second annotator agrees.
    breaths = np.r_[np.arange(0, 60, 4.0), np.arange(60, 120, 3.0)]
    starts = np.array([0.0, 60.0, 45.0, 115.0])
    labels = rr_table.span_labels(_record(120.0, [breaths, breaths]), starts)
    inside = lambda t: breaths[(breaths >= t) & (breaths < t + 30)]
    expected = [60 * (len(b) - 1) / (b[-1] - b[0]) for b in map(inside, starts[:3])]
    np.testing.assert_allclose(labels[:3], expected)
    np.testing.assert_allclose(labels[:2], [15.0, 20.0])
    assert np.isnan(labels[3])  # 2 breaths in [115, 145): below MIN_BREATHS


def test_a_span_overlapping_a_co2_artifact_has_no_label():
    record = _record() | {"co2_artifacts": np.array([[40.0, 41.0]])}
    rows = rr_table.contact_rows(record)
    void = rows.filter(pl.col("rr_bpm").is_null())["segment_start_s"].sort().to_list()
    assert void == [15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
    assert rows.columns == [
        *waves.KEYS,
        "wave",
        "source",
        "subject_id",
        "rr_bpm",
        "seen_by_ppg_model",
    ]


def _segments(subjects: int = 60, seed: int = 0) -> pl.DataFrame:
    """Every subject: 4 natural segments (one unlabelled) and 6 shifted (one
    unlabelled), in 2 sources, random waves."""
    rng = np.random.default_rng(seed)
    starts = [0.0, 30.0, 60.0, 90.0, 5.0, 10.0, 15.0, 35.0, 40.0, 45.0]
    rows = [{"clip_id": f"c{s}", "segment_start_s": t, "source": "ab"[s % 2],
             "subject_id": str(s), "seen_by_ppg_model": False,
             "rr_bpm": None if t in (90.0, 45.0) else float(rng.uniform(6, 40))}
            for s in range(subjects) for t in starts]  # fmt: skip
    return pl.DataFrame(rows).with_columns(
        wave=pl.Series(rng.normal(size=(len(rows), 900)).astype(np.float32))
    )


def test_split_subjects_is_90_5_5_by_whole_subject_and_seed_stable():
    rows = _segments(200).drop("wave")
    split = rr_table.split_subjects(rows)
    per = split.group_by("subject_id").agg(
        pl.col("split").n_unique().alias("n"), pl.col("split").first()
    )
    assert (per["n"] == 1).all()
    assert dict(per.group_by("split").len().iter_rows()) == {
        "train": 180,
        "dev": 10,
        "test": 10,
    }
    assert split.equals(rr_table.split_subjects(rows))
    assert not split.equals(rr_table.split_subjects(rows, seed=1))
    assert split.select(rows.columns).equals(rows)  # same rows, same order


def test_build_keeps_shifts_and_copies_in_train_only():
    table, pp = rr_table.build(_segments())
    assert table.columns == list(rr_table.COLUMNS)
    outside = table.filter(pl.col("split") != "train")
    assert (
        outside.height
        and (outside["augment"] == "natural").all()
        and (outside["repeat"] == 0).all()
    )
    assert outside.select(waves.KEYS).is_unique().all()
    train = table.filter(pl.col("split") == "train")
    assert (train["repeat"] > 0).any() and (train["augment"] == "shifted").any()
    assert train.filter(pl.col("augment") == "shifted")["rr_bpm"].is_not_null().all()
    natural = train.filter(pl.col("augment") == "natural", pl.col("repeat") == 0)
    assert table[rr_table.NATURAL_MEAN].unique().to_list() == [
        pytest.approx(natural["rr_bpm"].mean())
    ]
    assert (
        pp.select(waves.KEYS)
        .sort(waves.KEYS)
        .equals(table.select(waves.KEYS).unique().sort(waves.KEYS))
    )
    joined = table.filter(pl.col("repeat") == 0).join(pp, on=waves.KEYS)
    np.testing.assert_allclose(
        rr_table.skewness(joined["wave"].to_numpy()), joined["skewness"], atol=1e-6
    )


def test_balance_tops_up_rare_train_bins_and_leaves_other_rows_alone():
    rng = np.random.default_rng(1)
    labels = [*rng.uniform(16, 18, 60), *rng.uniform(30, 32, 3), *[None] * 4]
    table = pl.DataFrame({"clip_id": [f"k{i}" for i in range(67)], "rr_bpm": labels,
                          "split": ["train"] * 63 + ["train", "dev", "test", "dev"]})  # fmt: skip
    table = pl.concat(
        [
            table,
            pl.DataFrame(
                {
                    "clip_id": ["d", "t"],
                    "rr_bpm": [31.0, 31.0],
                    "split": ["dev", "test"],
                }
            ),
        ]
    )
    out = rr_table.balance(table)
    is_train = (pl.col("split") == "train") & pl.col("rr_bpm").is_not_null()
    assert (
        out.filter(~is_train)
        .sort("clip_id")
        .equals(table.filter(~is_train).sort("clip_id"))
    )
    copies = out.filter(is_train).group_by("clip_id").len()
    assert copies.height == 63 and copies["len"].max() == rr_table.MAX_REPEAT
    # The 3 rows at 30-32 /min, 10 copies each; the crowded bin is not copied.
    rare = out.filter(is_train, pl.col("rr_bpm") >= 30)
    assert rare.height == 3 * rr_table.MAX_REPEAT
