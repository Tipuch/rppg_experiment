"""Comparing heart-rate readouts against each other on the same waveforms.

The readout is not settled: `heart_rate` is the vendored toolbox's bare argmax,
`spectral_hr` interpolates the peak, and the interval readout counts beats. On the
three clips inspected by hand they disagreed by up to 15 bpm and no single one won,
so the choice has to be made on the labelled split rather than by argument. This
covers the scoring half -- the forward pass needs a card and is not covered here.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl

from src.model.readout import VARIANTS, rates, score

FPS = 30.0


def _tone(bpm: float, n: int = 300, fps: float = FPS) -> np.ndarray:
    t = np.arange(n) / fps
    return np.sin(2 * math.pi * (bpm / 60.0) * t)


def _stack(bpms: list[float]) -> np.ndarray:
    return np.stack([_tone(b) for b in bpms])


def test_every_variant_reads_a_clean_tone_back() -> None:
    """A variant that cannot recover a synthetic tone is measuring its own
    window, and its MAE on real data would not mean anything."""
    waves = _stack([60.0, 90.0, 120.0])
    for name in VARIANTS:
        got = rates(waves, FPS, name)
        assert np.allclose(got, [60.0, 90.0, 120.0], atol=4.0), name


def test_a_perfect_prediction_scores_zero_error() -> None:
    truth = _stack([66.0, 78.0, 102.0])
    table = score(truth.copy(), truth, ["mcd"] * 3, FPS)
    for row in table.filter(pl.col("source") == "all").iter_rows(named=True):
        assert row["mae"] < 4.0, row["variant"]


def test_a_constant_offset_shows_up_as_that_offset() -> None:
    """The control the project already learned to demand: a predictor that is
    wrong by a fixed amount must not be able to hide behind an aggregate."""
    truth = _stack([70.0, 70.0, 70.0])
    predicted = _stack([100.0, 100.0, 100.0])
    table = score(predicted, truth, ["mcd"] * 3, FPS)
    for row in table.filter(pl.col("source") == "all").iter_rows(named=True):
        assert row["mae"] > 25.0, row["variant"]


def test_unreadable_windows_are_dropped_and_counted() -> None:
    """`evaluate.summarise` already refuses to average over a moving denominator.
    A sweep that silently dropped dead windows would rank a variant by how often
    it gave up."""
    truth = _stack([72.0, 72.0])
    predicted = np.stack([_tone(72.0), np.zeros(300)])
    table = score(predicted, truth, ["mcd"] * 2, FPS)
    row = table.filter(
        (pl.col("source") == "all") & (pl.col("variant") == "toolbox")
    ).to_dicts()[0]
    assert row["windows"] == 2
    assert row["dropped"] == 1


def test_results_are_broken_out_per_source() -> None:
    """MCD is ~98% of the test split, so an aggregate is a measurement of MCD."""
    truth = _stack([72.0, 72.0, 72.0])
    table = score(truth.copy(), truth, ["mcd", "mcd", "ubfc"], FPS)
    assert set(table["source"].unique()) == {"all", "mcd", "ubfc"}
