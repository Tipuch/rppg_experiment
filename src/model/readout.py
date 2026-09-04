"""Comparing heart-rate readouts on one set of predicted waveforms.

Two families of readout exist in this project, and they fail differently.

The spectral family reads the 0.75-2.5 Hz reporting band. `postprocess.heart_rate` is
the vendored toolbox's argmax over a rectangular periodogram; `postprocess.spectral_hr`
is the same measurement plus a window function, heavier zero-padding and interpolation
of the peak between bins. Both miss by tens of bpm when they lock onto a harmonic.

The interval family reads the 0.75-4 Hz detection band. `postprocess.interval_hr` finds
beats with MSPTD and aggregates the gaps, by median or by mean. Both are hurt by a
miscounted beat rather than by the spectrum.

`postprocess.reported_hr` is the middle of three of them, and is what this project
reports. It is scored here beside its own members, so the choice stays a measurement
rather than becoming a constant nobody re-checks.

On three clips inspected by hand the families disagreed by up to 15 bpm in both
directions. This scores every variant over a labelled split from a single forward pass.

The forward pass is separate from the scoring: the model runs once and its output is
cached, so a variant can be added and the sweep re-run without a GPU.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from .postprocess import (
    heart_rate,
    interval_hr,
    reported_hr,
    spectral_hr,
)

# Each variant reads one raw window and returns bpm, or NaN. Every one of them
# filters for itself, to whichever band it reads in. The vendored toolbox readout is
# first, as the control the others are compared against.
VARIANTS: dict[str, Callable[[np.ndarray, float], float]] = {
    "toolbox": lambda w, fps: heart_rate(w, fps),
    "boxcar_p1": lambda w, fps: spectral_hr(w, fps, pad=1, window="boxcar"),
    "boxcar_p8": lambda w, fps: spectral_hr(w, fps, pad=8, window="boxcar"),
    "hann_p8": lambda w, fps: spectral_hr(w, fps, pad=8, window="hann"),
    "hann_p8_nointerp": lambda w, fps: spectral_hr(
        w, fps, pad=8, window="hann", interpolate=False
    ),
    "boxcar_p8_nointerp": lambda w, fps: spectral_hr(
        w, fps, pad=8, window="boxcar", interpolate=False
    ),
    "interval": lambda w, fps: interval_hr(w, fps),
    # The same peaks aggregated the other way. The mean of N-1 gaps collapses to
    # (last - first) / (N - 1), so it reads the span and the count and nothing in
    # between: one spurious or missed peak moves it by a whole 1/(N-1) of the rate,
    # which is 6 bpm on a 10 s window at 72 bpm.
    "interval_mean": lambda w, fps: interval_hr(w, fps, aggregate=np.mean),
}


def _blend_mean(wave: np.ndarray, fps: float) -> float:
    """The three readouts averaged rather than voted on.

    Kept as the control `blend_median` is read against, and currently ahead of it. The
    argument for a vote is that the mean moves with every member, so a member that
    failed drags it, while the median is the middle of three and discards it. Under the
    previous beat detector the sweep agreed: median ahead by 0.11 bpm RMSE and 0.005
    rho. Detecting beats in 0.75-4 Hz reversed it -- the mean now leads by 0.15 RMSE and
    0.005 rho, and trails by 0.02 MAE.

    Which is why both stay in the table. The ordering is a property of the current
    configuration, not a settled result, and `reported_hr` is the median because that
    is what was chosen, not because this sweep still prefers it.
    """
    parts = np.array(
        [
            spectral_hr(wave, fps, pad=8, window="boxcar"),
            interval_hr(wave, fps),
            interval_hr(wave, fps, aggregate=np.mean),
        ],
        dtype=np.float64,
    )
    usable = parts[np.isfinite(parts)]
    return float(np.mean(usable)) if usable.size else float("nan")


VARIANTS["blend_mean"] = _blend_mean
# The reported readout. Scored here beside its own members so the choice stays
# checkable rather than becoming a constant nobody re-measures.
VARIANTS["blend_median"] = reported_hr

SOURCE_ALL = "all"


def rates(waves: np.ndarray, fps: float, variant: str) -> np.ndarray:
    """bpm per window, shape (n_windows,), for one variant.

    Each variant is handed the raw window and filters it itself. Band-passing once
    here would be simpler, and is what this did while every readout shared one band,
    but the interval readouts now detect beats in 0.75-4 Hz while the spectral ones
    read 0.75-2.5 Hz. Pre-filtering to either band would hand the other a signal it
    cannot use, and the sweep would be comparing a readout against a crippled version
    of its rival rather than against the rival.
    """
    read = VARIANTS[variant]
    return np.array(
        [read(np.asarray(w, dtype=np.float64), fps) for w in waves],
        dtype=np.float64,
    )


def score(
    predicted: np.ndarray, truth: np.ndarray, sources: list[str], fps: float,
    variants: list[str] | None = None,
) -> pl.DataFrame:
    """MAE, RMSE and rho per variant per source, plus the aggregate.

    The truth rate is read with the same variant as the prediction. Reading it with
    one readout and the prediction with another would measure the difference between
    the two methods and report it as model error.

    Windows whose rate could not be read are dropped and counted, for the reason
    `evaluate.summarise` gives: a moving denominator lets a variant that fails on the
    harder windows report the score of the rest.
    """
    if predicted.shape != truth.shape:
        raise ValueError(
            f"predicted {predicted.shape} and truth {truth.shape} must match: they "
            "are the same windows read two ways."
        )
    if len(sources) != len(predicted):
        raise ValueError(
            f"{len(sources)} sources for {len(predicted)} windows"
        )

    frames = [
        pl.DataFrame({
            "variant": variant,
            "source": sources,
            "hr_pred": rates(predicted, fps, variant),
            "hr_true": rates(truth, fps, variant),
        })
        for variant in (variants or list(VARIANTS))
    ]
    long = pl.concat(frames, how="vertical")
    return _aggregate(
        pl.concat([long, long.with_columns(pl.lit(SOURCE_ALL).alias("source"))])
    )


def _aggregate(long: pl.DataFrame) -> pl.DataFrame:
    """Group to one row per variant and source. Errors are in bpm."""
    usable = pl.col("hr_pred").is_finite() & pl.col("hr_true").is_finite()
    error = (pl.col("hr_pred") - pl.col("hr_true")).filter(usable)
    return (
        long.group_by("variant", "source")
        .agg(
            pl.len().alias("windows"),
            (~usable).sum().alias("dropped"),
            error.abs().mean().alias("mae"),
            (error.pow(2).mean().sqrt()).alias("rmse"),
            pl.corr(
                pl.col("hr_pred").filter(usable), pl.col("hr_true").filter(usable)
            ).alias("rho"),
        )
        .sort("source", "mae")
    )


def load_dump(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Predicted windows, target windows and per-window source, from a dump."""
    with np.load(path, allow_pickle=False) as data:
        return (
            data["predicted"].astype(np.float64),
            data["truth"].astype(np.float64),
            [str(s) for s in data["source"]],
        )


def save_dump(
    path: Path, predicted: np.ndarray, truth: np.ndarray, sources: list[str],
) -> None:
    """Cache one forward pass so variants can be added without a card."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        predicted=predicted.astype(np.float32),
        truth=truth.astype(np.float32),
        source=np.array(sources, dtype="<U16"),
    )
