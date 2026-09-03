"""Comparing heart-rate readouts on one set of predicted waveforms.

Three readouts exist in this project and they disagree. `postprocess.heart_rate`
is the vendored toolbox's bare argmax over a rectangular periodogram, and it is
what every number in README.md comes through. `postprocess.spectral_hr` is the
same measurement with the steps production PPG systems add: a window function,
heavier zero-padding, and interpolation of the peak between bins. The interval
readout counts beats and takes the median gap, which is what pulse oximeters and
wrist wearables actually display.

On three clips inspected by hand they disagreed by up to 15 bpm, in both
directions, and the one clip with contact PPG was not won by any of them. Three
clips cannot settle it, so this scores every variant over a labelled split at
once, from a single forward pass.

The forward pass is deliberately separate from the scoring: the model runs once
and its output is cached, so a variant can be added and the sweep re-run without
a card.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from .postprocess import bandpass, heart_rate, interval_hr, spectral_hr

# Each variant reads one band-passed window and returns bpm, or NaN. The control
# is first: a change is only worth making if it beats what is already reported.
VARIANTS: dict[str, Callable[[np.ndarray, float], float]] = {
    "toolbox": lambda w, fps: heart_rate(w, fps, filtered=True),
    "boxcar_p1": lambda w, fps: spectral_hr(
        w, fps, filtered=True, pad=1, window="boxcar"
    ),
    "boxcar_p8": lambda w, fps: spectral_hr(
        w, fps, filtered=True, pad=8, window="boxcar"
    ),
    "hann_p8": lambda w, fps: spectral_hr(w, fps, filtered=True, pad=8, window="hann"),
    "hann_p8_nointerp": lambda w, fps: spectral_hr(
        w, fps, filtered=True, pad=8, window="hann", interpolate=False
    ),
    "boxcar_p8_nointerp": lambda w, fps: spectral_hr(
        w, fps, filtered=True, pad=8, window="boxcar", interpolate=False
    ),
    "interval": lambda w, fps: interval_hr(w, fps, filtered=True),
}

SOURCE_ALL = "all"


def rates(waves: np.ndarray, fps: float, variant: str) -> np.ndarray:
    """bpm per window, shape (n_windows,), for one variant.

    Band-passing happens once here rather than inside each variant, so every
    variant is scored on identical input and the comparison is of the readout
    alone.
    """
    read = VARIANTS[variant]
    return np.array(
        [read(bandpass(np.asarray(w, dtype=np.float64), fps), fps) for w in waves],
        dtype=np.float64,
    )


def score(
    predicted: np.ndarray, truth: np.ndarray, sources: list[str], fps: float,
    variants: list[str] | None = None,
) -> pl.DataFrame:
    """MAE, RMSE and rho per variant per source, plus the aggregate.

    The truth rate is read with the **same** variant as the prediction. Reading it
    with one readout and the prediction with another would measure the difference
    between the two methods and report it as model error.

    Windows whose rate could not be read are dropped and counted, for the reason
    `evaluate.summarise` gives: a moving denominator lets a variant that gives up
    on the hard windows report the easy ones' score.
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
