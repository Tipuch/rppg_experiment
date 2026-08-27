"""Spectral audit: does each clip actually contain a recoverable pulse?

A model can only learn what is present in the data. Lossy inter-frame compression
removes rPPG almost completely -- the pulse is a ~0.1% spatially-smooth brightness
change, which is exactly what a motion-compensated codec discards as imperceptible.
Nothing downstream recovers it, so this has to be measured before training rather
than inferred afterwards from a disappointing score.

For each clip: take the mean skin luma trace, look for a cardiac peak, and compare
it against the labelled heart rate.

    uv run python -m src.cli audit
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import signal as sps
from torch.utils.data import DataLoader

from .dataset import TARGET_FPS, WindowDataset

HR_BAND = (0.7, 3.5)                 # 42-210 bpm
MATCH_TOLERANCE_BPM = 10.0
# A peak must stand this far above the band's median power to count as a peak
# rather than the top of a monotonic slope.
MIN_PROMINENCE = 3.0
BAND_EDGE_BPM = HR_BAND[0] * 60      # 42 bpm: what a peakless spectrum returns


def analyse_trace(mean_y: np.ndarray, label_hr: float) -> dict:
    x = np.asarray(mean_y, dtype=np.float64)
    x = x - x.mean()
    if x.size < 64 or not np.isfinite(x).all() or np.ptp(x) == 0:
        return {"peak_bpm": None, "prominence": 0.0, "lf_ratio": 0.0,
                "abs_err": None, "has_pulse": False, "band_edge": True}

    freqs, power = sps.welch(x, fs=TARGET_FPS, nperseg=min(256, x.size))
    band = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
    if not band.any() or power[band].max() <= 0:
        return {"peak_bpm": None, "prominence": 0.0, "lf_ratio": 0.0,
                "abs_err": None, "has_pulse": False, "band_edge": True}

    band_power = power[band]
    peak_bpm = float(freqs[band][int(np.argmax(band_power))] * 60.0)
    prominence = float(band_power.max() / np.median(band_power))
    lf_ratio = float(band_power[0] / band_power[-1])
    abs_err = abs(peak_bpm - label_hr)

    # Pinned to the bottom of the search band means there was no peak at all, just
    # a monotonically falling 1/f slope.
    band_edge = peak_bpm <= BAND_EDGE_BPM + 1.0
    return {
        "peak_bpm": peak_bpm,
        "prominence": prominence,
        "lf_ratio": lf_ratio,
        "abs_err": abs_err,
        "band_edge": band_edge,
        "has_pulse": bool(
            not band_edge
            and abs_err <= MATCH_TOLERANCE_BPM
            and prominence >= MIN_PROMINENCE
        ),
    }


def run(
    manifest: pl.DataFrame, n_frames: int = 300, workers: int = 8
) -> pl.DataFrame:
    # Heart rate only. The audit compares a spectral peak against the labelled HR
    # and never touches blood pressure, and a BP-less source such as UBFC-rPPG has
    # nulls in those columns that would fail the tensor conversion.
    dataset = WindowDataset(
        manifest, n_frames=n_frames, train=False, targets=("hr_bpm",),
        compute_mean_y=True, return_waveform=False,
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=workers,
        prefetch_factor=2 if workers else None, persistent_workers=False,
    )
    rows: list[dict] = []
    for i, batch in enumerate(loader):
        stats = analyse_trace(
            batch["mean_y"][0].numpy(), float(batch["targets"][0][0])
        )
        rows.append({
            "clip_id": batch["clip_id"][0],
            "subject_id": batch["subject_id"][0],
            "label_hr": float(batch["targets"][0][0]),
            **stats,
        })
        if (i + 1) % 200 == 0:
            usable = sum(r["has_pulse"] for r in rows)
            print(f"  {i + 1}/{len(dataset)}  usable so far {usable} "
                  f"({100 * usable / len(rows):.1f}%)", flush=True)
    return pl.DataFrame(rows)


def summarise(results: pl.DataFrame, manifest: pl.DataFrame) -> None:
    joined = results.join(
        manifest.select("clip_id", "source", "fps"), on="clip_id", how="left"
    )
    print("\nper source:")
    print(
        joined.group_by("source").agg(
            pl.len().alias("clips"),
            pl.col("has_pulse").sum().alias("with_pulse"),
            (pl.col("has_pulse").mean() * 100).round(1).alias("pct"),
            pl.col("band_edge").mean().mul(100).round(1).alias("peakless_pct"),
            pl.col("prominence").median().round(2).alias("med_prominence"),
            pl.col("lf_ratio").median().round(1).alias("med_lf_ratio"),
            pl.col("abs_err").median().round(1).alias("med_hr_err"),
        ).sort("source")
    )
    total = joined.height
    usable = int(joined["has_pulse"].sum())
    print(f"\nusable clips: {usable} / {total}  ({100 * usable / max(total, 1):.2f}%)")
    if usable:
        subj = joined.filter(pl.col("has_pulse"))["subject_id"].n_unique()
        print(f"usable subjects: {subj}")
