"""Contact-PPG recordings for the RR table: BIDMC, PPG-DaLiA, CapnoBase.

Each reader yields one record per recording, a dict with `RECORD_KEYS`:

    clip_id, source, subject_id   ids; a subject may own several records
    ppg, fs                       the raw PPG and its rate
    keep                          one bool per PPG sample, or None: False marks a
                                  sample no window may contain
    breaths                       one array of breath times, s, per annotator
    co2_artifacts                 CapnoBase only: (start, end) s pairs

    dataset    PPG             RR reference                     keep
    bidmc      finger, 125 Hz  2 annotators' breath marks       all
    dalia      wrist, 64 Hz    breaths found on the chest belt  low-motion samples
    capnobase  finger, 300 Hz  expert breath marks on CO2       no PPG artifact

The waveform then follows the path of CFMamba's estimate (`windows`): resampled to
30 Hz, cut into full 10 s windows, each processed apart. Only the input changes, so
a row means the same thing on camera and contact sources.

**RR is 60 over the mean breath interval** inside the segment, from the breaths
that fall in it (`breath_rate`); fewer than `MIN_BREATHS` gives null, and so does a
rate outside `RR_VALID`. BIDMC averages over those of its 2 annotators that give a
rate (`annotated_rate`). DaLiA has no breath marks, so `breaths` finds them on the
RespiBAN belt.

**BIDMC subjects** are the record's MIMIC II patient, from the record's `_Fix.txt`:
records 06-09, 20-23 and 38-39 are 3 people. Each DaLiA recording and each
CapnoBase case is its own subject.

**DaLiA keeps low-motion samples only**: `sitting`, `driving`, `lunch` or `working`
(`LOW_MOTION`). A wrist PPG during stairs or cycling is mostly motion.

**Polarity.** All 3 store the pulse upright (upstroke the short part of the cycle),
so nothing is inverted here; the camera rows are, in `rr_table`.

DaLiA's pickles are 1.45 GB each and contain only numpy arrays; `_load_pickle`
fails on any other class, so a tampered file can't run code on load.
"""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
import polars as pl
from scipy import signal as sps

from ..paths import DATA_ROOT
from .waves import WAVE_FPS

BIDMC_ROOT = DATA_ROOT / "bidmc"
CAPNOBASE_ROOT = DATA_ROOT / "capnobase"
DALIA_ROOT = DATA_ROOT / "ppg_dalia" / "PPG_FieldStudy"

WINDOW_S = 10.0
MIN_BREATHS = 3
RR_VALID = (4.0, 60.0)

BIDMC_FS = 125.0
CAPNOBASE_FS = 300.0
DALIA_PPG_FS = 64.0
DALIA_RESP_FS = 700.0
DALIA_ACTIVITY_FS = 4.0
# `sitting`, `driving`, `lunch`, `working` (PPG_FieldStudy_readme, appendix).
LOW_MOTION = (1, 5, 6, 8)

# Breath detection on the DaLiA belt. The belt is decimated to `RESP_FS` and
# band-passed to `RESP_BAND`, 6-42 breaths/min. A breath is a peak with a
# prominence of `PROMINENCE_FRAC` or more of the recording's upper-quartile
# prominence, at least `60 / RR_VALID[1]` s after the previous one.
RESP_FS = 25.0
RESP_BAND = (0.1, 0.7)
PROMINENCE_FRAC = 0.3

RECORD_KEYS = ("clip_id", "source", "subject_id", "ppg", "fs", "keep", "breaths")


def windows(
    clip_id: str, ppg: np.ndarray, fs: float, keep: np.ndarray | None = None
) -> pl.DataFrame:
    """Full 10 s windows of `ppg` at 30 Hz: `clip_id`, `window_start_s` and
    `ppg_true`, Array(Float32, 300). `keep`, one bool per input sample, removes each
    window that contains a False sample; a window with a non-finite sample is
    removed too."""
    ratio = Fraction(WAVE_FPS / fs).limit_denominator(1000)
    finite = np.isfinite(ppg)
    good = finite if keep is None else finite & keep
    # Gaps are filled before resampling so the filter doesn't smear NaN across a
    # full window; the windows that contained a NaN are removed below.
    y = sps.resample_poly(
        np.where(finite, ppg, np.nanmean(ppg)), ratio.numerator, ratio.denominator
    )
    n, span = round(WINDOW_S * WAVE_FPS), round(WINDOW_S * fs)
    count = min(len(y) // n, len(ppg) // span)
    return pl.DataFrame({
        "clip_id": clip_id,
        "window_start_s": np.arange(count, dtype=np.float64) * WINDOW_S,
        "ppg_true": pl.Series(y[: count * n].reshape(count, n).astype(np.float32)),
    }).filter(pl.Series(good[: count * span].reshape(count, span).all(axis=1)))  # fmt: skip


def breath_rate(clip_id: str, breaths_s: np.ndarray, segment_s: float) -> pl.DataFrame:
    """RR per segment from breath times, s: 60 over the mean interval of the
    breaths inside it, null below `MIN_BREATHS`, kept only inside `RR_VALID`."""
    rate = 60.0 * (pl.len() - 1) / (pl.col("t").max() - pl.col("t").min())
    return (
        pl.DataFrame({"t": np.asarray(breaths_s, dtype=np.float64)})
        .group_by(segment_start_s=(pl.col("t") / segment_s).floor() * segment_s)
        .agg(rr_bpm=pl.when(pl.len() >= MIN_BREATHS).then(rate))
        .with_columns(clip_id=pl.lit(clip_id),
                      rr_bpm=pl.when(pl.col("rr_bpm").is_between(*RR_VALID)).then("rr_bpm"))
        .select("clip_id", "segment_start_s", "rr_bpm")
    )  # fmt: skip


def annotated_rate(
    clip_id: str, breaths: list[np.ndarray], segment_s: float
) -> pl.DataFrame:
    """RR per segment from 1 or more annotators' breath times, s: `breath_rate` per
    annotator, then the mean over the annotators that give a rate."""
    keys = ["clip_id", "segment_start_s"]
    rates = [breath_rate(clip_id, b, segment_s).rename({"rr_bpm": f"rr_{i}"})
             for i, b in enumerate(breaths)]  # fmt: skip
    joined = rates[0]
    for rate in rates[1:]:
        joined = joined.join(rate, on=keys, how="full", coalesce=True)
    return joined.select(*keys, rr_bpm=pl.mean_horizontal(pl.exclude(keys)))


def breaths(resp: np.ndarray, fs: float) -> np.ndarray:
    """Breath times, s, on a respiration belt: band-passed peaks above a minimum
    prominence set from the full recording."""
    ratio = Fraction(RESP_FS / fs).limit_denominator(1000)
    x = sps.resample_poly(resp - np.mean(resp), ratio.numerator, ratio.denominator)
    y = sps.sosfiltfilt(
        sps.butter(2, RESP_BAND, btype="bandpass", fs=RESP_FS, output="sos"), x
    )
    _, props = sps.find_peaks(y, prominence=0)
    if props["prominences"].size == 0:
        return np.empty(0)
    floor = PROMINENCE_FRAC * np.percentile(props["prominences"], 75)
    peaks, _ = sps.find_peaks(
        y, prominence=floor, distance=max(1, round(60.0 / RR_VALID[1] * RESP_FS))
    )
    return peaks / RESP_FS


def _load_pickle(path: Path) -> dict:
    """A DaLiA `SX.pkl`, rebuilding numpy arrays and no other object."""
    allowed = {("numpy", "dtype"), ("numpy", "ndarray"),
               ("numpy.core.multiarray", "_reconstruct"),
               ("numpy._core.multiarray", "_reconstruct"),
               # Protocol 2 written from Python 3 rebuilds bytes through it.
               ("_codecs", "encode")}  # fmt: skip

    class Arrays(pickle.Unpickler):
        def find_class(self, module: str, name: str):
            if (module, name) not in allowed:
                raise pickle.UnpicklingError(f"{path} wants {module}.{name}")
            # Written by numpy 1.x; numpy 2 moved numpy.core to numpy._core.
            return super().find_class(module.replace("numpy.core", "numpy._core"), name)

    with open(path, "rb") as fh:
        return Arrays(fh, encoding="latin1").load()


def bidmc_records(root: Path = BIDMC_ROOT) -> Iterator[dict]:
    """One record per BIDMC recording under `root/bidmc_csv`: the raw PLETH, no
    mask, the 2 annotators' breaths."""
    for path in sorted((root / "bidmc_csv").glob("bidmc_*_Signals.csv")):
        record = path.name.split("_")[1]
        stem = path.with_name(f"bidmc_{record}")
        # Gaps are written as NaN, which polars infers as a string unless told.
        sig, ann = (
            pl.read_csv(p, null_values="NaN", infer_schema_length=None).rename(
                lambda c: c.strip()
            )
            for p in (path, f"{stem}_Breaths.csv")
        )
        fix = Path(f"{stem}_Fix.txt")
        patient = (
            re.search(r"wdb ID:\s*(\S+)", fix.read_text()) if fix.exists() else None
        )
        yield {"clip_id": f"bidmc_{record}", "source": "bidmc",
               "subject_id": patient[1] if patient else record,
               "ppg": sig["PLETH"].to_numpy(), "fs": BIDMC_FS, "keep": None,
               "breaths": [ann[c].drop_nulls().to_numpy() / BIDMC_FS for c in ann.columns]}  # fmt: skip


def dalia_records(root: Path = DALIA_ROOT) -> Iterator[dict]:
    """One record per DaLiA subject: the wrist PPG, `keep` True on low-motion
    samples, and the breaths found on the chest belt."""
    for path in sorted(root.glob("S*/S*.pkl")):
        data = _load_pickle(path)
        ppg = data["signal"]["wrist"]["BVP"][:, 0]
        activity = data["activity"][:, 0]
        # Activity at 4 Hz mapped to the 64 Hz PPG grid, sample by sample.
        index = np.minimum((np.arange(ppg.size) / DALIA_PPG_FS * DALIA_ACTIVITY_FS)
                           .astype(int), activity.size - 1)  # fmt: skip
        found = breaths(data["signal"]["chest"]["Resp"][:, 0], DALIA_RESP_FS)
        del data
        yield {"clip_id": f"dalia_{path.stem}", "source": "dalia", "subject_id": path.stem,
               "ppg": ppg, "fs": DALIA_PPG_FS, "keep": np.isin(activity[index], LOW_MOTION),
               "breaths": [found]}  # fmt: skip


def capnobase_records(root: Path = CAPNOBASE_ROOT) -> Iterator[dict]:
    """One record per CapnoBase case under `root/data/csv`: the pleth, `keep` False
    on marked PPG artifacts, the CO2 inspiration starts as 1 annotator, and
    `co2_artifacts`, (start, end) s pairs. A mark column is a space-separated list of
    sample numbers."""
    for path in sorted((root / "data" / "csv").glob("*_signal.tab")):
        case = path.name.removesuffix("_signal.tab")
        pleth = pl.read_csv(
            path,
            separator="\t",
            columns=["pleth_y"],
            schema_overrides={"pleth_y": pl.Float64},
        )["pleth_y"].to_numpy()
        row = pl.read_csv(
            path.with_name(f"{case}_labels.tab"), separator="\t", infer_schema=False
        ).row(0, named=True)
        if (row.get("units_x") or "").strip() != "samples":
            raise ValueError(f"{case}: marks in {row.get('units_x')!r}, not samples")
        marks = {
            k: np.array((row.get(k) or "").split(), dtype=np.float64) / CAPNOBASE_FS
            for k in ("co2_artif_x", "pleth_artif_x", "co2_startinsp_x")
        }
        clean = np.ones(pleth.size, dtype=bool)
        for a, b in marks["pleth_artif_x"].reshape(-1, 2):
            clean[round(a * CAPNOBASE_FS) : round(b * CAPNOBASE_FS) + 1] = False
        yield {"clip_id": f"capnobase_{case}", "source": "capnobase", "subject_id": case,
               "ppg": pleth, "fs": CAPNOBASE_FS, "keep": clean,
               "breaths": [marks["co2_startinsp_x"]],
               "co2_artifacts": marks["co2_artif_x"].reshape(-1, 2)}  # fmt: skip


# In the order the AH table was built: the order sets the rows `balance` sees.
READERS = (bidmc_records, dalia_records, capnobase_records)
