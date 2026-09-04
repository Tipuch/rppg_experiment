"""Video-level metrics: MAE, RMSE, MAPE, Pearson rho and SNR.

CFMamba Eqs. 22-27. Every number reported for this project comes through here, so
the definitions are recorded here:

  MAE, RMSE, MAPE  over per-window heart rates, in bpm
  rho              Pearson correlation between predicted and true rate *across
                   windows*, a different question from whether any one estimate is
                   close. A constant predictor scores a plausible MAE and a rho of
                   zero, which is why both are reported.
  SNR              in dB, on the predicted waveform against the true rate

The ground-truth rate is read from the contact PPG over the same window, never
from the manifest's label column -- see src/model/postprocess.compare.

POS and CHROM are the classical floor these numbers are read against. Neither
needs training, and on UBFC-rPPG they publish at 4.08 and 4.06 bpm MAE.
`src.cli baseline` scores them on the same windows.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch

from .losses import DEFAULT_ALPHA, DEFAULT_BETA, composite_loss
from .postprocess import compare


def summarise(rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Aggregate per-window results into the five reported metrics.

    Windows whose rate could not be estimated -- a flat prediction, a clip with no
    contact PPG -- are dropped and counted. Dropping them without a count would let
    a model that fails on the harder windows report the score of the rest.
    """
    rows = list(rows)
    usable = [
        r for r in rows
        if math.isfinite(r.get("hr_pred", float("nan")))
        and math.isfinite(r.get("hr_true", float("nan")))
    ]
    # Over every window, including the ones whose rate could not be read. A window
    # the readout failed on still has a defined loss, and dropping it here would
    # make the dev loss an average over whichever windows worked that epoch: a
    # moving denominator, which a loss curve cannot have.
    terms = {
        name: float(np.mean(values))
        for name in ("loss", "time", "freq")
        if (values := [r[name] for r in rows if name in r])
    }
    if not usable:
        return {"windows": len(rows), "dropped": len(rows), **terms}

    predicted = np.array([r["hr_pred"] for r in usable])
    truth = np.array([r["hr_true"] for r in usable])
    error = predicted - truth
    snr_values = np.array([r.get("snr", np.nan) for r in usable])
    macc_values = np.array([r.get("macc", np.nan) for r in usable])

    # A single window, or a constant prediction, leaves rho undefined rather than
    # zero. Reporting 0.0 would read as a measured absence of correlation when
    # nothing was measurable.
    if len(usable) > 1 and predicted.std() > 1e-9 and truth.std() > 1e-9:
        rho = float(np.corrcoef(predicted, truth)[0, 1])
    else:
        rho = float("nan")

    return {
        "windows": len(rows),
        "dropped": len(rows) - len(usable),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt((error**2).mean())),
        "mape": float((np.abs(error) / np.maximum(truth, 1e-6)).mean() * 100.0),
        "rho": rho,
        "snr": float(np.nanmean(snr_values)) if np.isfinite(snr_values).any() else float("nan"),
        "macc": float(np.nanmean(macc_values)) if np.isfinite(macc_values).any() else float("nan"),
        "hr_true_std": float(truth.std()),
        "hr_pred_std": float(predicted.std()),
        **terms,
    }


def format_metrics(name: str, metrics: dict[str, float]) -> str:
    if "mae" not in metrics:
        return f"{name:16} no usable windows ({metrics.get('windows', 0)} attempted)"
    dropped = f"  ({metrics['dropped']} dropped)" if metrics["dropped"] else ""
    return (
        f"{name:16} MAE {metrics['mae']:6.2f}  RMSE {metrics['rmse']:6.2f}  "
        f"MAPE {metrics['mape']:5.2f}%  rho {metrics['rho']:+.3f}  "
        f"SNR {metrics['snr']:+6.2f} dB  MACC {metrics['macc']:.3f}  "
        f"n={metrics['windows']}{dropped}"
    )


@torch.no_grad()
def evaluate(
    model, loader, device: str = "cuda", fps: float = 30.0,
    alpha: float = DEFAULT_ALPHA, beta: float = DEFAULT_BETA,
) -> list[dict]:
    """Per-window predictions, metrics and loss for one loader.

    Returns rows rather than an aggregate, so a caller can group by subject, look
    at the worst clips, or re-aggregate without a second forward pass.

    The loss is scored here too, per window. Heart-rate MAE is what the papers
    report, but MAE over a handful of subjects is quantised to the periodogram bin
    spacing and swings by several bpm between epochs on sampling noise. A dev loss
    moves on the same scale as the training loss and shows the generalisation gap.
    Eq. 19's two terms are kept apart because they fail differently.

    The forward pass already happened, so this costs one extra correlation and one
    extra periodogram per window.
    """
    model.eval()
    rows: list[dict] = []
    for batch in loader:
        frames = batch["frames"].to(device, non_blocking=True)
        skin = batch["skin"].to(device, non_blocking=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            predicted = model(frames, skin)
        predicted = predicted.float()
        target = batch["wave"].to(device, non_blocking=True)
        # Per window, not per batch. The batch mean would weight a short trailing
        # batch the same as a full one, and these splits are small enough for that
        # to move the number.
        losses = [
            {k: float(v) for k, v in composite_loss(
                predicted[i : i + 1], target[i : i + 1], fps=fps,
                alpha=alpha, beta=beta,
            )[1].items()}
            for i in range(len(predicted))
        ]
        predicted = predicted.cpu().numpy()
        truth = batch["wave"].numpy()
        scale = batch["fps_scale"].numpy()
        for i in range(len(predicted)):
            # The window may have been time-stretched by the augmentation. Undoing
            # it here is what keeps a reported bpm a real bpm: the loader resampled
            # the decode to fps*k, so a rate read back at fps is k times the truth.
            result = compare(predicted[i], truth[i], fps=fps)
            for key in ("hr_pred", "hr_true"):
                result[key] = result[key] * float(scale[i])
            rows.append({
                "clip_id": batch["clip_id"][i],
                "subject_id": batch["subject_id"][i],
                "source": batch["clip_id"][i].split("/")[0],
                **result,
                **losses[i],
            })
    return rows


def per_source(rows: list[dict]) -> list[tuple[str, dict[str, float]]]:
    """Metrics grouped by corpus. Required once the split holds more than one.

    On the pooled split MCD-rPPG is ~98% of the test segments, and it is the corpus
    whose video measured below chance for a recoverable pulse (DATASETS.md). An
    aggregate over that is largely a measurement of MCD. Reporting by source keeps
    "did it learn a pulse" apart from "did it learn MCD's population statistics".
    """
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("source", "?")), []).append(row)
    return sorted(
        ((name, summarise(group)) for name, group in groups.items()),
        key=lambda pair: -pair[1].get("windows", 0),
    )


def per_subject(rows: list[dict]) -> list[tuple[str, dict[str, float]]]:
    """Metrics grouped by subject, worst MAE first.

    An aggregate hides which people the model fails on, and over a few dozen
    subjects one poor subject moves the mean.
    """
    subjects: dict[str, list[dict]] = {}
    for row in rows:
        subjects.setdefault(row["subject_id"], []).append(row)
    scored = [(name, summarise(group)) for name, group in subjects.items()]
    return sorted(scored, key=lambda pair: -pair[1].get("mae", -1.0))
