"""The RR network's train-versus-dev figure (`rr_mamba_loss`), 2 panels from its
history parquet:

    left    train and dev MAE per epoch, both in eval mode on natural rows, so the
            gap between them is the overfit; the kept epoch (highest dev slope)
            marked; the dev error of predicting the natural train mean as a floor
    right   dev slope and spread per epoch, 1 for no pull to the mean

Palette and conventions follow src/model/predict_plot.py: CVD-validated on this
surface, and each series is named in a legend and directly, so colour isn't the
only cue to identity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl
from matplotlib.ticker import MaxNLocator

from ..model.predict_plot import GRID, INK, INK_MUTED, SURFACE, TRACE, TRUTH

BASELINE = "#a3a29c"  # the neutral the reference palette keeps for "no model"
# One look for all axes: recessive grid and spines, muted ticks.
STYLE = {
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": INK_MUTED, "ytick.color": INK_MUTED, "xtick.labelsize": 9,
    "ytick.labelsize": 9, "axes.labelcolor": INK_MUTED, "axes.labelsize": 10,
}  # fmt: skip


def rr_mamba_loss(history: Path, out: Path | None = None) -> Path:
    """Write the figure from `history` to `out` (default: the history's name with
    `_loss.png` for `_history.parquet`). The history holds everything drawn, the
    baseline included, so a finished run can be re-plotted from that one file."""
    rows = pl.read_parquet(history)
    baseline = rows["baseline_dev_mae"][0]
    out = out or history.with_name(
        history.name.replace("_history.parquet", "_loss.png")
    )
    epoch = rows["epoch"].to_numpy()
    best = int(rows["dev_slope"].arg_max())
    with plt.rc_context(STYLE):
        figure, (loss, pull) = plt.subplots(
            1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.6, 1]}
        )
    for axis in (loss, pull):
        axis.axvline(epoch[best], color=GRID, linewidth=1.5, zorder=0)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlabel("epoch")
    note = {"textcoords": "offset points", "fontsize": 9, "color": INK_MUTED}

    for column, name, colour in (
        ("train_mae", "train", TRACE),
        ("dev_mae", "dev", TRUTH),
    ):
        y = rows[column].to_numpy()
        loss.plot(
            epoch,
            y,
            color=colour,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"{name} MAE",
        )
        loss.annotate(name, (epoch[-1], y[-1]), xytext=(6, 0), va="center", **note)
    dev_best = rows["dev_mae"][best]
    loss.plot(epoch[best], dev_best, marker="o", markersize=9, color=TRUTH, markeredgecolor=SURFACE,
              markeredgewidth=2, linestyle="none", label="highest dev slope epoch (kept)")  # fmt: skip
    loss.annotate(f"kept: epoch {epoch[best]}, dev slope {rows['dev_slope'][best]:.2f}, "
                  f"RMSE {rows['dev_rmse'][best]:.2f}, MAE {dev_best:.2f}",
                  (epoch[best], dev_best), xytext=(0, -16), ha="center", va="top", **note)  # fmt: skip
    loss.axhline(
        baseline,
        color=BASELINE,
        linewidth=1.5,
        linestyle="--",
        label="dev, predicting the train mean",
    )
    loss.annotate(f"train mean on dev {baseline:.2f}", (epoch[-1], baseline), xytext=(0, 4),
                  ha="right", va="bottom", **note)  # fmt: skip
    loss.set_ylabel("MAE (breaths/min)")
    loss.set_title(
        "MAE per epoch: train and dev, eval mode, natural rows",
        color=INK,
        fontsize=11,
        loc="left",
    )
    loss.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="upper right")

    # Both dev, so both wear the dev colour; line style and labels tell them apart.
    pull.axhline(
        1.0,
        color=INK_MUTED,
        linewidth=1,
        linestyle="--",
        label="1: no pull to the mean",
    )
    for column, name, style, marker in (
        ("dev_slope", "slope", "-", "o"),
        ("dev_spread", "spread", ":", "s"),
    ):
        y = rows[column].to_numpy()
        pull.plot(epoch, y, color=TRUTH, linewidth=2, linestyle=style, marker=marker, markersize=3,
                  label=f"dev {name}")  # fmt: skip
        pull.annotate(name, (epoch[-1], y[-1]), xytext=(6, 0), va="center", **note)
    pull.set_ylabel("ratio")
    pull.set_title("Dev slope and spread per epoch", color=INK, fontsize=11, loc="left")
    pull.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED, loc="lower right")

    figure.suptitle("Respiratory rate -- RR Mamba, train against dev per epoch", color=INK,
                    fontsize=13, x=0.01, ha="left")  # fmt: skip
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return out
