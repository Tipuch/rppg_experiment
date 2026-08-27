"""Plot training and dev loss per epoch for one or more runs.

    uv run python tools/plot_training.py build/runs/<run> [...] --out loss.png
    uv run python tools/plot_training.py build/runs/kfold_top12 --kfold --folds 5

Both curves are the same quantity -- MSE with targets z-scored on the training
split -- so they share one axis. They do NOT share one baseline. Predicting the
training mean scores about 0.97 on train but 0.46 on the old 4-clip dev, because
that dev's heart rates were less spread out, so each split gets its own dashed
reference line. Reading the two curves against a single line is the mistake this
chart exists to prevent.

Under --kfold every fold holds out different subjects, so every panel gets its own
pair of baselines, recomputed from that fold's actual partition.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Categorical slots 1 and 2 of the reference palette. Validated together for the
# light surface: CVD dE 24.7, normal-vision dE 33.6, both above 3:1 contrast.
TRAIN_COLOUR = "#2a78d6"
DEV_COLOUR = "#eb6834"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"


# Under neg_pearson the reference is not a property of the labels: 1.0 is what
# uncorrelated signals score and 0.0 is an exact match, the same for every split.
NEG_PEARSON_REFERENCE = 1.0


def _pair(train: np.ndarray, evaluation: np.ndarray) -> tuple[float, float]:
    """Predict-the-training-mean loss, z-scored on the training split."""
    mean, std = train.mean(), train.std(ddof=1)
    return (float((((train - mean) / std) ** 2).mean()),
            float((((evaluation - mean) / std) ** 2).mean()))


def collect(args) -> list[tuple[str, dict, float, float]]:
    from src.model.dataset import load_manifest, split_manifest

    manifest = load_manifest(args.manifest)
    target = args.target
    panels: list[tuple[str, dict, float, float]] = []

    pearson = args.reference == "neg-pearson"

    if pearson:
        base = (NEG_PEARSON_REFERENCE, NEG_PEARSON_REFERENCE)
    else:
        splits = split_manifest(manifest)
        base = _pair(splits["train"][target].to_numpy(), splits["dev"][target].to_numpy())
    titles = args.titles.split("|") if args.titles else [r.name for r in args.runs]
    for run, title in zip(args.runs, titles, strict=False):
        history = json.loads((run / "history.json").read_text())["history"]
        panels.append((title, history, *base))
    return panels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--titles", default=None, help="Panel titles separated by '|'.")
    parser.add_argument("--target", default="hr_bpm")
    parser.add_argument("--reference", choices=["predict-mean", "neg-pearson"],
                        default="predict-mean",
                        help="What the dashed line means. predict-mean draws each "
                             "split's z-scored MSE for predicting the training mean; "
                             "neg-pearson draws 1.0, the score of uncorrelated "
                             "signals, which is split-independent.")
    parser.add_argument("--manifest", type=Path, default=Path("build/clips_clean_ubfc.parquet"))
    parser.add_argument("--suptitle", default="UBFC heart rate: training and dev loss per epoch")
    parser.add_argument("--out", type=Path, default=Path("build/runs/loss.png"))
    args = parser.parse_args()

    panels = collect(args)
    columns = min(len(panels), 3)
    rows = math.ceil(len(panels) / columns)
    top = 0.90 if rows > 1 else 0.86

    figure, axes = plt.subplots(
        rows, columns, figsize=(5.6 * columns, 4.3 * rows),
        sharey=True, squeeze=False, facecolor=SURFACE,
    )
    flat = axes.ravel()
    ceiling = max(max(max(h["train_loss"] for h in hist),
                      max(h.get("dev_loss", 0) or 0 for h in hist))
                  for _, hist, _, _ in panels) * 1.06

    for axis, (title, history, train_base, dev_base) in zip(flat, panels, strict=False):
        epochs = [r["epoch"] for r in history]
        train_loss = [r["train_loss"] for r in history]
        dev_loss = [r.get("dev_loss", float("nan")) for r in history]

        axis.set_facecolor(SURFACE)
        axis.grid(axis="y", color=GRID, linewidth=1, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylim(0, ceiling)
        # One shared dashed line when both references coincide, as they do under
        # neg_pearson where 1.0 is split-independent.
        single_reference = train_base == dev_base

        # Each split's own predict-the-mean level, coloured to match its curve.
        axis.axhline(train_base,
                     color=INK_MUTED if single_reference else TRAIN_COLOUR,
                     linewidth=1.5, linestyle=(0, (4, 3)), alpha=0.55)
        if not single_reference:
            axis.axhline(dev_base, color=DEV_COLOUR, linewidth=1.5,
                         linestyle=(0, (4, 3)), alpha=0.55)
        axis.plot(epochs, train_loss, color=TRAIN_COLOUR, linewidth=2, label="train", zorder=3)
        axis.plot(epochs, dev_loss, color=DEV_COLOUR, linewidth=2, label="dev", zorder=3)

        # A fold whose held-out subjects happen to match the training spread puts
        # the two baselines on top of each other, and the same for two curves that
        # converge. Push the dev label to the far side rather than let them stack.
        close = 0.09 * ceiling
        base_dy = -9 if abs(train_base - dev_base) < close else 4
        end_train, end_dev = (0, 0)
        if abs(train_loss[-1] - dev_loss[-1]) < close:
            end_train, end_dev = 6, -6
        axis.annotate("train", (epochs[-1], train_loss[-1]), xytext=(6, end_train),
                      textcoords="offset points", color=INK, fontsize=9.5, va="center")
        axis.annotate("dev", (epochs[-1], dev_loss[-1]), xytext=(6, end_dev),
                      textcoords="offset points", color=INK, fontsize=9.5, va="center")
        if single_reference:
            axis.annotate(f"uncorrelated {train_base:.2f}", (epochs[0], train_base),
                          xytext=(2, 4), textcoords="offset points",
                          color=INK_MUTED, fontsize=8)
        else:
            axis.annotate(f"mean, train {train_base:.2f}", (epochs[0], train_base),
                          xytext=(2, 4), textcoords="offset points",
                          color=INK_MUTED, fontsize=8)
            axis.annotate(f"mean, dev {dev_base:.2f}", (epochs[0], dev_base),
                          xytext=(2, base_dy), textcoords="offset points",
                          color=INK_MUTED, fontsize=8)

        axis.set_title(title, color=INK, fontsize=11.5, fontweight="bold", loc="left", pad=9)
        axis.set_xlabel("epoch", color=INK_MUTED, fontsize=9.5)
        axis.set_xlim(epochs[0], epochs[-1] + max(1.6, len(epochs) * 0.1))
        axis.tick_params(colors=INK_MUTED, labelsize=9)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)

    for spare in flat[len(panels):]:
        spare.set_visible(False)

    for index in range(0, len(panels), columns):
        flat[index].set_ylabel("loss  (MSE, z-scored on train)", color=INK_MUTED, fontsize=9.5)
    flat[0].legend(frameon=False, loc="upper right", fontsize=9.5, labelcolor=INK)

    figure.suptitle(args.suptitle, color=INK, fontsize=14, fontweight="bold",
                    x=0.006, ha="left", y=0.995)
    if args.reference == "neg-pearson":
        caption = (
            "Dashed line at 1.0 is what uncorrelated signals score; 0.0 is exact.",
            "Loss is 1 - Pearson correlation between predicted and measured pulse.",
        )
    else:
        caption = (
            "Dashed lines are each split's predict-the-mean loss; they differ per panel",
            "because each fold holds out different people. Read each against its own line.",
        )
    figure.text(0.006, top + 0.055, caption[0],
                color=INK_MUTED, fontsize=9.5, ha="left", va="top")
    figure.text(0.006, top + 0.028, caption[1],
                color=INK_MUTED, fontsize=9.5, ha="left", va="top")
    figure.tight_layout(rect=(0, 0, 1, top))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
