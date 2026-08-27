"""Plot a CFMamba-Phys training run from its history.json.

    uv run python tools/plot_loss.py build/runs/<name> [--out figure.png]

Six panels, because a run here can fail in ways one loss curve cannot separate.

  total loss      train against dev. This is the generalisation gap, and it is the
                  panel to read first: heart-rate MAE on a handful of held-out
                  subjects is quantised by the periodogram bin spacing and swings
                  several bpm between epochs on sampling noise alone, so a dev MAE
                  that looks flat can sit on top of a dev loss that is diverging.
  temporal        1 - Pearson, Eq. 19's first term. Stuck near 1.0 means the
                  predicted waveform is uncorrelated with the target whatever the
                  frequency term is doing.
  frequency       the cross-entropy term. Falling to zero on train while the dev
                  term climbs is memorisation of the training spectra.
  HR error        MAE and RMSE in bpm on the watched split. RMSE alongside MAE
                  because one badly-missed window moves RMSE and hides in MAE.
  waveform        rho and MACC, both 0-1, both on the same axis because they are
                  the same kind of quantity.
  schedule        the warmup and cosine actually happening, and the wall clock
                  split between waiting on the dataloader and computing.

Runs written before dev loss was recorded simply omit those traces.
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

# Ink and chrome. One committed light design -- a PNG has no viewer theme to
# follow, so there is nothing to switch on.
INK = "#0b0b0b"
SECOND = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

# Categorical slots 1, 2, 3 in fixed order, validated against this surface:
# worst adjacent CVD dE 9.2, normal-vision dE 27.6. Aqua sits below 3:1 on the
# light surface, so every series is also direct-labelled at its right-hand end --
# identity is never carried by colour alone.
TRAIN = "#2a78d6"      # slot 1, blue
DEV = "#eb6834"        # slot 2, orange
THIRD = "#1baf7a"      # slot 3, aqua


def _panel(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    ax.set_xlabel("epoch", color=SECOND, fontsize=9)
    ax.set_ylabel(ylabel, color=SECOND, fontsize=9)
    ax.set_title(title, color=INK, loc="left", fontsize=10.5, pad=8)


def _series(ax, epochs, values, colour, label, last=True):
    """One 2px line with >=8px markers, direct-labelled at its right-hand end."""
    pairs = [(e, v) for e, v in zip(epochs, values, strict=False)
             if v is not None and not math.isnan(v)]
    if not pairs:
        return
    xs, ys = zip(*pairs, strict=True)
    ax.plot(xs, ys, color=colour, linewidth=2.0, marker="o", markersize=3.2,
            markeredgecolor=SURFACE, markeredgewidth=0.8, label=label,
            solid_capstyle="round")
    if last:
        ax.annotate(f" {label} {ys[-1]:.3g}", (xs[-1], ys[-1]), color=colour,
                    fontsize=8, va="center", ha="left",
                    xytext=(4, 0), textcoords="offset points")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="directory holding history.json")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    path = args.run / "history.json" if args.run.is_dir() else args.run
    if not path.exists():
        print(f"no history at {path}", file=sys.stderr)
        return 1
    result = json.loads(path.read_text())
    history = result.get("history", [])
    if not history:
        print("history is empty -- the run has not finished an epoch yet", file=sys.stderr)
        return 1

    epochs = [r["epoch"] for r in history]
    watch = "dev" if "dev" in history[0] else "test"
    config = result.get("config", {})

    def train_of(key):
        return [r.get(f"train_{key}") for r in history]

    def watch_of(key):
        return [r.get(f"{watch}_{key}") for r in history]

    def metric(key):
        return [r.get(watch, {}).get(key) for r in history]

    figure, axes = plt.subplots(2, 3, figsize=(16.5, 8.6), facecolor=SURFACE)
    figure.patch.set_facecolor(SURFACE)
    ax = axes.ravel()

    # --- 1. total loss, train against dev ----------------------------------
    _panel(ax[0], "total loss: train against held out", "loss (log)")
    _series(ax[0], epochs, train_of("loss"), TRAIN, "train")
    _series(ax[0], epochs, watch_of("loss"), DEV, watch)
    ax[0].set_yscale("log")
    ax[0].legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="lower left")
    # Where the two curves stop moving together is where the run stops learning
    # something transferable, and it is worth marking rather than eyeballing.
    dev_loss = [v for v in watch_of("loss") if v is not None]
    if dev_loss:
        best = min(range(len(dev_loss)), key=lambda i: dev_loss[i])
        ax[0].axvline(epochs[best], color=MUTED, linewidth=1.0, linestyle=":")
        ax[0].annotate(f"best {watch} loss, epoch {epochs[best]}",
                       (epochs[best], max(dev_loss)), color=MUTED, fontsize=8,
                       rotation=90, va="top", ha="right",
                       xytext=(-3, 0), textcoords="offset points")

    # --- 2. temporal term ---------------------------------------------------
    _panel(ax[1], "temporal term (1 - Pearson)", "Eq. 19 first term")
    _series(ax[1], epochs, train_of("time"), TRAIN, "train")
    _series(ax[1], epochs, watch_of("time"), DEV, watch)
    ax[1].axhline(1.0, color=MUTED, linestyle=":", linewidth=1.0)
    ax[1].annotate("uncorrelated", (epochs[0], 1.0), color=MUTED, fontsize=8,
                   va="bottom", xytext=(0, 3), textcoords="offset points")
    ax[1].legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="best")

    # --- 3. frequency term --------------------------------------------------
    _panel(ax[2], "frequency term (cross-entropy)", "Eq. 19 second term")
    _series(ax[2], epochs, train_of("freq"), TRAIN, "train")
    _series(ax[2], epochs, watch_of("freq"), DEV, watch)
    ax[2].set_yscale("log")
    ax[2].legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="best")

    # --- 4. heart-rate error ------------------------------------------------
    windows = history[-1].get(watch, {}).get("windows")
    _panel(ax[3], f"{watch} heart-rate error ({windows} windows)", "bpm")
    _series(ax[3], epochs, metric("mae"), DEV, "MAE")
    _series(ax[3], epochs, metric("rmse"), THIRD, "RMSE")
    ax[3].legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="best")

    # --- 5. waveform quality ------------------------------------------------
    _panel(ax[4], f"{watch} waveform quality", "0 - 1")
    _series(ax[4], epochs, metric("rho"), TRAIN, "rho")
    _series(ax[4], epochs, metric("macc"), THIRD, "MACC")
    ax[4].set_ylim(0.0, 1.05)
    ax[4].legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="lower right")

    # --- 6. schedule and wall clock ----------------------------------------
    _panel(ax[5], "learning rate, and where the epoch went", "learning rate (log)")
    _series(ax[5], epochs, [r.get("lr") for r in history], TRAIN, "lr")
    ax[5].set_yscale("log")
    if any("fetch_s" in r for r in history):
        twin = ax[5].twinx()
        twin.set_facecolor("none")
        for side in ("top", "left"):
            twin.spines[side].set_visible(False)
        twin.spines["right"].set_color(AXIS)
        twin.spines["bottom"].set_color(AXIS)
        twin.tick_params(colors=MUTED, labelsize=8, length=0)
        twin.set_ylabel("seconds per epoch", color=SECOND, fontsize=9)
        twin.bar(epochs, [r.get("seconds", 0) for r in history], color=GRID,
                 width=0.62, label="epoch")
        twin.bar(epochs, [r.get("fetch_s", 0) for r in history], color=DEV,
                 width=0.62, label="blocked on the loader")
        twin.set_ylim(0, max(r.get("seconds", 0) for r in history) * 2.4)
        twin.legend(frameon=False, fontsize=8, labelcolor=SECOND, loc="upper right")

    # --- title and the one comparison that decides the run ------------------
    test = result.get("test", {})
    floors = result.get("baselines", {})
    verdict = ""
    if "mae" in test and "pos" in floors:
        pos = floors["pos"].get("mae", float("nan"))
        beaten = "beats" if test["mae"] < pos else "does NOT beat"
        verdict = (f"   |   test MAE {test['mae']:.2f} bpm {beaten} POS {pos:.2f}"
                   f"  (n={test.get('windows')})")
    elif "mae" in test:
        verdict = f"   |   test MAE {test['mae']:.2f} bpm (n={test.get('windows')})"

    figure.suptitle(
        f"CFMamba-Phys  |  {config.get('epochs', '?')} epochs, batch "
        f"{config.get('batch_size', '?')}, lr {config.get('lr', '?')}, "
        f"alpha {config.get('alpha', '?')}, {config.get('protocol', '?')} split"
        f"{verdict}",
        color=INK, x=0.006, ha="left", fontsize=11.5,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    out = args.out or (args.run if args.run.is_dir() else args.run.parent) / "loss.png"
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    print(f"wrote {out}")

    last = history[-1][watch]
    if "mae" in last:
        print(f"last epoch: {watch} MAE {last['mae']:.2f} bpm  "
              f"rho {last.get('rho', float('nan')):+.3f}")
    if dev_loss:
        print(f"best {watch} loss {min(dev_loss):.4f} at epoch {epochs[best]}, "
              f"last {dev_loss[-1]:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
