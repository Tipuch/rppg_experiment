"""Render what the model is actually fed during training.

    uv run python tools/sample_batch.py --clips 4 --frames 6

Not a reconstruction: this pulls tensors out of WindowDataset with train=True and
renders them straight, so what you see is the array that reaches the first
convolution -- face box applied, resized to 256, skin mask zeroing everything
else, random 224 crop, random window start.

Also plots each clip's mean skin luma across the window. That trace is where the
pulse lives: a cardiac cycle is a ~0.1-0.5 LSB brightness change spread smoothly
over the skin, invisible frame to frame but periodic over time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.paths import BUILD_ROOT

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
TRACE = "#2a78d6"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=BUILD_ROOT / "clips_clean_ubfc.parquet")
    parser.add_argument("--clips", type=int, default=4, help="Clips to sample.")
    parser.add_argument("--frames", type=int, default=6, help="Frames shown per clip.")
    parser.add_argument("--window", type=int, default=160, help="Window length the model gets.")
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--out", type=Path,
                        default=BUILD_ROOT / "samples" / "train_batch.png")
    args = parser.parse_args()

    from src.model.dataset import WindowDataset

    manifest = pl.read_parquet(args.manifest)
    picked = manifest.head(args.clips)
    # compute_mean_y=True only so the trace can be drawn; training leaves it off,
    # and it changes nothing about the frames themselves.
    dataset = WindowDataset(
        picked, n_frames=args.window, train=True, seed=args.seed,
        compute_mean_y=True, targets=("hr_bpm",),
    )

    columns = args.frames + 1
    figure, axes = plt.subplots(
        len(picked), columns, figsize=(2.05 * columns, 2.35 * len(picked)),
        squeeze=False, facecolor=SURFACE,
    )

    for row, index in enumerate(range(len(picked))):
        item = dataset[index]
        frames = item["frames"].numpy()           # (T, 3, 224, 224), float 0-1
        trace = item["mean_y"].numpy()
        hr = float(item["targets"][0])
        clip = item["clip_id"].replace("ubfc/", "")
        shown = np.linspace(0, len(frames) - 1, args.frames).astype(int)

        for column, frame_index in enumerate(shown):
            axis = axes[row][column]
            axis.imshow(np.transpose(frames[frame_index], (1, 2, 0)))
            axis.set_xticks([]); axis.set_yticks([])
            for side in axis.spines.values():
                side.set_color(GRID)
            if row == 0:
                axis.set_title(f"frame {frame_index}", color=INK_MUTED, fontsize=9, pad=6)
            if column == 0:
                axis.set_ylabel(f"{clip}\n{hr:.0f} bpm", color=INK, fontsize=9,
                                rotation=0, ha="right", va="center", labelpad=12)

        axis = axes[row][columns - 1]
        axis.set_facecolor(SURFACE)
        axis.plot(trace, color=TRACE, linewidth=1.2)
        axis.set_xticks([]); axis.tick_params(colors=INK_MUTED, labelsize=7.5)
        axis.grid(axis="y", color=GRID, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        if row == 0:
            axis.set_title("mean skin luma\nacross the window", color=INK_MUTED,
                           fontsize=9, pad=6)
        axis.annotate(f"range {np.ptp(trace):.1f} LSB", (0.5, 0.03),
                      xycoords="axes fraction", ha="center", color=INK_MUTED, fontsize=7.5)

    figure.suptitle(
        f"What the network is fed: {args.window}-frame windows, 224x224, skin only",
        color=INK, fontsize=13.5, fontweight="bold", x=0.006, ha="left", y=0.995,
    )
    figure.text(0.006, 0.955,
                "Face box, resize to 256, skin mask zeroing the rest, random 224 crop, "
                "random window start. Six frames sampled from each 160.",
                color=INK_MUTED, fontsize=9, ha="left", va="top")
    figure.tight_layout(rect=(0, 0, 1, 0.935))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=150, facecolor=SURFACE)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
