"""The figure `cli predict` writes.

Palette and conventions are the ones in tools/plot_loss.py and tools/sample_batch.py:
CVD-validated against this surface, and every series direct-named because aqua
sits at 2.74:1 against it -- under 3:1, so colour alone cannot carry identity.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e3e2df"
TRACE = "#2a78d6"
TRUTH = "#eb6834"
BEAT = "#1baf7a"


def _label_ends(axis, x: float, labels: list[tuple[str, float]], gap: float = 0.16) -> None:
    """Direct-label each trace at its right-hand end, pushed apart if they overlap.

    `gap` is a fraction of the y range -- roughly the height of a line of text at
    this figure size.
    """
    low, high = axis.get_ylim()
    span = (high - low) or 1.0
    placed: list[float] = []
    for name, y in sorted(labels, key=lambda pair: pair[1]):
        for other in placed:
            if abs(y - other) < gap * span:
                y = other + gap * span
        placed.append(y)
        axis.annotate(name, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                      color=INK_MUTED, fontsize=9, va="center", annotation_clip=False)


def plot(
    trace: np.ndarray, peaks: np.ndarray, *, fps: float, start_s: float,
    bpm_fft: float, bpm_beats: float, seams: list[int], title: str, subtitle: str,
    out: Path, truth: np.ndarray | None = None, bpm_true: float | None = None,
) -> Path:
    """One panel: predicted BVP against time, with the detected beats marked.

    `truth` is the contact PPG over the same span. It shares the one y axis rather
    than getting a second: both traces are z-scored into the same units, so a
    second scale would invent a difference between them that is not there.
    """
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t = start_s + np.arange(len(trace)) / fps

    figure, axis = plt.subplots(figsize=(14, 4.6), facecolor=SURFACE)
    axis.set_facecolor(SURFACE)

    # Seams first, so the trace draws over them.
    for seam in seams:
        axis.axvline(start_s + seam / fps, color=GRID, linewidth=1.0, zorder=1)

    if truth is not None:
        axis.plot(t, truth[: len(t)], color=TRUTH, linewidth=1.4, alpha=0.9, zorder=2)
    axis.plot(t, trace, color=TRACE, linewidth=1.4, zorder=3)
    if len(peaks):
        axis.plot(t[peaks], trace[peaks], "o", color=BEAT, markersize=5,
                  markeredgecolor=SURFACE, markeredgewidth=1.2, zorder=4)

    # Direct labels at the right-hand end. Text in ink; the mark beside it carries
    # the colour. Two traces that happen to end at the same height would print one
    # label over the other, so they are nudged apart -- the label is the only thing
    # bringing identity here, since aqua and orange both sit under 3:1 against this
    # surface.
    labels = [("predicted BVP", float(trace[-1]))]
    if truth is not None:
        labels.append(("contact PPG", float(truth[len(t) - 1])))
    _label_ends(axis, t[-1], labels)

    if len(peaks):
        axis.annotate("beats", xy=(t[peaks[-1]], trace[peaks[-1]]), xytext=(7, 9),
                      textcoords="offset points", color=INK_MUTED, fontsize=9,
                      va="center", annotation_clip=False)

    axis.set_xlim(t[0], t[-1])
    axis.set_xlabel("seconds", color=INK_MUTED, fontsize=9.5)
    # Amplitude is not a measurement: the temporal loss is scale-invariant, so the
    # scale is whatever the window came back with. Only the timing is real.
    axis.set_ylabel("z-scored, arbitrary units", color=INK_MUTED, fontsize=9.5)
    axis.grid(axis="y", color=GRID, linewidth=0.8)
    axis.set_axisbelow(True)
    axis.tick_params(colors=INK_MUTED, labelsize=8.5)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axis.spines[side].set_color(GRID)

    beats_note = (
        f"{bpm_beats:.1f} bpm from {len(peaks)} beats" if np.isfinite(bpm_beats)
        else "too few beats to time"
    )
    headline = f"{bpm_fft:.1f} bpm"
    if bpm_true is not None and np.isfinite(bpm_true):
        headline += f"   vs {bpm_true:.1f} contact   ({bpm_fft - bpm_true:+.1f})"
    figure.text(0.008, 0.965, headline, color=INK, fontsize=19,
                fontweight="bold", va="top")
    figure.text(0.008, 0.885, f"dominant cardiac-band peak  ·  {beats_note}",
                color=INK_MUTED, fontsize=9.5, va="top")
    figure.text(0.998, 0.965, title, color=INK, fontsize=10.5, ha="right", va="top")
    figure.text(0.998, 0.905, subtitle, color=INK_MUTED, fontsize=9, ha="right",
                va="top")

    figure.tight_layout(rect=(0, 0, 1, 0.86))
    figure.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(figure)
    return out
