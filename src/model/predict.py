"""Run one video through a trained checkpoint and read a pulse off the output.

`evaluate` scores a split against contact PPG. This takes an unlabelled recording
and produces the predicted waveform plus a rate in bpm.

The window length is not an argument: it comes from the checkpoint's own config,
because the model's temporal axis is fixed at the length it was built with.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import numpy as np
import polars as pl
import torch
from scipy import signal as sps
from torch.utils.data import DataLoader

from .cfmamba import CFMambaPhys
from .dataset import TARGET_FPS, WindowDataset, expand_to_segments
from .postprocess import HIGH_HZ, bandpass, heart_rate, macc
from .train import TrainConfig, build_model, load_checkpoint
from .waveform import load_ppg

DEFAULT_RUNS = Path("build/runs")


def latest_checkpoint(runs_dir: Path = DEFAULT_RUNS) -> Path:
    """Most recently written checkpoint, across run directories.

    One level down only: `build/runs/best.pt` sits at the top level and carries no
    `config`, so `load_model` cannot rebuild an architecture for it.
    """
    runs_dir = Path(runs_dir)
    found = list(runs_dir.glob("*/final.pt")) + list(runs_dir.glob("*/last.pt"))
    if not found:
        raise FileNotFoundError(
            f"no checkpoint under {runs_dir}/*/ (looked for final.pt and last.pt). "
            "Train a model first, or pass one with --model."
        )
    return max(found, key=lambda p: p.stat().st_mtime)


def config_from_checkpoint(saved: dict) -> TrainConfig:
    """Rebuild a TrainConfig from the dict `asdict` flattened into the checkpoint.

    `asdict` stringified `out_dir` and turned the tuples into lists. Unknown keys
    are dropped and absent ones take the dataclass default, so a checkpoint written
    by an older revision still loads.
    """
    known = {f.name for f in fields(TrainConfig)}
    kept = {k: v for k, v in saved.items() if k in known}
    if "out_dir" in kept:
        kept["out_dir"] = Path(kept["out_dir"])
    for name in ("betas", "sources"):
        if name in kept and kept[name] is not None:
            kept[name] = tuple(kept[name])
    return TrainConfig(**kept)


def load_model(path: Path, device: str = "cuda") -> tuple[CFMambaPhys, TrainConfig, dict]:
    """Weights plus the configuration they were trained under.

    `build_model` is the constructor the training run used, so every ablation flag
    follows the file rather than this process's defaults.
    """
    state = load_checkpoint(Path(path))
    if "config" not in state:
        raise ValueError(
            f"{path} has no 'config' payload, so the architecture it was trained "
            "with cannot be recovered. It came before the current checkpoint format."
        )
    config = config_from_checkpoint(state["config"])
    model = build_model(config)
    model.load_state_dict(state["model"])
    return model.to(device).eval(), config, state


def clip_name(video: Path) -> str:
    """A name for this recording that includes its directory.

    The stem alone is not unique: every UBFC-rPPG recording is called `vid.avi`,
    so a plain stem would write all 42 skin masks to build/masks/vid.npy and each
    run would invisibly segment against the previous subject's face.
    """
    return f"{video.resolve().parent.name}__{video.stem}"


def prepare(
    video: Path, config: TrainConfig, start_s: float = 0.0,
    seconds: float | None = None,
) -> tuple[WindowDataset, pl.DataFrame, dict]:
    """Detect the face, segment the skin, and enumerate the windows to run.

    `clips.build_clip` is the per-clip half of the manifest builder and returns
    exactly the row `WindowDataset` reads: `box_x/box_y/box_side`, `mask_path`, fps
    and duration. Running it inline is what gives an unlisted video the same
    preprocessing a training window got.
    """
    from . import clips as clips_mod

    name = clip_name(video)
    row = clips_mod.build_clip(
        clip_id=name, source="user", subject_id=name,
        video=video, targets={"hr_bpm": None},
    )
    if row is None:
        raise ValueError(
            f"no usable face or skin in {video}. build_clip returns None when the "
            "container will not probe, when no frame decodes, or when SegFace finds "
            "no skin."
        )

    manifest = pl.DataFrame([row])
    segments = expand_to_segments(manifest, config.n_frames, TARGET_FPS)

    # --seconds is a span of the recording, not a window count. A window is kept
    # when it starts inside the span; one that runs past the end is still whole
    # footage, so it stays.
    span = config.n_frames / TARGET_FPS
    segments = segments.filter(pl.col("window_start_s") >= start_s - 1e-9)
    if seconds:
        segments = segments.filter(pl.col("window_start_s") < start_s + seconds - 1e-9)
    if segments.height == 0:
        raise ValueError(
            f"no {span:.1f}s window starts between {start_s:.1f}s and "
            f"{start_s + (seconds or 0):.1f}s of a {row['duration_s']:.1f}s recording."
        )

    # A recording that came out of one of the corpora has its contact PPG beside
    # it, and that is worth plotting against. An arbitrary video does not, and
    # `_waveform` would hand back zeros rather than raise -- so ask first.
    has_truth = load_ppg(video.parent, video_path=video, fps=row["fps"]) is not None

    dataset = WindowDataset(
        segments, n_frames=config.n_frames, train=False,
        resolution=config.resolution, frame_norm=config.frame_norm,
        apply_skin_mask=config.apply_skin_mask,
        return_waveform=has_truth,
        # Never cached, and a stem that clashed with a built clip_id would read
        # someone else's frames.
        cache_dir=None,
    )
    return dataset, segments, row


@torch.no_grad()
def run_windows(
    model: CFMambaPhys, dataset: WindowDataset, batch_size: int = 4,
    workers: int = 4, device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Predicted BVP per window, shape (n_windows, n_frames), and the truth.

    Same forward pass as `evaluate`. `shuffle=False` matters: these windows are
    about to be laid end to end in time. The second return is the contact PPG over
    the same windows when the dataset was built with one, otherwise None.
    """
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
        prefetch_factor=2 if workers else None,
    )
    windows: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for batch in loader:
        # train=False fixes the resampling factor at 1.0, so a rate read at
        # TARGET_FPS is a real rate. `evaluate` has to undo the stretch; check
        # rather than inherit the assumption.
        scale = batch["fps_scale"].numpy()
        if not np.allclose(scale, 1.0):
            raise RuntimeError(
                f"expected fps_scale 1.0 from an evaluation loader, got {scale}. "
                "Every bpm below would be wrong by that factor."
            )
        frames = batch["frames"].to(device, non_blocking=True)
        skin = batch["skin"].to(device, non_blocking=True)
        with torch.autocast(device, dtype=torch.bfloat16):
            predicted = model(frames, skin)
        windows.append(predicted.float().cpu().numpy())
        if "wave" in batch:
            truth.append(batch["wave"].numpy())
    return (
        np.concatenate(windows, axis=0),
        np.concatenate(truth, axis=0) if truth else None,
    )


def stitch(windows: np.ndarray) -> np.ndarray:
    """Lay the windows end to end, z-scoring each one first.

    Nothing in training fixed the scale: Eq. 19's temporal term is negative
    Pearson, which is invariant to a positive scale factor, so windows can come
    back at different amplitudes for the same loss. The sign is not invariant --
    Pearson fixes it -- so no sign alignment is needed.

    Seams are left where they fall and drawn in the plot.
    """
    out = np.empty_like(windows, dtype=np.float64)
    for i, window in enumerate(windows):
        spread = window.std()
        out[i] = (window - window.mean()) / spread if spread > 1e-8 else 0.0
    return out.reshape(-1)


def beats(trace: np.ndarray, fps: float = TARGET_FPS) -> np.ndarray:
    """Peak indices, one per cardiac cycle.

    Minimum spacing is the period of 2.5 Hz, the top of the reporting band, so two
    peaks cannot fall inside one beat. `trace` is expected band-passed.
    """
    trace = np.asarray(trace, dtype=np.float64).ravel()
    if trace.size < 8 or not np.isfinite(trace).all():
        return np.empty(0, dtype=int)
    found, _ = sps.find_peaks(trace, distance=max(1, int(fps / HIGH_HZ)))
    return found


def _refine(trace: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """Sub-sample peak positions, by the vertex of a parabola through 3 points.

    An integer index quantises the interval by one frame, which at 30 fps is 3.4
    bpm at 110 and 6 bpm at 150 -- coarser than the difference this number exists
    to show. The band-pass leaves a smooth trace, so the three samples around a
    peak locate its vertex.
    """
    if len(peaks) == 0:
        return peaks.astype(np.float64)
    inner = peaks[(peaks > 0) & (peaks < len(trace) - 1)]
    if len(inner) != len(peaks):
        return peaks.astype(np.float64)
    left, mid, right = trace[peaks - 1], trace[peaks], trace[peaks + 1]
    denominator = left - 2 * mid + right
    shift = np.where(np.abs(denominator) > 1e-12, 0.5 * (left - right) / denominator, 0.0)
    return peaks + np.clip(shift, -0.5, 0.5)


def analyse(
    windows: np.ndarray, fps: float = TARGET_FPS, truth: np.ndarray | None = None,
) -> dict:
    """Everything reportable from a set of predicted windows.

    Per-window rates are read off the windows themselves, before the stitch, so
    their spread is independent of it.

    `truth` is the contact PPG over the same windows. It goes through the identical
    stitch and band-pass -- its amplitude is no more a measurement than the
    prediction's -- so the two traces are comparable sample for sample.
    """
    trace = stitch(windows)
    filtered = bandpass(trace, fps)
    peaks = beats(filtered, fps)
    # Median interval rather than mean: one missed or doubled beat moves a mean of
    # a dozen intervals by several bpm.
    ibi = float(np.median(np.diff(_refine(filtered, peaks)))) / fps if len(peaks) > 1 else float("nan")
    per_window = np.array([heart_rate(w, fps) for w in windows], dtype=np.float64)
    return {
        "trace": filtered,
        "peaks": peaks,
        "bpm_fft": heart_rate(filtered, fps, filtered=True),
        "bpm_beats": 60.0 / ibi,
        "per_window_bpm": per_window,
        "seconds": len(filtered) / fps,
        # Where one window ends and the next begins, in samples.
        "seams": [i * windows.shape[1] for i in range(1, len(windows))],
    } | _truth_terms(truth, fps, filtered)


def _truth_terms(truth: np.ndarray | None, fps: float, predicted: np.ndarray) -> dict:
    """The contact-PPG trace and the error against it, or empty when there is none."""
    if truth is None:
        return {}
    reference = bandpass(stitch(truth), fps)
    bpm_true = heart_rate(reference, fps, filtered=True)
    return {
        "truth": reference,
        "bpm_true": bpm_true,
        "macc": float(macc(predicted, reference)),
    }
