"""Classical rPPG estimators, as the floor a learned model has to clear.

POS (Wang et al., 2017) and CHROM (de Haan and Jeanne, 2013) need no training and
publish 4.08 and 4.06 bpm MAE on UBFC-rPPG in both source papers' tables. They are
the right baseline for a waveform model: a network that cannot beat a projection
onto a fixed chrominance direction has not learned anything a decade-old algorithm
does not already do.

This replaces the predict-the-training-mean baseline the project used before, which
is meaningful for a scalar regression and meaningless for a waveform.

Both are taken from the vendored rPPG-Toolbox rather than reimplemented, so the
numbers are comparable with the tables they appear in.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

_TOOLBOX = Path(__file__).resolve().parents[2] / "tools" / "rPPG-Toolbox"


def _load(module_name: str, relative: str):
    """Load one vendored estimator by file path, with a minimal `utils` shim.

    Neither estimator can be loaded from a bare path the way post_process can,
    because both do `import unsupervised_methods.utils`. That module's top-level
    imports pull in scikit-image and scikit-learn, and **neither is used on this
    code path** -- POS calls only `utils.detrend`, CHROM calls nothing. Rather than
    take on two large dependencies to satisfy dead imports, a stand-in package
    exposing `detrend` and `process_video` is registered first.

    `detrend` is not a reimplementation: it is the same routine, already loaded
    from the toolbox's own post_process module, so the algorithms still run on
    toolbox code and the numbers stay comparable with the tables they appear in.
    """
    # POS calls `np.mat`, an alias NumPy removed in 2.0. `np.asmatrix` is the
    # migration NumPy's own error message prescribes, so restoring the alias to it
    # is exact rather than approximate -- and it keeps the estimator running on
    # upstream code instead of a local fork of a published algorithm. Only an
    # attribute that used to exist is added; nothing here uses `np.mat` otherwise.
    if not hasattr(np, "mat"):
        np.mat = np.asmatrix

    if "unsupervised_methods" not in sys.modules:
        from .postprocess import detrend

        package = types.ModuleType("unsupervised_methods")
        package.__path__ = [str(_TOOLBOX / "unsupervised_methods")]
        utils = types.ModuleType("unsupervised_methods.utils")
        utils.detrend = detrend

        def process_video(frames: np.ndarray) -> np.ndarray:
            """Per-frame spatial mean RGB, shaped (1, 3, T). Verbatim behaviour."""
            rgb = np.asarray([
                np.sum(np.sum(frame, axis=0), axis=0) / (frame.shape[0] * frame.shape[1])
                for frame in frames
            ])
            return np.asarray(rgb.transpose(1, 0).reshape(1, 3, -1))

        utils.process_video = process_video
        package.utils = utils
        methods = types.ModuleType("unsupervised_methods.methods")
        methods.__path__ = [str(_TOOLBOX / "unsupervised_methods" / "methods")]
        sys.modules["unsupervised_methods"] = package
        sys.modules["unsupervised_methods.utils"] = utils
        sys.modules["unsupervised_methods.methods"] = methods

    if module_name in sys.modules:
        return sys.modules[module_name]
    path = _TOOLBOX / "unsupervised_methods" / "methods" / relative
    if not path.exists():
        raise FileNotFoundError(f"vendored estimator not found at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def skin_rgb_trace(frames: torch.Tensor, skin: torch.Tensor | None = None) -> np.ndarray:
    """(T, 3, H, W) -> (T, 1, 1, 3): the per-frame mean RGB over skin pixels.

    Both estimators use nothing but each frame's spatial mean, so reducing to a
    1x1 "video" first is exactly equivalent and orders of magnitude faster. It also
    lets the mean be taken over skin pixels only -- averaging in the background
    would dilute a signal that is already a fraction of a percent.
    """
    array = frames.detach().float().cpu().numpy()
    if skin is not None:
        mask = skin.detach().float().cpu().numpy() > 0.5
        if mask.any():
            # (T, 3, H, W) -> (T, 3, P) over skin pixels, then mean over P.
            trace = array[:, :, mask].mean(axis=2)
        else:
            trace = array.mean(axis=(2, 3))
    else:
        trace = array.mean(axis=(2, 3))
    return trace[:, None, None, :]


def pos(frames: torch.Tensor, skin: torch.Tensor | None = None,
        fps: float = 30.0) -> np.ndarray:
    """Plane-orthogonal-to-skin. Returns an unfiltered pulse estimate."""
    module = _load("unsupervised_methods.methods.POS_WANG", "POS_WANG.py")
    return np.asarray(module.POS_WANG(skin_rgb_trace(frames, skin), fps)).ravel()


def chrom(frames: torch.Tensor, skin: torch.Tensor | None = None,
          fps: float = 30.0) -> np.ndarray:
    """Chrominance-based estimate. Returns an unfiltered pulse estimate."""
    module = _load("unsupervised_methods.methods.CHROME_DEHAAN", "CHROME_DEHAAN.py")
    return np.asarray(module.CHROME_DEHAAN(skin_rgb_trace(frames, skin), fps)).ravel()


METHODS = {"pos": pos, "chrom": chrom}


def run(loader, methods: tuple[str, ...] = ("pos", "chrom"),
        fps: float = 30.0) -> dict[str, list[dict]]:
    """Per-window rows for each classical method, shaped like evaluate.evaluate.

    Scored on exactly the same windows the model is scored on. A baseline computed
    over a different split, or over the clips that happened to decode, is not a
    baseline.
    """
    from .postprocess import compare

    rows: dict[str, list[dict]] = {name: [] for name in methods}
    for batch in loader:
        for i in range(len(batch["frames"])):
            truth = batch["wave"][i].numpy()
            scale = float(batch["fps_scale"][i])
            for name in methods:
                try:
                    estimate = METHODS[name](batch["frames"][i], batch["skin"][i], fps)
                except Exception as error:                  # noqa: BLE001
                    # A classical estimator failing on a degenerate window is
                    # information, not a reason to abandon the whole comparison.
                    rows[name].append({
                        "clip_id": batch["clip_id"][i],
                        "subject_id": batch["subject_id"][i],
                        "source": batch["clip_id"][i].split("/")[0],
                        "hr_pred": float("nan"), "hr_true": float("nan"),
                        "snr": float("nan"), "macc": float("nan"),
                        "error": type(error).__name__,
                    })
                    continue
                if len(estimate) != len(truth):
                    # POS and CHROM window internally and can return a shorter
                    # signal. Compare on the overlap rather than silently padding.
                    length = min(len(estimate), len(truth))
                    estimate, window_truth = estimate[:length], truth[:length]
                else:
                    window_truth = truth
                result = compare(estimate, window_truth, fps=fps)
                for key in ("hr_pred", "hr_true"):
                    result[key] = result[key] * scale
                rows[name].append({
                    "clip_id": batch["clip_id"][i],
                    "subject_id": batch["subject_id"][i],
                    "source": batch["clip_id"][i].split("/")[0],
                    **result,
                })
    return rows
