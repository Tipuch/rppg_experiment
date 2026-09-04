"""Face ROI selection.

One box per clip, taken as the median over sampled frames. Per-frame boxes jitter,
and that jitter is motion noise competing with the pulse signal, so the median box
is used rather than a per-frame one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Resolved against the package root, not the process CWD: a relative path makes
# a missing model look like a successful run with no face detected.
_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = _ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.7
# Faces are wider than the useful ROI is tall; pad the box then square it so the
# forehead and cheeks (strongest rPPG regions) survive the crop.
#
# rPPG-Toolbox, which all three source papers used, enlarges by LARGE_BOX_COEF=1.5,
# equivalent to BOX_PAD=0.5 here. 0.25 is a deviation: a tighter crop carries fewer
# non-skin pixels, and this pipeline already has a skin mask and PGA's Gaussian
# prior. Which setting scores better is measurable with POS, which needs no
# training; `pad` is a parameter so that measurement runs without editing this file.
BOX_PAD = 0.25


def _detector(w: int, h: int) -> cv2.FaceDetectorYN:
    return cv2.FaceDetectorYN.create(
        str(MODEL_PATH), "", (w, h), score_threshold=SCORE_THRESHOLD
    )


def median_face_box(
    sample_frames: list[np.ndarray], max_samples: int = 24, pad: float = BOX_PAD
) -> tuple[int, int, int, int] | None:
    """Median (x, y, side, side) square box over sampled BGR frames, or None.

    `pad` enlarges the detected box before squaring: 0.25 gives a 1.25x crop, 0.5
    matches the toolbox's 1.5x.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YuNet model missing at {MODEL_PATH}. Without it every crop invisibly "
            "falls back to a centre square with no face ROI."
        )
    if not sample_frames:
        return None
    step = max(1, len(sample_frames) // max_samples)
    picks = sample_frames[::step][:max_samples]
    h, w = picks[0].shape[:2]
    det = _detector(w, h)
    boxes = []
    for frame in picks:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        ok, faces = det.detect(np.ascontiguousarray(frame))
        if ok is not None and faces is not None and len(faces):
            # Largest detection: these datasets are single-subject.
            boxes.append(max(faces, key=lambda f: f[2] * f[3])[:4])
    if not boxes:
        return None

    x, y, bw, bh = np.median(np.asarray(boxes, dtype=np.float64), axis=0)
    cx, cy = x + bw / 2.0, y + bh / 2.0
    side = max(bw, bh) * (1.0 + pad)
    side = min(side, float(min(h, w)))
    left = int(np.clip(cx - side / 2.0, 0, w - side))
    top = int(np.clip(cy - side / 2.0, 0, h - side))
    return left, top, int(side), int(side)


def apply_box(frames: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """Crop (T, H, W, C) with box, or centre-square if box is None."""
    if box is None:
        _, h, w, _ = frames.shape
        side = min(h, w)
        top, left = (h - side) // 2, (w - side) // 2
        return frames[:, top : top + side, left : left + side, :]
    left, top, bw, bh = box
    return frames[:, top : top + bh, left : left + bw, :]
