"""Skin segmentation and brightness normalisation.

Pipeline per clip:

    YuNet median box -> square crop -> 256x256 -> SegFace skin mask
      -> mean Y over mask -> Y - mean + 128 -> non-skin to black

SegFace runs on sampled frames only and the masks are medianed into one stable
mask per clip. Two reasons: it is far too slow to run per frame in a training
loop (~20-50 ms/frame), and a mask whose area flickers frame to frame makes the
mean-Y trace jump for reasons unrelated to blood volume -- a fake signal at
exactly the frequencies rPPG cares about.

The heavy part is therefore precomputed once and cached as a few KB per clip;
decode and arithmetic stay cheap enough to run per minibatch.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .segface.segface_celeb import SegFaceCeleb

# SegFace's CelebAMask-HQ ordering. This is NOT the standard published order --
# skin is 2 here, not 1 -- so it is transcribed from the repo's inference.py.
CELEBA_LABELS = [
    "background", "neck", "skin", "cloth", "l_ear", "r_ear", "l_brow",
    "r_brow", "l_eye", "r_eye", "nose", "mouth", "l_lip", "u_lip", "hair",
    "eye_g", "hat", "ear_r", "neck_l",
]
# Perfused facial skin. Nose included: it is a large, well-perfused central
# region. Eyes, brows, lips, mouth, hair, glasses, ears, neck and cloth excluded.
SKIN_CLASSES = (CELEBA_LABELS.index("skin"), CELEBA_LABELS.index("nose"))

_ROOT = Path(__file__).resolve().parent.parent.parent
WEIGHTS = _ROOT / "models" / "segface" / "swinb_celeba_256.pt"
INPUT_RES = 256
# ImageNet statistics, matching the torchvision Swin backbone SegFace builds on.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_MODEL: SegFaceCeleb | None = None


def load_model(device: str | None = None) -> tuple[SegFaceCeleb, str]:
    """Load SegFace once and keep it. Weights are ~MIT-licensed swinb_celeba_256."""
    global _MODEL
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _MODEL is None:
        model = SegFaceCeleb(INPUT_RES, "swin_base")
        # weights_only=True refuses to unpickle arbitrary objects. These are
        # third-party weights, so try the safe path first and only fall back
        # if the checkpoint really wraps non-tensor state.
        try:
            state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 - fall back to the colour heuristic if SegFace fails
            state = torch.load(WEIGHTS, map_location="cpu", weights_only=False)
        # SegFace checkpoints wrap the weights as "state_dict_backbone"
        # alongside optimizer/scheduler state, not the usual "state_dict".
        for key in ("state_dict_backbone", "state_dict"):
            if isinstance(state, dict) and key in state:
                state = state[key]
                break
        # Checkpoints from DDP training carry a "module." prefix.
        state = {k.removeprefix("module."): v for k, v in state.items()}
        missing, _unexpected = model.load_state_dict(state, strict=False)
        if len(missing) > 20:
            raise RuntimeError(f"SegFace weights did not fit: {len(missing)} missing")
        model.eval().to(dev)
        _MODEL = model
    return _MODEL, dev


@torch.no_grad()
def segment(frames_rgb: np.ndarray, batch: int = 8) -> np.ndarray:
    """Per-frame class maps for (T, 256, 256, 3) uint8 RGB -> (T, 256, 256) int."""
    model, dev = load_model()
    out = np.empty(frames_rgb.shape[:3], dtype=np.uint8)
    for i in range(0, len(frames_rgb), batch):
        chunk = frames_rgb[i : i + batch].astype(np.float32) / 255.0
        chunk = (chunk - MEAN) / STD
        tensor = torch.from_numpy(chunk).permute(0, 3, 1, 2).to(dev)
        logits = model(tensor, None, None)
        logits = torch.nn.functional.interpolate(
            logits, size=(INPUT_RES, INPUT_RES), mode="bilinear", align_corners=False
        )
        out[i : i + batch] = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
    return out


def median_skin_mask(frames_rgb: np.ndarray, max_samples: int = 16) -> np.ndarray:
    """One stable boolean skin mask per clip: majority vote over sampled frames."""
    step = max(1, len(frames_rgb) // max_samples)
    picks = frames_rgb[::step][:max_samples]
    classes = segment(picks)
    skin = np.isin(classes, SKIN_CLASSES)
    return skin.mean(axis=0) >= 0.5


def normalise_brightness(
    frames_rgb: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Black out non-skin and remove each frame's mean skin luma.

    Returns (frames, mean_y) where mean_y is one float per frame.

    The subtraction is why mean_y must be kept. An rPPG pulse is a small,
    spatially near-uniform brightness change across skin, so removing the
    per-frame spatial mean removes most of the pulse from the pixels. It is not
    lost -- it moves into mean_y. A model fed only the frames will underperform;
    it needs both.

    Y is re-centred to 128 rather than clipped at 0, because Y - mean goes
    negative for half the pixels and clipping would discard that half. Cb and Cr
    are left unchanged: chrominance carries pulse information too.
    """
    out = np.zeros_like(frames_rgb)
    mean_y = np.zeros(len(frames_rgb), dtype=np.float32)
    if not mask.any():
        return out, mean_y

    for i, frame in enumerate(frames_rgb):
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        y = ycrcb[:, :, 0]
        m = float(y[mask].mean())
        mean_y[i] = m
        ycrcb[:, :, 0] = np.clip(y - m + 128.0, 0, 255)
        rgb = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        out[i][mask] = rgb[mask]
    return out, mean_y
