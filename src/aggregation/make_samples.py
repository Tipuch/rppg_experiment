"""Render inspection samples: one clip per source, through the full skin pipeline.

Writes to build/samples/: a contact sheet per clip showing the original crop,
the skin mask, and the normalised output, plus the raw frames and the mean-Y
trace, so the numbers can be read as well as the picture.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..paths import BUILD_ROOT
from .face import apply_box, median_face_box
from .skin import median_skin_mask, normalise_brightness
from .video import probe, read_frames

OUT = BUILD_ROOT / "samples"
CROP = 256
CONTACT_FRAMES = 4
DEFAULT_SECONDS = 5.0
DEFAULT_PER_SOURCE = 4


def prepare(video: Path, seconds: float = DEFAULT_SECONDS) -> dict | None:
    """Decode, crop to a square 256 face box, segment skin, normalise brightness.

    Decoded at native fps: resampling would discard the waveform detail the
    downstream model needs.
    """
    info = probe(video)
    got = read_frames(video)
    if got is None:
        return None
    raw, _, _ = got
    raw = raw[: int(seconds * info["fps"])]

    box = median_face_box(list(raw))
    cropped = apply_box(raw, box)
    frames = np.stack(
        [cv2.resize(f, (CROP, CROP), interpolation=cv2.INTER_AREA) for f in cropped]
    )
    rgb = frames[:, :, :, ::-1].copy()          # ffmpeg gives BGR, SegFace wants RGB

    mask = median_skin_mask(rgb)
    normalised, mean_y = normalise_brightness(rgb, mask)
    return {
        "fps": info["fps"],
        "box": box,
        "rgb": rgb,
        "mask": mask,
        "normalised": normalised,
        "mean_y": mean_y,
        "native_box_px": None if box is None else box[2],
    }


def contact_sheet(result: dict) -> np.ndarray:
    """Rows: original / mask / normalised. Columns: frames spread over the clip."""
    idx = np.linspace(0, len(result["rgb"]) - 1, CONTACT_FRAMES).astype(int)
    mask_vis = (result["mask"].astype(np.uint8) * 255)
    mask_rgb = np.dstack([mask_vis] * 3)
    rows = [
        np.hstack([result["rgb"][i] for i in idx]),
        np.hstack([mask_rgb for _ in idx]),
        np.hstack([result["normalised"][i] for i in idx]),
    ]
    return np.vstack(rows)[:, :, ::-1]          # back to BGR for imwrite


def _targets(per_source: int = DEFAULT_PER_SOURCE) -> list[tuple[str, Path]]:
    """Up to `per_source` distinct subjects per corpus.

    Driven by the reader registry rather than a hardcoded list.

    MR-NIRP is absent by construction: it ships no container this can decode, and
    its own ingest already writes a crop per session.
    """
    from ..datasets import REGISTRY, SELF_PREPARED

    out: list[tuple[str, Path]] = []
    for name, reader in REGISTRY.items():
        if name in SELF_PREPARED:
            continue
        found = reader.discover()
        if not found.height:
            continue
        seen: set[str] = set()
        for row in found.iter_rows(named=True):
            if row["subject_id"] in seen:
                continue
            path = Path(row["video_path"])
            if path.exists() and path.stat().st_size > 1_000_000:
                out.append((name, path))
                seen.add(row["subject_id"])
            if len(seen) >= per_source:
                break
    return out


def main(
    per_source: int = DEFAULT_PER_SOURCE, seconds: float = DEFAULT_SECONDS
) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    targets = _targets(per_source)
    if not targets:
        print("no source videos found")
        return 1

    for source, video in targets:
        print(f"{source}: {video.name}", flush=True)
        try:
            result = prepare(video, seconds)
        except Exception as exc:  # noqa: BLE001 - one unrenderable sample must not end the sheet
            print(f"  FAILED {type(exc).__name__}: {exc}")
            continue
        if result is None:
            print("  decode failed")
            continue

        stem = f"{source}__{video.stem}"
        cv2.imwrite(str(OUT / f"{stem}__contact.png"), contact_sheet(result))
        np.save(OUT / f"{stem}__frames.npy", result["normalised"])
        np.save(OUT / f"{stem}__mean_y.npy", result["mean_y"])

        mask, mean_y = result["mask"], result["mean_y"]
        print(f"  fps {result['fps']:.2f} | {len(result['rgb'])} frames | "
              f"face box {result['native_box_px']} px native")
        print(f"  skin {100 * mask.mean():.1f}% of crop | "
              f"mean Y {mean_y.min():.1f}-{mean_y.max():.1f} "
              f"(range {np.ptp(mean_y):.2f}, std {mean_y.std():.2f})")

    print(f"\nsamples in {OUT.resolve()}")
    return 0
