"""Materialised face-box frames, so training reads 23 MB a window instead of 147.

UBFC-rPPG releases as uncompressed rawvideo AVI. One 640x480 frame is 921,600 bytes
on disk, so decoding a 160-frame window reads 147 MB -- and `read_window` crops the
face box inside the ffmpeg filter graph, which removes the pipe and the numpy
allocation but not the read: ffmpeg still pulls every byte off disk before the crop
filter sees it. The box is about 9% of the frame, so 92% of that read is thrown away.

Measured on this corpus, one epoch of 484 segments moved 71 GB off a LUKS-encrypted
volume to produce 3.7 GB of pixels, and the run was at 790 MB/s for 90 s an epoch
with the CPU at a fifth of its capacity. Storage bandwidth was the limit, not
decode and not the GPU.

So the box is decoded once, at native resolution and native frame rate, into one
uint8 array per clip. **13.5 GB for all 48 UBFC clips**, which fits in page cache
on a 62 GB machine, so after the first epoch the read costs nothing.

Native resolution and native rate, both intentionally:

  resolution  `WindowDataset` crops an 87.5% sub-window and then resamples once,
              INTER_AREA when shrinking. Caching at 128 or 160 would put a
              resample *before* that crop, and the pulse is a 0.1-0.5 LSB change
              -- the arithmetic ARCHITECTURE.md §2 exists to protect. Storing the
              box at source resolution resamples nothing: ffmpeg's crop is exact
              pixel selection.
  frame rate  the HR-balance augmentation decodes at TARGET_FPS*k with k in
              [0.7, 1.4]. A cache fixed at 30 fps would have to resample again to
              reach those, so the k that the augmentation depends on is exactly
              what a 30 fps cache ruins.

## Why this is not exactly equal to ffmpeg, and why that is safe

It cannot be, and the difference was measured rather than waved at.

ffmpeg's `fps=` filter is **pure nearest-frame selection** -- checked frame by
frame against a full decode, every frame it delivers is an exact source frame and
never an interpolation. But it selects with a running running sum, while index
arithmetic rounds each output frame independently. The two therefore disagree on
the duplication boundaries where 28.67 fps is stretched to 30.

What was measured, across clips at 28.67 and 29.78 fps and k in {0.75, 1.0, 1.35}:

  * the **anchor is identical** in every case -- `round(start * fps)` is the frame
    ffmpeg delivers first;
  * every other frame departs by **at most one source frame**, 35 ms at 28.67 fps;
  * the window covers the same span, because both walk fps/(TARGET_FPS*k) source
    frames per output frame.

A one-frame tie-break on a duplicated frame cannot move a 1-2 Hz rate, and
`tests/test_framecache.py` fixes that directly by reading the pulse out of both.
What it gains is a resampler that is explicit, shared by both paths and testable,
instead of one that resides inside a filter graph. `window_indices` is now the
definition, and `WindowDataset` uses it whether the frames came from the cache or
from the decoder.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import polars as pl

from ..paths import BUILD_ROOT

CACHE_DIR = BUILD_ROOT / "frames_cache"

# Frames are stored as a headerless uint8 dump beside a JSON sidecar rather than as
# .npy. The shape is then recoverable from the file size alone, so a build terminated
# part way through leaves a file whose length states how far it got instead of a .npy
# header claiming frames that were never written.
FRAMES_SUFFIX = ".raw"
SIDECAR_SUFFIX = ".json"
PPG_SUFFIX = "_ppg.npz"

# Same timebase every source is resampled onto. Imported from dataset would be
# circular -- dataset imports this module -- so it is defined here and pinned equal
# by tests/test_framecache.py.
TARGET_FPS = 30.0


def slug(clip_id: str) -> str:
    """`ubfc/DATASET_2_subject13` -> `ubfc__DATASET_2_subject13`.

    The same scheme `src/model/clips.py` already uses for build/masks, so the two
    caches sort together and a clip is recognisable in either from its filename.
    """
    return clip_id.replace("/", "__")


def window_indices(
    start_s: float, n_frames: int, fps: float, k: float, n_available: int
) -> np.ndarray:
    """Source frame index for each of `n_frames` output frames. See module docstring.

    Anchored on `round(start_s * fps)`, which is the frame ffmpeg's own seek lands
    on, then advancing fps/(TARGET_FPS*k) source frames per output frame so the
    window covers n_frames/(TARGET_FPS*k) seconds whatever the source rate.

    Indices past the end are clamped to the last frame, which repeats it -- exactly
    what `WindowDataset` already does when the decoder returns a short window, so a
    clip near the end of a recording still contributes.
    """
    anchor = round(start_s * fps)
    step = np.arange(n_frames, dtype=np.float64) * fps / (TARGET_FPS * k)
    return np.clip(anchor + np.rint(step).astype(np.int64), 0, max(n_available - 1, 0))


def frames_path(out_dir: Path, clip_id: str) -> Path:
    return Path(out_dir) / f"{slug(clip_id)}{FRAMES_SUFFIX}"


def sidecar_path(out_dir: Path, clip_id: str) -> Path:
    return Path(out_dir) / f"{slug(clip_id)}{SIDECAR_SUFFIX}"


def ppg_path(out_dir: Path, clip_id: str) -> Path:
    return Path(out_dir) / f"{slug(clip_id)}{PPG_SUFFIX}"


def open_clip(out_dir: Path, clip_id: str) -> np.memmap | None:
    """The clip's cached box as a read-only (T, side, side, 3) memmap, or None.

    Read-only and memory-mapped, so twelve dataloader workers share one set of
    pages rather than each holding its own copy, and a fancy-index of 160 frames
    modifies 23 MB instead of reading the whole clip.
    """
    sidecar = sidecar_path(out_dir, clip_id)
    frames = frames_path(out_dir, clip_id)
    if not (sidecar.exists() and frames.exists()):
        return None
    meta = json.loads(sidecar.read_text())
    side, count = int(meta["side"]), int(meta["n_frames"])
    if count <= 0:
        return None
    return np.memmap(frames, dtype=np.uint8, mode="r", shape=(count, side, side, 3))


def open_ppg(out_dir: Path, clip_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """The clip's contact PPG as (times, values), parsed once at build time.

    `load_ppg` reads text with np.loadtxt, and the evaluation loaders are not
    persistent (`src/model/train.py:145`), so their workers re-parsed every dev
    clip on every epoch. This is the same trace in binary.
    """
    path = ppg_path(out_dir, clip_id)
    if not path.exists():
        return None
    with np.load(path) as loaded:
        return loaded["times"], loaded["values"]


def _decode_box(row: dict, destination: Path) -> int:
    """Stream the whole clip's face box to `destination`. Returns the frame count.

    Written straight off the pipe in frame-sized chunks. The whole array would be
    266-450 MB per clip, which fits, but building 48 of them back to back would
    keep giving the allocator a fresh half-gigabyte.
    """
    side = int(row["box_side"])
    x, y = int(row["box_x"]), int(row["box_y"])
    cmd = [
        "ffmpeg", "-v", "error", "-threads", "1", "-an", "-sn", "-dn",
        "-i", str(row["video_path"]), "-map", "0:v:0",
        "-vf", f"crop={side}:{side}:{x}:{y}",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frame_bytes = side * side * 3
    written = 0
    with (
        destination.open("wb") as sink,
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc,
    ):
        assert proc.stdout is not None
        while chunk := proc.stdout.read(frame_bytes):
            if len(chunk) < frame_bytes:
                break                       # trailing partial frame
            sink.write(chunk)
            written += 1
        proc.wait()
    return written


def _cache_ppg(row: dict, out_dir: Path) -> bool:
    from .waveform import load_ppg

    video = Path(row["video_path"])
    loaded = load_ppg(video.parent, video_path=video, fps=row["fps"])
    if loaded is None:
        return False
    times, values = loaded
    np.savez(ppg_path(out_dir, row["clip_id"]), times=times, values=values)
    return True


def build(
    manifest: pl.DataFrame,
    out_dir: Path = CACHE_DIR,
    force: bool = False,
    progress: bool = False,
) -> dict[str, int]:
    """Decode every clip's face box once. Resumable; skips what is already there.

    A clip with no detected box (`box_side` 0) is skipped rather than cached whole:
    the fallback squares the frame itself in `WindowDataset._face_box`, and caching
    a full 640x480 frame would cost eleven times the disk for the one case the
    cache was built to avoid.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"built": 0, "skipped": 0, "no_box": 0, "failed": 0}

    rows = manifest.to_dicts()
    for n, row in enumerate(rows, 1):
        clip_id = row["clip_id"]
        if not int(row["box_side"] or 0):
            counts["no_box"] += 1
            continue
        if not force and open_clip(out_dir, clip_id) is not None:
            counts["skipped"] += 1
            continue

        frames = frames_path(out_dir, clip_id)
        written = _decode_box(row, frames)
        if written == 0:
            frames.unlink(missing_ok=True)
            counts["failed"] += 1
            continue

        # The sidecar is written last and on purpose: `open_clip` needs both files,
        # so a build interrupted mid-decode leaves a .raw with no sidecar, which
        # reads as absent and is rebuilt rather than read short without notice.
        sidecar_path(out_dir, clip_id).write_text(json.dumps({
            "clip_id": clip_id,
            "side": int(row["box_side"]),
            "box": [int(row["box_x"]), int(row["box_y"]), int(row["box_side"])],
            "fps": float(row["fps"]),
            "n_frames": written,
            "video_path": str(row["video_path"]),
        }, indent=1))
        _cache_ppg(row, out_dir)
        counts["built"] += 1
        if progress:
            gb = frames.stat().st_size / 1e9
            print(f"  [{n}/{len(rows)}] {clip_id}  {written} frames  {gb:.2f} GB",
                  flush=True)
    return counts


def total_bytes(manifest: pl.DataFrame) -> int:
    """What `build` will cost on disk, for printing before it runs."""
    return sum(
        int(r["n_frames"]) * int(r["box_side"]) ** 2 * 3
        for r in manifest.to_dicts() if int(r["box_side"] or 0)
    )
