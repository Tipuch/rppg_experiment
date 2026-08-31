"""Rewrite MCD-rPPG's AVI container so seeking stops scanning from frame 0.

MCD's AVIs carry no index. `ffprobe` reports `duration_ts=0`, `nb_frames=N/A` and
`duration=N/A`, so ffmpeg has no way to map a timestamp onto a byte offset and
`-ss` falls back to decoding forward from the start of the file.

Measured on this corpus, that costs ~15 ms per second of video, and it is the
loader's dominant expense -- not the process spawn, and not the decode:

    ffmpeg process spawn      83 ms     5%
    container open            22 ms     1%
    seek                   1,500 ms    90%
    decode 160 frames         60 ms     4%

Keyframes are dense here, one every 0.4 s, so this was never a keyframe problem.
Confirmed by contrast against CLBP-300's indexed H.264 .mov, where seek cost is
flat with depth (5 s 340 ms, 15 s 334 ms, 25 s 333 ms, 31 s 320 ms) against MCD's
80 -> 1,449 ms over the same span.

`-c copy` rewrites the container and copies every video packet unchanged, so the
decoded pixels are unchanged. Measured over six clips spanning all three frame
rates present (24.0, 29.916666, 30.0): frame counts identical, decoded windows
exactly equal across an eight-point grid, and seek flat at 43-48 ms at every depth.
One clip went from 1,449 ms to 44 ms.

**The output must stay AVI.** Remuxing to MP4 or MKV drops three frames without notice on
the 29.9167 fps clips -- 1,200 of 3,600 -- and the loss is mid-stream rather than at
the tail, so every frame after it shifts and the contact-PPG target desynchronises
from the video. MP4 with `-fflags +genpts` drops them too. AVI-to-AVI drops nothing,
because AVI's frame-index model accepts the timestamps that MP4 and MKV reject.
`tests/test_remux.py` fixes that.

The same missing metadata is why `probe` needs its `_duration_from_keyframes`
fallback in `video.py`, which steps through the whole keyframe index at ~650 ms per clip on
first touch in every worker. Indexed files make that branch dead.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import polars as pl

from ..paths import BUILD_ROOT

REMUX_DIR = BUILD_ROOT / "mcd_remux"


def remux_path(out_dir: Path, clip_id: str) -> Path:
    """Where a clip's rewritten container resides.

    Stays `.avi` intentionally -- see the module docstring. The slug matches the
    scheme `src/model/clips.py` and `src/model/framecache.py` already use, so the
    three caches sort together and a clip is recognisable in any of them.
    """
    return Path(out_dir) / f"{clip_id.replace('/', '__')}.avi"


def is_indexed(path: Path) -> bool:
    """Whether ffprobe can read a duration and a frame count from `path`.

    This is the whole point of the remux, so it doubles as the completion test:
    a file that reports these is one ffmpeg can seek in, and a partially written
    file is not.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames,duration_ts",
         "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    if probe.returncode != 0:
        return False
    try:
        doc = json.loads(probe.stdout)
        stream = doc["streams"][0]
    except (json.JSONDecodeError, KeyError, IndexError):
        return False
    frames = stream.get("nb_frames")
    stamps = stream.get("duration_ts")
    duration = doc.get("format", {}).get("duration")
    return (
        frames not in (None, "N/A")
        and stamps not in (None, "N/A", "0", 0)
        and duration not in (None, "N/A")
    )


def remux_clip(source: Path, out_dir: Path, clip_id: str) -> Path | None:
    """Rewrite one clip's container. Returns the output path, or None on failure.

    Written to a temporary name and renamed only once ffmpeg exits cleanly and the
    result probes as indexed, so an interrupted run never leaves a half-written
    file that `build` would mistake for finished work.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = remux_path(out_dir, clip_id)
    partial = destination.with_suffix(".avi.partial")

    # `-f avi` is required: the temporary name ends in .partial, and ffmpeg
    # picks its muxer from the extension, so without it the write fails directly.
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(source),
         "-c", "copy", "-map", "0:v:0", "-f", "avi", str(partial)],
        capture_output=True, check=False,
    )
    if result.returncode != 0 or not is_indexed(partial):
        partial.unlink(missing_ok=True)
        return None
    partial.replace(destination)
    return destination


def build(
    manifest: pl.DataFrame,
    out_dir: Path = REMUX_DIR,
    force: bool = False,
    progress: bool = False,
) -> dict[str, int]:
    """Remux every clip in `manifest`. Resumable; skips what is already indexed.

    Only the container changes, so the manifest's `box_x`, `box_y`, `box_side` and
    the cached SegFace masks all stay valid -- frame geometry is unchanged.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {"remuxed": 0, "skipped": 0, "failed": 0}

    rows = manifest.to_dicts()
    for n, row in enumerate(rows, 1):
        clip_id = row["clip_id"]
        destination = remux_path(out_dir, clip_id)
        if not force and is_indexed(destination):
            counts["skipped"] += 1
            continue
        written = remux_clip(Path(row["video_path"]), out_dir, clip_id)
        if written is None:
            counts["failed"] += 1
            print(f"  FAILED {clip_id}", flush=True)
            continue
        counts["remuxed"] += 1
        if progress and n % 100 == 0:
            done = counts["remuxed"] + counts["skipped"]
            print(f"  [{n}/{len(rows)}] {done} ready, {counts['failed']} failed",
                  flush=True)
    return counts


def rewrite_manifest(
    manifest: pl.DataFrame,
    out_dir: Path = REMUX_DIR,
    source: str = "mcd",
    verify: bool = False,
) -> pl.DataFrame:
    """Point `video_path` at the remuxed copy, for rows that have one.

    Returns a new frame rather than mutating: the original manifest stays valid
    against the original files, so the change is reversible by not using the
    result. Rows whose remux is missing keep their old path and still load.

    **`ppg_video_path` is added, holding the original location.** MCD's contact PPG
    is found *from the video path* -- `load_ppg` requires the video to sit in a
    `video/` directory and reads its `ppg_sync/<stem>.txt` peer. Repointing
    `video_path` alone therefore severs the label from its clip, `load_ppg` returns
    None, and `WindowDataset._waveform` falls back to zeros without raising: every
    tensor keeps its shape and a whole epoch trains against a flat target. That
    happened once here. The column exists so the pixels and the labels can move
    independently.

    Existence is the default test, not `is_indexed`. `remux_clip` renames into
    place only after the result probes as indexed, so a file being present already
    carries that guarantee -- and re-probing costs an ffprobe spawn per clip, which
    on this corpus is 3,600 subprocesses to re-establish what the build just
    checked. `verify=True` forces the slow path for a manifest built elsewhere.
    """
    check = is_indexed if verify else (lambda p: Path(p).exists())
    # Absolute, because every other path in the manifest is and a dataloader worker
    # does not necessarily share the cwd this was written from.
    remapped = [
        str(remux_path(out_dir, row["clip_id"]).resolve())
        if row["source"] == source and check(remux_path(out_dir, row["clip_id"]))
        else row["video_path"]
        for row in manifest.to_dicts()
    ]
    return manifest.with_columns(
        pl.Series("video_path", remapped),
        pl.Series("ppg_video_path", manifest["video_path"].to_list()),
    )
