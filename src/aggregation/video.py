"""Frame decoding via ffmpeg.

cv2.VideoCapture is not used. opencv-python 5.0.0.93 (the pre-release this
project fixes) damages the heap decoding UBFC's rawvideo AVIs -- it opens the
file, reports the right metadata, then fails inside the first read() with
"free(): invalid next size". ffmpeg decodes the same files cleanly.

ffmpeg also does the fps resample and the downscale in one pass, which keeps 4K
sources from ever materialising at full size: a decoded 32 s 4K clip is ~48 GB.
"""

from __future__ import annotations

import functools
import json
import subprocess
from pathlib import Path

import numpy as np

# Working resolution for detection and cropping. Outputs are 128 or 256 square,
# so decoding above this gains no detail and costs memory.
WORK_MAX_SIDE = 640


def _duration_from_keyframes(path: Path) -> float:
    """Approximate duration from the final keyframe, for files with no metadata."""
    # check=False: a file with no keyframe metadata exits non-zero, and an empty
    # result is the answer here rather than an error.
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-skip_frame", "nokey", "-select_streams", "v:0",
            "-show_entries", "frame=pts_time", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True, check=False,
    )
    stamps = [float(v) for v in out.stdout.split() if v and v[0].isdigit()]
    if not stamps:
        return 0.0
    # Add one keyframe interval: the last keyframe is not the last frame.
    gap = stamps[-1] - stamps[-2] if len(stamps) > 1 else 0.0
    return stamps[-1] + gap


# Container metadata never changes, and reading it is expensive: `probe` spawns an
# ffprobe, and on the MCD AVIs -- which carry no usable duration -- it spawns a
# second one that steps through the entire keyframe index. Profiled inside the training
# loader, the pair was 3.38 s of every 5.96 s spent producing 11 segments: 57% of
# the time, to re-derive constants. One entry per clip per worker is enough.
#
# 4096 covers this corpus (3653 clips) in a single worker without eviction.
@functools.lru_cache(maxsize=4096)
def _probe_uncached(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,codec_name,duration",
            "-show_entries", "format=duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    doc = json.loads(out.stdout)
    stream = doc["streams"][0]
    num, _, den = stream["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)

    # The MCD AVIs carry no usable container metadata: duration_ts is 0 and
    # nb_frames is absent, so both the stream and format fields read zero. Falling
    # back to the last keyframe timestamp recovers it in ~0.26 s and lands within
    # 0.2 s of truth, where -count_frames would have to decode the whole file.
    duration = float(
        stream.get("duration") or doc.get("format", {}).get("duration") or 0.0
    )
    if duration <= 0:
        duration = _duration_from_keyframes(path)
    n_frames = int(stream.get("nb_frames") or 0) or int(duration * fps)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "duration_s": duration,
        "n_frames": n_frames,
        "codec": stream.get("codec_name", ""),
    }


def probe(path: Path) -> dict:
    """Container metadata for `path`. Cached per process; see _probe_uncached.

    A fresh dict is returned each call so a caller mutating the result cannot
    poison the cache for everything after it.
    """
    return dict(_probe_uncached(path))


def read_keyframes(
    path: Path, max_frames: int = 24, max_side: int = WORK_MAX_SIDE
) -> np.ndarray | None:
    """Decode only keyframes, evenly subsampled to at most max_frames.

    Face detection and skin segmentation need a spread of frames across the clip,
    not every frame. Decoding the whole stream to sample 24 of them costs 3.15 s per
    MCD recording against 0.30 s for keyframes alone -- a 10x difference, which over
    3605 clips is three hours versus twenty minutes.
    """
    info = probe(path)
    w, h = info["width"], info["height"]
    scale = min(1.0, max_side / max(w, h))
    ow, oh = (int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2
    frame_bytes = ow * oh * 3

    cmd = [
        "ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", str(path),
        "-vf", f"scale={ow}:{oh}", "-fps_mode", "passthrough",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    # check=False: a truncated or illegible clip is handled by the frame count
    # below, which returns None, rather than by raising.
    proc = subprocess.run(cmd, capture_output=True, check=False)
    count = len(proc.stdout) // frame_bytes
    if count == 0:
        return None
    frames = np.frombuffer(
        proc.stdout[: count * frame_bytes], dtype=np.uint8
    ).reshape(count, oh, ow, 3)
    if count <= max_frames:
        return frames.copy()
    picks = np.linspace(0, count - 1, max_frames).astype(int)
    return frames[picks].copy()


def read_window(
    path: Path,
    start_s: float,
    n_frames: int,
    target_fps: float | None = None,
    max_side: int = WORK_MAX_SIDE,
    crop: tuple[int, int, int] | None = None,
    trust_crop: bool = False,
) -> np.ndarray | None:
    """Decode n_frames starting at start_s, optionally resampled to target_fps.

    `crop` is `(x, y, side)` in source pixels, applied inside the filter graph. It
    is not a convenience -- it is the single largest speedup in the loader.

    A face box is about 9% of a 640x480 frame, so decoding the full frame and
    cropping in Python moves eleven times more data than it uses: measured on one
    MCD clip, 4978 MB piped against 257 MB, and 4.14 s against 0.64 s for a whole
    180 s clip. Per segment the win is 0.237 s -> 0.133 s on MCD and 0.150 s ->
    0.069 s on UBFC. The decode itself costs the same either way; what disappears is
    the pipe, the buffer copy and the numpy allocation.

    Cropping is exact pixel selection, so it adds no resampling and forfeits nothing.

    Two things measured and rejected: `-hwaccel cuda` is 3x *slower* than plain
    software decode here (the round trip to the GPU costs more than MPEG-4 SP
    decoding saves), and `-threads 2` is slower still on MCD. Neither is used.

    target_fps normalises heterogeneous sources onto one timebase. Without it a
    fixed frame count means a different span per source -- 160 frames is 2.7 s of
    CLBP-300 at 60 fps but 6.7 s of MCD's 24 fps IriunWebcam -- and TSM's one-frame
    shift covers a different interval in each. The cost is real: 60 -> 30 halves the
    Nyquist limit to 15 Hz, and 24 -> 30 interpolates frames bringing no new
    information. Uniform timebase is worth more than that here, because the model
    cannot otherwise tell how much time a window spans.

    Seeking is cheap on these files: MCD is MPEG-4 with keyframes every 0.4 s, so
    `-ss` before `-i` lands within half a second of the target and decodes forward
    from there. Measured 0.27-0.64 s for a 300-frame window depending on seek depth,
    which is what makes decoding per minibatch viable instead of pre-materialising
    frames to disk.

    `trust_crop` states the caller already knows the box lies inside the frame --
    which a manifest row does, because `src/model/clips.py` derived it from the
    real frames. It then **skips the probe entirely**, and that is not a micro-
    optimisation: a cold `probe` costs 650 ms on an MCD clip against 130 ms to
    decode the window it describes, because those AVIs carry no usable duration and
    the fallback steps through the whole keyframe index in a second subprocess. The LRU
    cache hides it only after a worker has seen the clip, and across 3,600 clips
    and twelve workers it is cold for most of the first epoch.

    Everything probe returns is already in the manifest except width and height,
    and those serve only the clamp above and a downscale that cannot fire while the
    box is smaller than max_side. When the box *is* larger, the flag defers and
    probes anyway rather than emitting a filter graph built on an assumption.
    """
    filters: list[str] = []
    if target_fps is not None:
        filters.append(f"fps={target_fps}")

    fast = trust_crop and crop is not None and max(crop[2], 2) <= max_side
    if fast:
        x, y, side = crop
        side = max(2, side)
        filters.append(f"crop={side}:{side}:{x}:{y}")
        ow = oh = side
    else:
        info = probe(path)
        w, h = info["width"], info["height"]
        if crop is not None:
            x, y, side = crop
            # Clamp into the frame: a box built on a differently-scaled decode would
            # otherwise make ffmpeg fail with an unhelpful filter error.
            side = max(2, min(side, w, h))
            x = max(0, min(x, w - side))
            y = max(0, min(y, h - side))
            filters.append(f"crop={side}:{side}:{x}:{y}")
            w, h = side, side

        # Downscale only what is still too large after cropping. A cropped face box
        # is already well under max_side, so for this corpus this branch does not
        # fire.
        #
        # The even-rounding applies to the *scaled* size only. Rounding down when no
        # scaling happens shaved a pixel, without notice, off every odd-sided crop -- UBFC's
        # 269 px box came back 268 -- which would put the frames and the cached skin
        # mask on subtly different grids.
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            ow, oh = (int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2
            filters.append(f"scale={ow}:{oh}:flags=area")
        else:
            ow, oh = w, h
    frame_bytes = ow * oh * 3

    # flags=area when shrinking. swscale defaults to bicubic, which aliases on a
    # large reduction -- 3840 -> 640 on CLBP-300 is a 6x decimation, and bicubic
    # point-samples rather than integrating, so it turns high-frequency detail into
    # false low-frequency structure. An area filter averages every source pixel
    # that lands in an output pixel, which is what preserves a sub-LSB signal.
    # UBFC and MCD are already 640x480, so for them this is a no-op.
    cmd = [
        "ffmpeg", "-v", "error",
        # One decode thread per process. ffmpeg defaults to -threads 0, meaning one
        # thread per core, so twelve dataloader workers were asking a 16-thread
        # machine for up to 192 decode threads. Parallelism here comes from running
        # many workers, not from threading each of them; measured on a single
        # process, extra threads were slower anyway.
        "-threads", "1",
        # Video only. Without these ffmpeg still demuxes the audio and data streams
        # before discarding them.
        "-an", "-sn", "-dn",
        "-ss", f"{start_s:.3f}", "-i", str(path),
        "-map", "0:v:0",
        "-frames:v", str(n_frames),
        "-vf", ",".join(filters) if filters else "null",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    # One preallocated buffer, filled by readinto. The previous loop did a read(),
    # an np.frombuffer and a reshape per frame -- 160 of each -- and then an
    # np.stack that copied the whole window a second time.
    payload = bytearray(n_frames * frame_bytes)
    view = memoryview(payload)
    filled = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        while filled < len(payload):
            got = proc.stdout.readinto(view[filled:])
            if not got:
                break
            filled += got
        proc.kill()
        proc.wait()

    count = filled // frame_bytes
    if count == 0:
        return None
    return np.frombuffer(
        payload, dtype=np.uint8, count=count * frame_bytes
    ).reshape(count, oh, ow, 3)


def read_frames(
    path: Path, target_fps: float | None = None, max_side: int = WORK_MAX_SIDE
) -> tuple[np.ndarray, int, int] | None:
    """Decode to (T, H, W, 3) uint8 BGR, longest side <= max_side.

    target_fps=None decodes at the source rate, which is the default and what
    every caller should want. Resampling costs exactly the detail this dataset
    exists to capture: CLBP-300 is 60 fps and forcing it to 30 halves the Nyquist
    limit to 15 Hz, discarding the upstroke and dicrotic-notch harmonics that
    carry pulse pressure. MCD's IriunWebcam is 24 fps and forcing it up to 30
    interpolates frames that contain no new information.

    Frames are read off the pipe one at a time rather than buffered whole. A
    180 s clip at working resolution is ~5 GB, and capturing that as a single
    bytes object then slicing it costs a second copy of the same size.
    """
    info = probe(path)
    w, h = info["width"], info["height"]
    scale = min(1.0, max_side / max(w, h))
    ow, oh = (int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2
    frame_bytes = ow * oh * 3

    filters = f"scale={ow}:{oh}"
    if target_fps is not None:
        filters = f"fps={target_fps}," + filters
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", filters,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    frames: list[np.ndarray] = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        assert proc.stdout is not None
        while chunk := proc.stdout.read(frame_bytes):
            if len(chunk) < frame_bytes:
                break          # trailing partial frame
            frames.append(
                np.frombuffer(chunk, dtype=np.uint8).reshape(oh, ow, 3)
            )
        proc.wait()

    if not frames:
        return None
    return np.stack(frames), ow, oh
